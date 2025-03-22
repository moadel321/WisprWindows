#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performance tests for the Speech-to-Text application
Measures latency and resource usage
"""

import sys
import os
import time
import logging
import argparse
import tempfile
import traceback
from pathlib import Path
import numpy as np
import soundfile as sf
import psutil
import threading
from datetime import datetime
import torch

# Add parent directory to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.faster_whisper_model import FasterWhisperModel
from src.models.vad_model import SileroVAD
from src.text_insertion.text_inserter import TextInserter


class PerformanceMetrics:
    """Class to track performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.start_time = {}
        self.end_time = {}
        self.durations = {}
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_memory = []
        self.metrics_lock = threading.Lock()
    
    def start_timer(self, name):
        """Start a timer for a specific operation"""
        self.start_time[name] = time.time()
    
    def stop_timer(self, name):
        """Stop a timer for a specific operation"""
        if name in self.start_time:
            self.end_time[name] = time.time()
            duration = self.end_time[name] - self.start_time[name]
            
            with self.metrics_lock:
                if name not in self.durations:
                    self.durations[name] = []
                self.durations[name].append(duration)
    
    def record_system_metrics(self):
        """Record current system metrics"""
        # Record CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Record RAM usage
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        # GPU memory would require additional libraries like pynvml
        # For simplicity, we'll log placeholder values
        gpu_memory = 0
        
        with self.metrics_lock:
            self.cpu_usage.append(cpu_percent)
            self.ram_usage.append(ram_usage)
            self.gpu_memory.append(gpu_memory)
    
    def print_report(self):
        """Print a report of all metrics"""
        print("\n=== Performance Metrics Report ===\n")
        
        # Print latency metrics
        print("Latency Metrics (seconds):")
        print("-" * 40)
        for name, durations in self.durations.items():
            if durations:
                avg = sum(durations) / len(durations)
                min_val = min(durations)
                max_val = max(durations)
                print(f"{name}:")
                print(f"  Average: {avg:.4f}s")
                print(f"  Min: {min_val:.4f}s")
                print(f"  Max: {max_val:.4f}s")
                print(f"  Samples: {len(durations)}")
                print()
        
        # Print resource usage metrics
        print("\nResource Usage Metrics:")
        print("-" * 40)
        
        if self.cpu_usage:
            avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
            max_cpu = max(self.cpu_usage)
            print(f"CPU Usage:")
            print(f"  Average: {avg_cpu:.2f}%")
            print(f"  Peak: {max_cpu:.2f}%")
            print()
        
        if self.ram_usage:
            avg_ram = sum(self.ram_usage) / len(self.ram_usage)
            max_ram = max(self.ram_usage)
            print(f"RAM Usage:")
            print(f"  Average: {avg_ram:.2f} MB")
            print(f"  Peak: {max_ram:.2f} MB")
            print()


def create_test_audio(duration=5, sample_rate=16000):
    """
    Create a test audio file with a tone
    
    Args:
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        str: Path to the temporary audio file
    """
    # Create a simple sine wave
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Generate a 440 Hz tone (A4 note)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    
    # Write the audio to the file
    sf.write(temp_path, audio, sample_rate)
    
    return temp_path


def test_vad_performance(iterations=10):
    """
    Test VAD model performance
    
    Args:
        iterations: Number of test iterations
        
    Returns:
        PerformanceMetrics: Performance metrics
    """
    print("\n=== Testing VAD Performance ===")
    
    metrics = PerformanceMetrics()
    
    try:
        # Create VAD model
        vad_model = SileroVAD()
        
        # Create test audio
        audio_length = 1.0  # 1 second
        audio_path = create_test_audio(duration=audio_length, sample_rate=16000)
        
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Run performance test
        print(f"Running {iterations} iterations...")
        
        # Start system metrics collection thread
        def collect_metrics():
            while not stop_thread.is_set():
                metrics.record_system_metrics()
                time.sleep(0.5)
        
        stop_thread = threading.Event()
        metrics_thread = threading.Thread(target=collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        # Run VAD tests
        for i in range(iterations):
            print(f"\rIteration {i+1}/{iterations}", end="")
            
            # Process audio with VAD
            metrics.start_timer("vad_inference")
            result = vad_model.process_audio(audio)
            metrics.stop_timer("vad_inference")
        
        print("\nVAD performance test completed")
        
        # Stop metrics collection
        stop_thread.set()
        metrics_thread.join()
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
    except Exception as e:
        print(f"\nError during VAD performance test: {str(e)}")
        traceback.print_exc()
    
    return metrics


def test_whisper_performance(model_path, iterations=5):
    """
    Test Whisper model performance
    
    Args:
        model_path: Path to Whisper model directory
        iterations: Number of test iterations
        
    Returns:
        PerformanceMetrics: Performance metrics
    """
    print("\n=== Testing Faster Whisper Model Performance ===")
    
    metrics = PerformanceMetrics()
    
    try:
        # Create Whisper model
        whisper_model = FasterWhisperModel(
            model_dir=model_path,
            model_name="large-v3",
            language="en",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        
        # Load model
        metrics.start_timer("whisper_load")
        result = whisper_model.load_model()
        metrics.stop_timer("whisper_load")
        
        if not result["success"]:
            print(f"Failed to load Faster Whisper model: {result['error']}")
            return metrics
        
        print(f"Model loaded on device: {whisper_model.device} with compute type: {whisper_model.compute_type}")
        
        # Create test audio files of different lengths
        durations = [1, 3, 5]  # seconds
        audio_paths = {}
        
        for duration in durations:
            audio_paths[duration] = create_test_audio(duration=duration)
        
        # Start system metrics collection thread
        def collect_metrics():
            while not stop_thread.is_set():
                metrics.record_system_metrics()
                time.sleep(0.5)
        
        stop_thread = threading.Event()
        metrics_thread = threading.Thread(target=collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        # Run transcription tests for each duration
        for duration in durations:
            audio_path = audio_paths[duration]
            
            print(f"\nTesting {duration}s audio file ({iterations} iterations):")
            
            for i in range(iterations):
                print(f"\rIteration {i+1}/{iterations}", end="")
                
                # Transcribe audio
                metrics.start_timer(f"whisper_inference_{duration}s")
                result = whisper_model.transcribe(audio_path)
                metrics.stop_timer(f"whisper_inference_{duration}s")
                
                if not result["success"]:
                    print(f"\nTranscription failed: {result['error']}")
        
        print("\nWhisper performance test completed")
        
        # Stop metrics collection
        stop_thread.set()
        metrics_thread.join()
        
        # Clean up
        for audio_path in audio_paths.values():
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
    except Exception as e:
        print(f"\nError during Whisper performance test: {str(e)}")
        traceback.print_exc()
    
    return metrics


def test_text_insertion_performance(iterations=20):
    """
    Test text insertion performance
    
    Args:
        iterations: Number of test iterations
        
    Returns:
        PerformanceMetrics: Performance metrics
    """
    print("\n=== Testing Text Insertion Performance ===")
    
    metrics = PerformanceMetrics()
    
    try:
        # Create text inserter
        text_inserter = TextInserter()
        
        print("Please focus a text field (e.g., Notepad, browser input) within 5 seconds...")
        time.sleep(5)
        
        # Check if we have a focused element
        element_info = text_inserter.get_focused_element()
        if not element_info or not element_info['editable']:
            print("No editable text field focused. Skipping test.")
            return metrics
        
        print(f"Found editable element in: {element_info['app']}")
        
        # Generate test texts of different lengths
        short_text = "This is a short test. "
        medium_text = "This is a medium length test with multiple words and sentences. This should simulate typical conversational speech. "
        long_text = "This is a longer test that contains multiple sentences and should simulate a more extended speech segment. " + \
                   "When people speak for longer periods, the text insertion mechanism needs to handle larger chunks of text efficiently. " + \
                   "This test helps us measure the performance impact of inserting larger amounts of text at once. "
        
        test_texts = {
            "short": short_text,
            "medium": medium_text,
            "long": long_text
        }
        
        # Start system metrics collection thread
        def collect_metrics():
            while not stop_thread.is_set():
                metrics.record_system_metrics()
                time.sleep(0.5)
        
        stop_thread = threading.Event()
        metrics_thread = threading.Thread(target=collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        # Run insertion tests for each text length
        for name, text in test_texts.items():
            print(f"\nTesting {name} text insertion ({iterations} iterations):")
            
            for i in range(iterations):
                print(f"\rIteration {i+1}/{iterations}", end="")
                
                # Measure element detection time
                metrics.start_timer(f"element_detection_{name}")
                element_info = text_inserter.get_focused_element()
                metrics.stop_timer(f"element_detection_{name}")
                
                if not element_info or not element_info['editable']:
                    print("\nLost focus on editable element. Please refocus and press Enter to continue...")
                    input()
                    continue
                
                # Measure text insertion time
                metrics.start_timer(f"text_insertion_{name}")
                success = text_inserter.insert_text(text)
                metrics.stop_timer(f"text_insertion_{name}")
                
                if not success:
                    print("\nText insertion failed. Please check the focused element and press Enter to continue...")
                    input()
                
                # Add a small delay to avoid overwhelming the target application
                time.sleep(0.5)
        
        print("\nText insertion performance test completed")
        
        # Stop metrics collection
        stop_thread.set()
        metrics_thread.join()
        
    except Exception as e:
        print(f"\nError during text insertion performance test: {str(e)}")
        traceback.print_exc()
    
    return metrics


def write_report_to_file(metrics, report_filename="performance_report.txt"):
    """
    Write performance metrics to a file
    
    Args:
        metrics: Dictionary of metrics objects
        report_filename: Output filename
    """
    with open(report_filename, 'w') as f:
        f.write("Speech-to-Text Performance Test Report\n")
        f.write("======================================\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for test_name, metric in metrics.items():
            f.write(f"\n{test_name.upper()} PERFORMANCE\n")
            f.write("=" * len(test_name) + "============\n\n")
            
            # Write latency metrics
            f.write("Latency Metrics (seconds):\n")
            f.write("-" * 40 + "\n")
            for name, durations in metric.durations.items():
                if durations:
                    avg = sum(durations) / len(durations)
                    min_val = min(durations)
                    max_val = max(durations)
                    f.write(f"{name}:\n")
                    f.write(f"  Average: {avg:.4f}s\n")
                    f.write(f"  Min: {min_val:.4f}s\n")
                    f.write(f"  Max: {max_val:.4f}s\n")
                    f.write(f"  Samples: {len(durations)}\n\n")
            
            # Write resource usage metrics
            f.write("\nResource Usage Metrics:\n")
            f.write("-" * 40 + "\n")
            
            if metric.cpu_usage:
                avg_cpu = sum(metric.cpu_usage) / len(metric.cpu_usage)
                max_cpu = max(metric.cpu_usage)
                f.write(f"CPU Usage:\n")
                f.write(f"  Average: {avg_cpu:.2f}%\n")
                f.write(f"  Peak: {max_cpu:.2f}%\n\n")
            
            if metric.ram_usage:
                avg_ram = sum(metric.ram_usage) / len(metric.ram_usage)
                max_ram = max(metric.ram_usage)
                f.write(f"RAM Usage:\n")
                f.write(f"  Average: {avg_ram:.2f} MB\n")
                f.write(f"  Peak: {max_ram:.2f} MB\n\n")
    
    print(f"\nPerformance report written to: {report_filename}")


def run_performance_tests(model_path, output_file=None):
    """
    Run all performance tests
    
    Args:
        model_path: Path to Whisper model directory
        output_file: Path to output file for detailed metrics
    """
    # Configure logging
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Speech-to-Text Performance Tests ===")
    print("Testing latency and resource usage for critical components")
    
    metrics = {}
    
    # Test VAD performance
    try:
        print("\nRunning VAD performance test...")
        metrics["vad"] = test_vad_performance()
        metrics["vad"].print_report()
    except Exception as e:
        print(f"Error during VAD performance test: {str(e)}")
    
    # Test Whisper model performance
    try:
        print("\nRunning Whisper model performance test...")
        metrics["whisper"] = test_whisper_performance(model_path)
        metrics["whisper"].print_report()
    except Exception as e:
        print(f"Error during Whisper performance test: {str(e)}")
    
    # Test text insertion performance
    try:
        print("\nRunning text insertion performance test...")
        metrics["text_insertion"] = test_text_insertion_performance()
        metrics["text_insertion"].print_report()
    except Exception as e:
        print(f"Error during text insertion performance test: {str(e)}")
    
    # Write report to file if requested
    if output_file:
        write_report_to_file(metrics, output_file)
    
    print("\n=== Performance Testing Complete ===")


def main(model_path=None, output_file=None):
    """Main function"""
    if model_path is None:
        # Only parse arguments if called directly
        parser = argparse.ArgumentParser(description="Run performance tests for the Speech-to-Text application")
        parser.add_argument(
            "--model-path",
            type=str,
            help="Path to Whisper model directory",
            default=os.path.expanduser("~/whisper-models")
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Path to output file for detailed metrics",
            default="performance_report.txt"
        )
        
        args = parser.parse_args()
        model_path = args.model_path
        output_file = args.output
    
    # Make sure model path exists
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        print("Please specify a valid model directory with --model-path")
        return 1
    
    run_performance_tests(model_path, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main()) 