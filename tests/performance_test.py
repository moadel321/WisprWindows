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
from unittest.mock import patch, MagicMock

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


def collect_metrics(metrics, stop_event, interval=0.5):
    """
    Collect system metrics in a background thread
    
    Args:
        metrics: Dictionary to store metrics
        stop_event: Event to signal thread to stop
        interval: Collection interval in seconds
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    
    while not stop_event.is_set():
        try:
            # Get CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            metrics["cpu"].append(cpu_percent)
            
            # Get memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            metrics["memory"].append(memory_mb)
            
            # Add timestamp
            metrics["timestamp"].append(time.time())
            
            # Sleep
            time.sleep(interval)
        except Exception as e:
            print(f"Error collecting metrics: {str(e)}")
            break


def create_test_audio_file(output_path, duration=3.0, sample_rate=16000):
    """
    Create a test audio file with silence and a tone
    
    Args:
        output_path: Output file path
        duration: Duration of the audio in seconds
        sample_rate: Sample rate of the audio
        
    Returns:
        str: Path to the created audio file
    """
    try:
        import soundfile as sf
        import numpy as np
        
        # Create a simple sine wave
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Add some silence
        audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
        audio[int(0.5 * sample_rate):int(2.5 * sample_rate)] = tone[:int(2 * sample_rate)]
        
        # Save audio file
        sf.write(output_path, audio, sample_rate)
        
        return output_path
    except ImportError:
        # If soundfile is not available, create a dummy file
        with open(output_path, "wb") as f:
            # Write a minimal WAV header
            f.write(b"RIFF")
            f.write((36 + int(duration * sample_rate * 2)).to_bytes(4, byteorder="little"))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write((16).to_bytes(4, byteorder="little"))
            f.write((1).to_bytes(2, byteorder="little"))  # PCM format
            f.write((1).to_bytes(2, byteorder="little"))  # Mono
            f.write((sample_rate).to_bytes(4, byteorder="little"))
            f.write((sample_rate * 2).to_bytes(4, byteorder="little"))
            f.write((2).to_bytes(2, byteorder="little"))
            f.write((16).to_bytes(2, byteorder="little"))
            f.write(b"data")
            f.write((int(duration * sample_rate * 2)).to_bytes(4, byteorder="little"))
            
            # Write dummy audio data (silence)
            for _ in range(int(duration * sample_rate)):
                f.write((0).to_bytes(2, byteorder="little"))
        
        return output_path


def test_whisper_performance():
    """Test Whisper model performance"""
    print("\n=== Whisper Model Performance Test ===")
    
    # Skip this test in CI environment or if specified
    if os.environ.get('CI') or os.environ.get('SKIP_PERF_TESTS', 'false').lower() == 'true':
        print("Skipping Whisper performance test in CI environment")
        return True
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create mock audio file
        audio_file = os.path.join(temp_dir, "test_audio.wav")
        create_test_audio_file(audio_file)
        
        # Create model with mock
        with patch('src.models.faster_whisper_model.download_model', return_value="/mock/path/to/model"):
            with patch('src.models.faster_whisper_model.WhisperModel') as mock_whisper:
                # Set up the mock
                mock_instance = MagicMock()
                mock_segment = MagicMock()
                mock_segment.text = "This is a test transcription."
                mock_instance.transcribe.return_value = iter([mock_segment])
                mock_whisper.return_value = mock_instance
                
                # Create the model
                model = FasterWhisperModel(
                    model_dir=temp_dir,
                    model_name="distil-large-v3",
                    language="en",
                    device="cpu",
                    compute_type="int8"
                )
                
                # Load the model
                model.load_model()
                
                # Set up metrics collector thread
                metrics = {"cpu": [], "memory": [], "timestamp": []}
                stop_event = threading.Event()
                metrics_thread = threading.Thread(
                    target=collect_metrics,
                    args=(metrics, stop_event),
                    daemon=True
                )
                metrics_thread.start()
                
                try:
                    # Perform transcription
                    start_time = time.time()
                    result = model.transcribe(audio_file)
                    end_time = time.time()
                    
                    # Stop metrics collection
                    stop_event.set()
                    metrics_thread.join(timeout=1.0)
                    
                    # Calculate performance metrics
                    elapsed_time = end_time - start_time
                    avg_cpu = np.mean(metrics["cpu"]) if metrics["cpu"] else 0
                    max_cpu = np.max(metrics["cpu"]) if metrics["cpu"] else 0
                    avg_memory = np.mean(metrics["memory"]) if metrics["memory"] else 0
                    max_memory = np.max(metrics["memory"]) if metrics["memory"] else 0
                    
                    # Print results
                    print(f"Transcription time: {elapsed_time:.2f} seconds")
                    print(f"Average CPU usage: {avg_cpu:.1f}%")
                    print(f"Peak CPU usage: {max_cpu:.1f}%")
                    print(f"Average memory usage: {avg_memory:.1f} MB")
                    print(f"Peak memory usage: {max_memory:.1f} MB")
                    
                    if not result["success"]:
                        print(f"⚠️ Transcription failed: {result['error']}")
                    
                    # Test passed if transcription was successful
                    return result["success"]
                    
                except Exception as e:
                    print(f"❌ Error during Whisper performance test: {str(e)}")
                    stop_event.set()
                    return False
                
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
            os.rmdir(temp_dir)


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
        metrics["whisper"] = test_whisper_performance()
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