# Speech-to-Text Productivity Tool for Windows

A Windows desktop application that enhances productivity by converting spoken words into text. The application leverages the Faster Whisper implementation of the Whisper model for high-performance speech recognition, with all processing performed locally on your device to ensure data privacy.

## Features

- Real-time speech-to-text transcription with up to 4x faster processing
- Local processing with no data leaving the device
- Voice activity detection to filter out non-speech sounds
- Precise insertion of transcribed text into the focused text box
- Microphone selection from available devices
- Transcription history with export capability
- Modern, intuitive graphical user interface
- Support for multiple Whisper model sizes (tiny to large-v3)
- Optimized GPU inference with 8-bit quantization support

## System Requirements

- **Operating System:** Windows 10/11
- **Hardware:** 
  - CPU: Modern multi-core processor (Intel Core i5/i7 or AMD Ryzen 5/7 or equivalent)
  - RAM: Minimum 8GB
  - GPU: (Optional) NVIDIA GPU with CUDA support (minimum 4GB VRAM, 8GB recommended)
  - Microphone: Any working microphone (built-in or external)
- **Software:**
  - Python 3.9+ with pip

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/speech-to-text-tool.git
   cd speech-to-text-tool
   ```

2. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Download the Faster Whisper model:**
   ```powershell
   # Create models directory
   mkdir -p ~/whisper-models
   
   # Download the model (choose one):
   # For best accuracy (larger model, but slower on CPU):
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='distil-whisper/distil-large-v3', local_dir='~/whisper-models/distil-large-v3', local_dir_use_symlinks=False)"
   
   # For balanced performance (recommended for CPU):
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-medium', local_dir='~/whisper-models/medium', local_dir_use_symlinks=False)"
   
   # For fastest processing (best for CPU):
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-small', local_dir='~/whisper-models/small', local_dir_use_symlinks=False)"
   ```

4. **Run the application:**
   ```
   # CPU mode (default)
   python run.py --model-path ~/whisper-models/distil-large-v3
   
   # Or with GPU acceleration if you have CUDA 12.x installed:
   python run.py --model-path ~/whisper-models/distil-large-v3 --use-cuda
   ```

### GPU Acceleration Setup

> **Note:** The application runs in CPU mode by default for maximum compatibility. To use GPU acceleration for faster transcription:

1. **Install NVIDIA CUDA Dependencies:**
   - Download and install [NVIDIA CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) (required for latest faster-whisper)
   - Download and install [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) compatible with your CUDA version
   - Add CUDA bin directory to your system PATH environment variable

2. **Run with the GPU flag:**
   ```
   python run.py --use-cuda
   ```
   
3. **Verify GPU usage:**
   - The application will display a message confirming CUDA acceleration is enabled
   - You should see GPU memory usage in Task Manager or `nvidia-smi`
   - Transcription should be significantly faster than CPU mode

## Usage

1. **Start the application:**
   ```
   python run.py
   ```

2. **Configure the model:**
   - Open Settings
   - Set the model directory to your downloaded model path
   - Choose your preferred model size
   - Select compute type:
     - `int8`: Optimized for CPU processing (default and recommended)
     - `float16`: If you've enabled GPU acceleration (see Optional GPU Setup)
     - `int8_float16`: If you've enabled GPU with limited VRAM

3. **Select your microphone** from the dropdown menu.

4. **Click "Start Transcription"** to begin listening.

5. **Focus on the text field** where you want the transcribed text to appear.

6. **Speak clearly** into your microphone. The application will detect your speech, transcribe it, and insert the text into the focused field.

7. **Click "Stop"** when you're finished.

## Command-line Arguments

The `run.py` script supports several command-line arguments:

- `--model-path PATH`: Specify the path to your Whisper model directory
- `--use-cuda`: Enable CUDA GPU acceleration (requires CUDA 12.x)
- `--debug`: Enable debug logging
- `--test`: Run system tests
- `--performance`: Run performance tests

## Testing

Run the comprehensive test suite:
```powershell
# Run system tests
python run.py --test --model-path ~/whisper-models/distil-large-v3

# Run performance tests
python run.py --performance --model-path ~/whisper-models/distil-large-v3

# Test Faster Whisper specifically
python tests/test_faster_whisper.py --model-path ~/whisper-models/distil-large-v3  #--audio-file path/to/test.mp3
```

## Troubleshooting

### Performance Optimization

- **CPU Mode** (Default):
  - Use smaller model sizes for faster processing (small or medium)
  - Increase the CPU threads in settings if you have a multi-core processor
  - Enable VAD filter to reduce processing of non-speech audio
  - Close other CPU-intensive applications while using the app

- **GPU Mode** (If enabled):
  - Try using `int8_float16` compute type if you have limited VRAM
  - Use a smaller model if you're experiencing out-of-memory errors
  - For older GPUs, install compatible ctranslate2:
    ```
    pip install --force-reinstall ctranslate2==3.24.0
    ```

### CUDA Issues and Fix History

The application currently defaults to CPU mode due to incompatibilities between faster-whisper library dependencies and various CUDA versions. Specifically, newer versions of faster-whisper require:

- CUDA 12.x libraries (specifically cublas64_12.dll)
- CTranslate2 compiled against CUDA 12.x
- Matching cuDNN libraries

If you're experiencing the error `Library cublas64_12.dll is not found or cannot be loaded` when trying to use GPU mode, you have the following options:

1. **Update to CUDA 12.x**: Install the latest CUDA 12.x toolkit and compatible cuDNN
2. **Use CPU mode**: Our app now defaults to CPU mode for reliability
3. **Try older faster-whisper**: Install specific versions that work with your CUDA:
   ```
   pip install --force-reinstall faster-whisper==0.9.0 ctranslate2==3.17.1
   ```

### Model Loading Issues

- Check model path in settings is correct
- Verify model files are properly downloaded
- Try downloading a different model size
- Check disk space is sufficient

### Voice Activity Detection Issues

- Adjust VAD sensitivity in settings
- Move closer to the microphone
- Ensure microphone is properly selected and working
- Test in a quiet environment first

## Privacy and Security

This application processes all audio data locally on your device using Faster Whisper. No data is sent to external servers, ensuring your privacy and security.

## License

[Insert your license information here]

## Acknowledgements

- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for the optimized Whisper implementation
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) for the fast inference engine
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection
- [PyQt](https://www.riverbankcomputing.com/software/pyqt/) for the graphical user interface
- [PyWinAuto](https://github.com/pywinauto/pywinauto) for Windows automation 