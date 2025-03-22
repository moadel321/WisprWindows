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
  - GPU: NVIDIA GPU with CUDA support (minimum 4GB VRAM, 8GB recommended)
  - Microphone: Any working microphone (built-in or external)
- **Software:**
  - Python 3.9+ with pip
  - NVIDIA CUDA Toolkit 11.8 or later
  - NVIDIA cuDNN 8.9.2 or later

## Installation

1. **Install NVIDIA CUDA Dependencies:**
   - Download and install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Download and install [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Add CUDA paths to your system environment variables

2. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/speech-to-text-tool.git
   cd speech-to-text-tool
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Download the Faster Whisper model:**
   ```powershell
   # Create models directory
   mkdir -p ~/whisper-models
   
   # Download the model (choose one):
   # For best accuracy (recommended if you have 8GB+ VRAM):
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='distil-whisper/distil-large-v3', local_dir='~/whisper-models/distil-large-v3', local_dir_use_symlinks=False)"
   
   # For balanced performance (4GB+ VRAM):
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-medium', local_dir='~/whisper-models/medium', local_dir_use_symlinks=False)"
   
   # For fastest processing (2GB+ VRAM):
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-small', local_dir='~/whisper-models/small', local_dir_use_symlinks=False)"
   ```

5. **Run the application:**
   ```
   python run.py --model-path ~/whisper-models/distil-large-v3
   ```

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
     - `float16`: Best for modern GPUs with sufficient VRAM
     - `int8_float16`: Optimized for GPUs with limited VRAM
     - `int8`: Best for CPU-only systems

3. **Select your microphone** from the dropdown menu.

4. **Click "Start Transcription"** to begin listening.

5. **Focus on the text field** where you want the transcribed text to appear.

6. **Speak clearly** into your microphone. The application will detect your speech, transcribe it, and insert the text into the focused field.

7. **Click "Stop"** when you're finished.

## Command-line Arguments

The `run.py` script supports several command-line arguments:

- `--model-path PATH`: Specify the path to your Whisper model directory
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
python tests/test_faster_whisper.py --model-path ~/whisper-models/distil-large-v3 --audio-file path/to/test.mp3
```

## Troubleshooting

### Model Loading Issues

- Ensure CUDA Toolkit and cuDNN are properly installed
- Try using `int8_float16` compute type if you have limited VRAM
- For older GPUs, install compatible ctranslate2:
  ```
  pip install --force-reinstall ctranslate2==3.24.0
  ```

### CUDA Issues

- Verify CUDA installation: `nvidia-smi` should show your GPU
- Check CUDA version compatibility with `torch.cuda.is_available()`
- Update NVIDIA drivers to the latest version

### Performance Issues

- Try different compute types in Settings
- Use a smaller model if you have limited VRAM
- Enable VAD filter to reduce processing of non-speech audio

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