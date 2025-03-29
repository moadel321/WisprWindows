# test_cuda_env.py
import os
import sys
print(f"--- Start test_cuda_env.py ---")
print(f"Running Python: {sys.executable}") # Verify venv python
os.environ["STT_USE_CUDA"] = "1" # Simulate run.py flag
print(f"STT_USE_CUDA env var: {os.environ.get('STT_USE_CUDA')}")

print("\n--- Checking PATH ---")
print("Sys Path:")
# Limit output for brevity
for p in sys.path[:5] + sys.path[-5:]: print(f"  {p}")
print("OS Path:")
# Limit output for brevity
path_parts = os.environ.get("PATH","").split(os.pathsep)
for p in path_parts[:5] + path_parts[-5:]: print(f"  {p}")

try:
    print("\n--- Importing torch ---")
    import torch
    print(f"Torch version: {torch.__version__}")
    is_available_after_torch = torch.cuda.is_available()
    print(f"CUDA available after torch import: {is_available_after_torch}")
    if not is_available_after_torch:
        print("!!! CUDA became unavailable just after importing torch !!!")

    # --- Add other imports one by one below ---

    # print("\n--- Importing ctranslate2 (via faster_whisper import) ---")
    # from faster_whisper import WhisperModel # This implicitly imports ctranslate2
    # is_available_after_fw = torch.cuda.is_available()
    # print(f"CUDA available after faster_whisper import: {is_available_after_fw}")
    # if is_available_after_torch and not is_available_after_fw:
    #     print("!!! CUDA became unavailable after importing faster_whisper/ctranslate2 !!!")

    # print("\n--- Importing PyQt6 ---")
    # from PyQt6.QtCore import QCoreApplication # Import a core component
    # is_available_after_pyqt = torch.cuda.is_available()
    # print(f"CUDA available after PyQt6 import: {is_available_after_pyqt}")
    # if is_available_after_fw and not is_available_after_pyqt: # Check against previous step
    #     print("!!! CUDA became unavailable after importing PyQt6 !!!")

    # Add checks for PyAudio, onnxruntime etc. similarly if needed

except Exception as e:
    print(f"\n--- ERROR during import checks ---")
    import traceback
    traceback.print_exc()

print("\n--- End test_cuda_env.py ---")