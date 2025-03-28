# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[
    ('src/resources/*', 'resources'),
    ('docs/*', 'docs'),
    ('README.md', '.'),
    ('CLAUDE.md', '.'),
    ('C:\Users\mo/.cache/torch/hub/snakers4_silero-vad_master/*', 'models/silero-vad'),
],
    hiddenimports=[
    'torch',
    'numpy',
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'faster_whisper',
    'ctranslate2',
    'pywinauto',
    'win32clipboard',
    'sentencepiece',
    'pyaudio',
    'wave',
    'soundfile',
    'librosa',
    'win32api',
    'win32con',
    'win32gui',
    'ctypes',
    'huggingface_hub',
],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)




exe = EXE(
    pyz,
    a.scripts,
    
    
    
    [],
    name='SpeechToText',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="{icon_path}",
)

coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, upx_exclude=[], name="SpeechToText")
