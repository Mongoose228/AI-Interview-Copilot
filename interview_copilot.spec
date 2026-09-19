# -*- mode: python ; coding: utf-8 -*-
# PyInstaller one-folder build (no torch/NLLB).

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = [("context/example_profile.md", "context")]
hiddenimports = [
    "soundcard",
    "cffi",
    "onnxruntime",
    "faster_whisper",
    "ctranslate2",
    "deepl",
    "openai",
    "httpx",
    "pydantic",
    "pydantic_settings",
]

tmp_ret = collect_all("faster_whisper")
datas += tmp_ret[0]
hiddenimports += tmp_ret[2]

tmp_ret = collect_all("onnxruntime")
datas += tmp_ret[0]
hiddenimports += tmp_ret[2]

a = Analysis(
    ["src/interview_copilot/gui/app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchaudio", "transformers", "tensorflow"],
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
    exclude_binaries=True,
    name="interview-copilot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="interview-copilot",
)
