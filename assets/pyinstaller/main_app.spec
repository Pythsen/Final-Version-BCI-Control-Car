# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

ROOT = Path(__file__).resolve().parents[2]

datas = [
    (str(ROOT / 'models' / 'online' / 'v50pro'), 'models/online/v50pro'),
    (str(ROOT / 'outputs' / 'timelines'), 'outputs/timelines'),
]
binaries = []
hiddenimports = []
datas += copy_metadata('mne')
tmp_ret = collect_all('mne')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [str(ROOT / 'src' / 'online' / 'main_app.py')],
    pathex=[str(ROOT / 'src' / 'online')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_app',
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_app',
)
