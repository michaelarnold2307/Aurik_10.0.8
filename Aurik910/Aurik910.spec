# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Specification for AURIK 9.0
Packages the application as a standalone executable
"""

block_cipher = None

# All Python dependencies that need to be included
hiddenimports = [
    'scipy',
    'scipy.signal',
    'scipy.fft',
    'numpy',
    'soundfile',
    'librosa',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',
    'backend',
    'backend.core',
    'denker',
    'denker.aurik_denker',
    'denker.defekt_denker',
    'denker.exzellenz_denker',
    'denker.rekonstruktions_denker',
    'denker.reparatur_denker',
    'denker.restaurier_denker',
    'denker.strategie_denker',
    'denker.tontraeger_denker',
    'denker.tontraegerkette_denker',
    'shared',
    'shared.enums',
    'dsp',
    'dsp.bass_enhancement',
    'dsp.drums_enhancement',
    'dsp.guitar_enhancement',
    'dsp.piano_restoration',
    'dsp.brass_enhancement',
    'dsp.spatial_enhancement',
]

# Data files to include (presets, configs, etc.)
datas = [
    ('Aurik910/resources/*', 'resources'),
]

a = Analysis(
    ['Aurik910/main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'tests',
        'pytest',
        'setuptools',
    ],
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
    name='AURIK910',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Aurik910/resources/icon.ico' if os.path.exists('Aurik910/resources/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AURIK910',
)

# macOS App Bundle (optional)
app = BUNDLE(
    coll,
    name='AURIK 9.10.45.app',
    icon='Aurik910/resources/icon.icns' if os.path.exists('Aurik910/resources/icon.icns') else None,
    bundle_identifier='com.aurik.90',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleName': 'AURIK 9',
        'CFBundleDisplayName': 'AURIK 9',
        'CFBundleVersion': '9.10.45',
        'CFBundleShortVersionString': '9.10.45',
        'NSHumanReadableCopyright': '© 2026 AURIK Team',
    },
)
