# -*- mode: python -*-

block_cipher = None
from kivy.tools.packaging.pyinstaller_hooks import get_deps_all, hookspath, runtime_hooks

a = Analysis(['/Users/samcooper/Packaging/main.py'],
             pathex=['/Users/samcooper/Packaging/'],
             binaries=[],
             datas=[],
             hiddenimports=['h5py.defs','h5py.utils','h5py.h5ac','h5py._proxy','h5py._errors','sklearn.neighbors.typedefs','sklearn.tree._utils','win32timezone','pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='nuclitrack',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe, Tree('/Users/samcooper/Packaging/'),
               Tree('/Library/Frameworks/SDL2_ttf.framework/Versions/A/Frameworks/FreeType.framework'),
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='nuclitrack')
app = BUNDLE(coll,
             name='NucliTrack.app',
             icon=None,
         bundle_identifier=None)
