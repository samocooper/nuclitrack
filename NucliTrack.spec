# -*- mode: python -*-

from kivy.deps import sdl2, glew

block_cipher = None


a = Analysis(['C:\\Users\\adminsam\\nuclitrack\\main.py'],
             pathex=['C:\\Users\\adminsam\\NucliTrackApp'],
             binaries=[],
             datas=[],
             hiddenimports=['h5py.defs','h5py.utils','h5py.h5ac','h5py._proxy','h5py._errors','sklearn.neighbors.typedefs','sklearn.tree._utils','win32timezone'],
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
          name='NucliTrack',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,Tree('C:\\Users\\adminsam\\nuclitrack\\'),
               a.binaries,
               a.zipfiles,
               a.datas,
		*[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               name='NucliTrack')
