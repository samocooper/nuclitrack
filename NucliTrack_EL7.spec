# -*- mode: python -*-

block_cipher = None

a = Analysis(['/home/localadmin/Packaging/main.py'],
             pathex=['/home/localadmin/Packaging/'],
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
exe = EXE(pyz, Tree('/home/localadmin/Packaging'),
          a.scripts,
	  a.binaries,
	  a.zipfiles,
	  a.datas,
          name='nuclitrack',
          debug=False,
          strip=False,
          upx=True,
          console=False )
