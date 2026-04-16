# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import importlib.util
import os

def _package_dir(package_name: str):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return list(spec.submodule_search_locations)[0]
    if spec.origin:
        return os.path.dirname(spec.origin)
    return None


def _site_packages_dir(package_name: str):
    pkg_dir = _package_dir(package_name)
    if not pkg_dir:
        return None
    return os.path.dirname(pkg_dir)


def _maybe_datas_for_sibling_dir(package_name: str, sibling_dir_name: str):
    site_pkgs = _site_packages_dir(package_name)
    if not site_pkgs:
        return []
    src = os.path.join(site_pkgs, sibling_dir_name)
    if os.path.isdir(src):
        return [(src, sibling_dir_name)]
    return []


datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('rasterio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyproj')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fiona')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('shapely')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

datas += _maybe_datas_for_sibling_dir('rasterio', 'rasterio.libs')
datas += _maybe_datas_for_sibling_dir('fiona', 'fiona.libs')
datas += _maybe_datas_for_sibling_dir('pyproj', 'pyproj.libs')
datas += _maybe_datas_for_sibling_dir('shapely', 'shapely.libs')


a = Analysis(
    ['main.py'],
    pathex=[],
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
    a.binaries,
    a.datas,
    [],
    name='ELT_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
