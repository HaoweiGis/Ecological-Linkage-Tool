#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 0：生态源地预处理
======================
输入 : EcologicalSource.tif（值=1 为生态源地）
输出 :
    mspa_core_filled.tif  — 孔洞填充后的源地栅格（斑块 ID）
    mspa_core.shp         — 源地面矢量
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
from shapely.geometry import shape, Polygon, MultiPolygon
import geopandas as gpd

from utils import write_raster


# ──────────────────────────────────────────────
# 几何处理
# ──────────────────────────────────────────────

def _contains_other(outer_poly, inner_geoms):
    """判断 outer_poly 是否包含 inner_geoms 中的任意一个多边形。"""
    return any(outer_poly.contains(g) for g in inner_geoms)


def fill_holes(geometry, all_geometries, min_hole_area):
    """
    填充多边形内部的小孔洞。
    保留面积 >= min_hole_area 的孔洞，以及包含其他斑块的孔洞。

    参数:
        geometry       : Polygon 或 MultiPolygon
        all_geometries : 所有斑块几何列表（用于检测嵌套斑块）
        min_hole_area  : 孔洞面积阈值（m²），小于此值的孔洞被填充
    """
    if isinstance(geometry, Polygon):
        kept_rings = [
            ring for ring in geometry.interiors
            if Polygon(ring).area >= min_hole_area
            or _contains_other(Polygon(ring), all_geometries)
        ]
        return Polygon(geometry.exterior, kept_rings)

    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([
            fill_holes(part, all_geometries, min_hole_area)
            for part in geometry.geoms
        ])

    return geometry


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def run_preprocess(eco_source_path, output_dir, min_area_km2=1.0, log_callback=print):
    """
    执行预处理。

    参数:
        eco_source_path : 输入源地栅格路径
        output_dir      : 输出目录
        min_area_km2    : 斑块最小保留面积（km²），默认 1.0
        log_callback    : 日志回调函数
    """
    log = log_callback
    log("=" * 55)
    log("步骤 0: 生态源地预处理")
    log("=" * 55)

    # 输出路径
    poly_shp   = os.path.join(output_dir, 'mspa_core.shp')
    poly_tif   = os.path.join(output_dir, 'mspa_core_filled.tif')

    # ── 1. 读取栅格 ─────────────────────────────
    log("[1/4] 读取栅格数据...")
    with rasterio.open(eco_source_path) as src:
        raster    = src.read(1).astype('int32')
        transform = src.transform
        crs       = src.crs

    unique_vals = np.unique(raster)
    log(f"  栅格尺寸 : {raster.shape}")
    log(f"  唯一值   : {unique_vals}")

    # ── 2. 栅格转矢量 ───────────────────────────
    log("[2/4] 栅格转矢量面...")
    mask   = (raster == 1)
    geoms  = [shape(g) for g, _ in shapes(raster, mask=mask, transform=transform)]
    gdf    = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    # 投影检查 — 面积计算需要投影坐标系
    original_crs = crs
    if not gdf.crs.is_projected:
        log("  坐标系为地理坐标，转换至 EPSG:3857 以计算面积...")
        gdf = gdf.to_crs("EPSG:3857")

    # ── 3. 面积过滤 ─────────────────────────────
    gdf['Area'] = gdf.geometry.area / 1e6          # 转换为 km²
    n_before    = len(gdf)
    gdf         = gdf[gdf['Area'] >= min_area_km2].copy().reset_index(drop=True)
    gdf['SourceID'] = range(1, len(gdf) + 1)

    log(f"  斑块过滤 : {n_before} → {len(gdf)} 个  (面积阈值 {min_area_km2} km²)")

    # ── 4. 孔洞填充 ─────────────────────────────
    log("[3/4] 孔洞填充（保留包含其他斑块的内环）...")
    # 以最小斑块面积的 100 倍作为孔洞保留阈值
    min_hole_area = gdf.geometry.area.min() * 100
    all_geoms     = list(gdf.geometry)

    gdf['geometry'] = gdf.geometry.apply(
        lambda g: fill_holes(g, all_geoms, min_hole_area)
    )
    log(f"  孔洞面积阈值 : {min_hole_area / 1e6:.4f} km²")

    # ── 5. 保存矢量文件 ─────────────────────────
    log("保存矢量文件...")
    gdf.to_file(poly_shp)
    log(f"  ✓ 面矢量   : {poly_shp}")

    # ── 6. 栅格化（转回原始 CRS）──────────────
    log("栅格化矢量数据...")
    if gdf.crs != original_crs:
        gdf_orig       = gdf.to_crs(original_crs)
    else:
        gdf_orig       = gdf

    def _rasterize(df):
        pairs = zip(df.geometry, df['SourceID'])
        return rasterize(
            ((geom, val) for geom, val in pairs),
            out_shape  = raster.shape,
            transform  = transform,
            fill       = 0,
            dtype      = 'int32',
        )

    poly_arr  = _rasterize(gdf_orig).astype('int32')

    write_raster(poly_tif,  poly_arr,  transform, original_crs, nodata=0)
    log(f"  ✓ 源地栅格 : {poly_tif}")

    log("─" * 55)
    log(f"步骤 0 完成，共 {len(gdf)} 个斑块。")
    log("下一步: 步骤 1（邻接关系计算）")
