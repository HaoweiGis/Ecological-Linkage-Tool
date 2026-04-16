#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生态源地预处理工具
===================

功能：
    将生态源地栅格数据进行预处理，生成ELT生态网络分析所需的输入数据
    
输入：
    - EcoSource_Huline1km.tif（生态源地栅格）
    
输出：
    - mspa_core_filled.tif（面部件消除后的源地栅格）
    - mspa_core_center.tif（斑块中心点栅格）
    - 对应的Shapefile文件

作者：Haowei
日期：2024
"""

# 执行时间记录（geobase；InputData -> OutputData；2026-04-16）
# 本脚本：25.48 s

import numpy as np
import os
import json
import geopandas as gpd
from osgeo import gdal, ogr, osr
from shapely.geometry import shape, Polygon, MultiPolygon

def _read_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.int32, copy=False)
    gt = ds.GetGeoTransform()
    proj_wkt = ds.GetProjection()
    nodata = band.GetNoDataValue()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    return arr, gt, proj_wkt, nodata, xsize, ysize

def _polygonize_mask(src_arr, gt, proj_wkt, nodata=None, target_value=1):
    ysize, xsize = src_arr.shape
    mem_driver = gdal.GetDriverByName('MEM')
    src_ds = mem_driver.Create('', xsize, ysize, 1, gdal.GDT_Int32)
    src_ds.SetGeoTransform(gt)
    src_ds.SetProjection(proj_wkt)
    src_band = src_ds.GetRasterBand(1)
    src_band.WriteArray(src_arr)
    if nodata is not None:
        src_band.SetNoDataValue(nodata)

    mask_ds = mem_driver.Create('', xsize, ysize, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(gt)
    mask_ds.SetProjection(proj_wkt)
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.WriteArray((src_arr == int(target_value)).astype(np.uint8, copy=False))

    mem_vec_driver = ogr.GetDriverByName('Memory')
    vec_ds = mem_vec_driver.CreateDataSource('mem')
    srs = None
    if proj_wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj_wkt)
    layer = vec_ds.CreateLayer('polys', srs=srs)
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))

    gdal.Polygonize(src_band, mask_band, layer, 0, options=[])

    geoms = []
    for feat in layer:
        geom_ref = feat.GetGeometryRef()
        if geom_ref is None:
            continue
        geom_json = geom_ref.ExportToJson()
        geoms.append(shape(json.loads(geom_json)))
    return geoms

def _rasterize_shp_to_tif(shp_path, out_tif_path, xsize, ysize, gt, proj_wkt, burn_field):
    drv = gdal.GetDriverByName('GTiff')
    out_ds = drv.Create(
        out_tif_path,
        xsize,
        ysize,
        1,
        gdal.GDT_Int32,
        options=["TILED=YES", "COMPRESS=LZW"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj_wkt)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)
    out_band.Fill(0)

    vec_ds = ogr.Open(shp_path)
    if vec_ds is None:
        raise FileNotFoundError(shp_path)
    layer = vec_ds.GetLayer()
    gdal.RasterizeLayer(out_ds, [1], layer, options=[f"ATTRIBUTE={burn_field}"])

    out_band.FlushCache()
    del vec_ds
    del out_ds

def fill_holes_within_polygons(geometry, all_geometries, min_area):
    """
    填充多边形的内部孔洞，保留嵌套的小斑块
    
    参数:
        geometry: 输入多边形
        all_geometries: 所有多边形列表
        min_area: 最小面积阈值
    """
    def contains_other_polygons(outer, inner_geoms):
        for inner in inner_geoms:
            if outer.contains(inner):
                return True
        return False
    
    if isinstance(geometry, Polygon):
        exterior = geometry.exterior
        interiors = [
            interior for interior in geometry.interiors
            if Polygon(interior).area >= min_area or 
              contains_other_polygons(Polygon(interior), all_geometries)
        ]
        return Polygon(exterior, interiors)
    elif isinstance(geometry, MultiPolygon):
        new_geoms = [fill_holes_within_polygons(poly, all_geometries, min_area) 
                     for poly in geometry.geoms]
        return MultiPolygon(new_geoms)
    else:
        return geometry

def main():
    """主函数"""
    print("="*60)
    print("生态源地预处理工具")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'InputData')
    output_dir = os.path.join(base_dir, 'OutputData')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 输入输出路径
    raster_path = os.path.join(input_dir, 'EcologicalSource.tif')
    polySHPPath = os.path.join(output_dir, 'mspa_core.shp')
    pointSHPPath = os.path.join(output_dir, 'mspa_core_center.shp')
    polyPath = os.path.join(output_dir, 'mspa_core_filled.tif')
    pointPath = os.path.join(output_dir, 'mspa_core_center.tif')
    
    print(f"\n输入文件: {raster_path}")
    print(f"输出目录: {output_dir}")
    
    # 检查输入文件是否存在
    if not os.path.exists(raster_path):
        print(f"\n✗ 错误：输入文件不存在: {raster_path}")
        return
    
    # 加载栅格数据
    print("\n[1/4] 读取栅格数据...")
    raster_data, gt, proj_wkt, nodata, xsize, ysize = _read_raster(raster_path)
    
    print(f"  ✓ 栅格尺寸: {raster_data.shape}")
    print(f"  ✓ 唯一值: {np.unique(raster_data)}")
    
    # 栅格转矢量
    print("\n[2/4] 栅格转矢量面...")
    polygons = _polygonize_mask(raster_data, gt, proj_wkt, nodata=nodata, target_value=1)
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=proj_wkt if proj_wkt else None)
    
    # 坐标系检查与转换
    original_crs = gdf.crs
    if gdf.crs is not None and not gdf.crs.is_projected:
        print("  ⚠ 坐标系未投影，转换到EPSG:3857...")
        gdf = gdf.to_crs("EPSG:3857")
    
    # 计算面积并过滤
    gdf["Area"] = gdf.geometry.area / 1e6  # 平方公里
    original_count = len(gdf)
    
    # 过滤小斑块（面积阈值：10平方公里）
    min_area_km2 = 1
    gdf = gdf[gdf["Area"] >= min_area_km2].copy()
    print(f"  ✓ 过滤小斑块: {original_count} -> {len(gdf)} 个")
    print(f"  ✓ 面积阈值: {min_area_km2} km²")
    
    # 创建SourceID
    gdf["SourceID"] = range(1, len(gdf) + 1)
    
    # 面部件消除
    print("\n[3/4] 面部件消除（孔洞填充）...")
    min_area = gdf.geometry.area.min() * 100
    all_geometries = list(gdf.geometry)
    
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: fill_holes_within_polygons(geom, all_geometries, min_area)
    )
    
    print(f"  ✓ 面部件消除完成")
    
    # 提取中心点
    print("\n[4/4] 提取斑块中心点...")
    centroids = gdf.copy()
    centroids.geometry = gdf.geometry.centroid
    print(f"  ✓ 提取中心点: {len(centroids)} 个")
    
    # 保存矢量文件
    print("\n保存矢量文件...")
    gdf.to_file(polySHPPath)
    print(f"  ✓ 面文件: {polySHPPath}")

    if original_crs is not None and gdf.crs != original_crs:
        gdf_orig = gdf.to_crs(original_crs)
    else:
        gdf_orig = gdf
    centroids_orig = gdf_orig.copy()
    centroids_orig.geometry = gdf_orig.geometry.centroid
    centroids_orig.to_file(pointSHPPath)
    print(f"  ✓ 点文件: {pointSHPPath}")
    
    # 栅格化矢量数据
    print("\n栅格化矢量数据...")
    _rasterize_shp_to_tif(polySHPPath, polyPath, xsize, ysize, gt, proj_wkt, "SourceID")
    print(f"  ✓ 面栅格: {polyPath}")

    _rasterize_shp_to_tif(pointSHPPath, pointPath, xsize, ysize, gt, proj_wkt, "SourceID")
    print(f"  ✓ 点栅格: {pointPath}")
    
    print("\n" + "="*60)
    print("✓ 预处理完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  1. {os.path.basename(polyPath)} - 面部件消除后的源地栅格")
    print(f"  2. {os.path.basename(pointPath)} - 斑块中心点栅格")
    print(f"  3. {os.path.basename(polySHPPath)} - 源地矢量面")
    print(f"  4. {os.path.basename(pointSHPPath)} - 中心点矢量")
    print(f"\n下一步：运行 1_AdjacentCore.py")

if __name__ == '__main__':
    main()
