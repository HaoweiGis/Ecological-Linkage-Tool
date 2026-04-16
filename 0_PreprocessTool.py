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
import rasterio
import geopandas as gpd
from rasterio.features import shapes, rasterize
from shapely.geometry import shape, Polygon, MultiPolygon

def rasterize_shapefile(gdf, transform, out_shape, output_file, crs):
    """矢量数据栅格化"""
    shapes_iter = ((geom, value) for geom, value in zip(gdf.geometry, gdf["SourceID"]))
    rasterized = rasterize(
        shapes_iter,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='int32'
    )

    with rasterio.open(
        output_file, "w",
        driver="GTiff",
        height=out_shape[0],
        width=out_shape[1],
        count=1,
        dtype="int32",
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(rasterized, 1)

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
    
    base_dir = r'D:\2_HaoweiPapers\1_SOCIAndEco\Ecological-Linkage-Tool-main'
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
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1).astype("int32")
        transform = src.transform
        crs = src.crs
    
    print(f"  ✓ 栅格尺寸: {raster_data.shape}")
    print(f"  ✓ 唯一值: {np.unique(raster_data)}")
    
    # 栅格转矢量
    print("\n[2/4] 栅格转矢量面...")
    mask = raster_data == 1
    shapes_gen = shapes(raster_data, mask=mask, transform=transform)
    
    polygons = [shape(geom) for geom, value in shapes_gen]
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    
    # 坐标系检查与转换
    if not gdf.crs.is_projected:
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
    
    centroids.to_file(pointSHPPath)
    print(f"  ✓ 点文件: {pointSHPPath}")
    
    # 栅格化矢量数据
    print("\n栅格化矢量数据...")
    rasterize_shapefile(gdf, transform, raster_data.shape, polyPath, crs)
    print(f"  ✓ 面栅格: {polyPath}")
    
    rasterize_shapefile(centroids, transform, raster_data.shape, pointPath, crs)
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
