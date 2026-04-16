#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享工具函数
============
栅格读写、像素坐标转地理坐标
"""

import os
import numpy as np
import rasterio
from rasterio import transform as rio_transform
from pyproj import Transformer


def read_raster(path):
    """
    读取单波段栅格文件。

    返回:
        data      : 2D numpy array
        transform : rasterio 仿射变换
        crs       : 坐标参考系
        nodata    : NoData 值（可能为 None）
    """
    with rasterio.open(path) as src:
        data = src.read(1)
        return data, src.transform, src.crs, src.nodata


def write_raster(path, data, transform, crs, nodata=0):
    """
    将 2D numpy array 写入单波段 GeoTIFF。
    自动根据 dtype 确定存储类型，启用 LZW 压缩。
    """
    dtype = data.dtype
    # rasterio 不支持 int64，降级为 int32
    if dtype == np.int64:
        data = data.astype(np.int32)
        dtype = np.int32
    elif dtype == np.uint64:
        data = data.astype(np.uint32)
        dtype = np.uint32

    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw',
        tiled=True,
    ) as dst:
        dst.write(data, 1)


# 缓存坐标转换器，避免重复构建
_transformer_cache = {}


def pixel_to_geo(row, col, transform, crs):
    """
    将像素坐标 (row, col) 转换为 WGS84 地理坐标 (lon, lat)。

    参数:
        row       : 行号（像素纵坐标）
        col       : 列号（像素横坐标）
        transform : rasterio 仿射变换
        crs       : 源坐标系

    返回:
        (lon, lat) : float 元组
    """
    # 像素中心坐标（原始 CRS）
    x, y = rio_transform.xy(transform, row, col)

    src_epsg = crs.to_epsg()
    if src_epsg == 4326:
        return float(x), float(y)

    # 缓存 Transformer
    key = src_epsg
    if key not in _transformer_cache:
        _transformer_cache[key] = Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        )
    lon, lat = _transformer_cache[key].transform(x, y)
    return float(lon), float(lat)


def write_data_description(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "数据说明.md")
    content = """# 数据说明（Ecological Linkage Tool 输出）

本文件用于说明软件运行后在输出目录生成的数据含义。

## 输入数据

- EcologicalSource.tif：生态源地栅格（源地像元值=1，非源地=0）
- ResistanceSurface.tif：阻力面栅格（用于最小累积阻力路径搜索）
- Connectivity.tif：可以输出属性栅格，建议采用连通性数据（用于廊道属性计算）

## 输出数据（按步骤）

### 步骤 0：生态源地预处理

- mspa_core_filled.tif：源地斑块栅格（像元值为斑块 ID，0 为背景）
- mspa_core.*：源地斑块面矢量（Shapefile，一组文件：.shp/.shx/.dbf/.prj/.cpg）

### 步骤 1：邻接关系计算

- mspa_core_filled_eucall.tif：欧氏分配结果（背景像元被分配到最近斑块）
- adjacency.csv：邻接关系表
  - Block：斑块 ID（较小者）
  - Adjacent：相邻斑块 ID（较大者）

### 步骤 2：最近点对计算（用于廊道范围优化）

- adjacency.csv：在步骤 1 的基础上更新为“带最近点”的邻接表
  - NearestP_Block1_X / NearestP_Block1_Y：Block 的最近点（像素行/列）
  - NearestP_Block2_X / NearestP_Block2_Y：Adjacent 的最近点（像素行/列）
  - Distance：两最近点在像素坐标下的距离

### 步骤 3：廊道构建（MCR）

- ecological_network.graphml：源地节点网络（GraphML）
- ecological_network_act.graphml：包含激活点节点的网络（GraphML）
- ecological_network.*：廊道线矢量（Shapefile，一组文件：.shp/.shx/.dbf/.prj/.cpg）
- ecological_network_point.*：节点点矢量（Shapefile，一组文件：.shp/.shx/.dbf/.prj/.cpg）
  - type：S=源地中心节点，A=激活点节点
  - sourceid：对应的源地斑块 ID

## 说明

- 软件文档与原理介绍：https://github.com/HaoweiGis/Ecological-Linkage-Tool

## 推荐引用

参考：README.md（L21-L31）

If you use **ELT** in your research, please cite:

> Mu H, Guo S, Zhang X, et al. An enhanced ecological network for spatial planning considering spatial conflicts and structural resilience[J]. Geography and Sustainability, 2026: 100420. (https://doi.org/10.1016/j.geosus.2026.100420)

If you use **ELT-Direction** (corridor direction analysis) and **ELT-BiologicalFlow** (biological flow simulation), please cite:

> Mu H, Guo S, Pan K, et al. Revealing the dynamic biological flow between eastern and western China from the perspective of ecological network[J]. Environmental Impact Assessment Review, 2026, 116: 108138. (https://doi.org/10.1016/j.eiar.2025.108138)

If you use **Connectivity.tif** (omnidirectional connectivity data) in this repository, please cite:

> Mu H, Guo S, Zhang X, et al. Moving in the landscape: Omnidirectional connectivity dynamics in China from 1985 to 2020[J]. Environmental Impact Assessment Review, 2025, 110: 107721. (https://doi.org/10.1016/j.eiar.2024.107721)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
