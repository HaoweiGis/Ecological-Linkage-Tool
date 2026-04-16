#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 2：激活点计算
==================
输入 :
    mspa_core_filled.tif — 源地栅格（带斑块 ID）
    adjacency.csv        — 邻接关系表 [Block, Adjacent]
输出 :
    adjacency.csv          — 原文件追加列:
                             NearestP_Block1_X/Y, NearestP_Block2_X/Y, Distance
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
from tqdm import tqdm

from utils import read_raster, write_raster


# ──────────────────────────────────────────────
# 核心算法
# ──────────────────────────────────────────────

def get_boundary_pixels(raster, nodata=0):
    """
    通过腐蚀提取每个斑块的内部边界像素。

    原理: boundary = patch - eroded(patch)
    使用 8 连通结构元素（3×3 全 1）。

    返回:
        boundaries : 2D int32 array，边界像素值为斑块 ID，背景为 0
    """
    struct       = np.ones((3, 3), dtype=bool)
    unique_labels = np.unique(raster)
    unique_labels = unique_labels[unique_labels != nodata]

    boundaries = np.zeros_like(raster, dtype='int32')
    for label in unique_labels:
        patch          = (raster == label)
        eroded         = binary_erosion(patch, structure=struct)
        boundary       = patch & ~eroded
        boundaries[boundary] = int(label)

    return boundaries


def find_nearest_boundary_pairs(boundary_raster, adjacency_df, log_callback=print):
    """
    对邻接 DataFrame 中每对斑块，找到两者边界像素之间的最近点对。

    使用 scipy.spatial.distance.cdist 计算距离矩阵，避免 Python 级别循环。

    返回:
        pd.DataFrame，列 [Block, Adjacent,
                           NearestP_Block1_X, NearestP_Block1_Y,
                           NearestP_Block2_X, NearestP_Block2_Y,
                           Distance]
    """
    unique_labels = np.unique(boundary_raster)
    unique_labels = unique_labels[unique_labels != 0]

    # 预计算各斑块边界像素位置，避免重复 argwhere
    log_callback("  预计算边界像素位置...")
    boundary_positions = {
        int(lbl): np.argwhere(boundary_raster == lbl)
        for lbl in unique_labels
    }

    records = []
    skipped = []

    for _, row in tqdm(adjacency_df.iterrows(), total=len(adjacency_df),
                       desc="计算最近点对", unit="pair"):
        b1 = int(row['Block'])
        b2 = int(row['Adjacent'])

        p1 = boundary_positions.get(b1)
        p2 = boundary_positions.get(b2)

        if p1 is None or len(p1) == 0 or p2 is None or len(p2) == 0:
            skipped.append((b1, b2))
            continue

        dists   = cdist(p1, p2)
        idx     = np.unravel_index(np.argmin(dists), dists.shape)
        r1, c1  = p1[idx[0]]
        r2, c2  = p2[idx[1]]
        min_d   = float(dists[idx])

        records.append({
            'Block'              : b1,
            'Adjacent'           : b2,
            'NearestP_Block1_X'  : int(r1),
            'NearestP_Block1_Y'  : int(c1),
            'NearestP_Block2_X'  : int(r2),
            'NearestP_Block2_Y'  : int(c2),
            'Distance'           : min_d,
        })

    if skipped:
        log_callback(f"  ⚠ {len(skipped)} 对斑块无边界像素，已跳过: {skipped[:5]}...")

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def run_activate(filled_tif, adj_csv_path, output_dir, log_callback=print):
    """
    执行激活点计算。

    参数:
        filled_tif   : mspa_core_filled.tif 路径
        adj_csv_path : adjacency.csv 路径（步骤 1 输出）
        output_dir   : 输出目录
        log_callback : 日志回调函数
    """
    log = log_callback
    log("=" * 55)
    log("步骤 2: 激活点计算")
    log("=" * 55)

    # ── 1. 读取数据 ─────────────────────────────
    log("[1/3] 读取栅格和邻接关系...")
    raster, transform, crs, nodata = read_raster(filled_tif)
    nodata_int  = int(nodata) if nodata is not None else 0
    adjacency_df = pd.read_csv(adj_csv_path)
    log(f"  邻接对数 : {len(adjacency_df)}")

    if len(adjacency_df) == 0:
        log("  ⚠ 邻接关系表为空，请先成功运行 步骤 1 以生成有效的邻接对。")
        # 返回空结果以防止后续代码崩溃
        result_df = pd.DataFrame(columns=['Block', 'Adjacent', 'NearestP_Block1_X', 'NearestP_Block1_Y',
                                          'NearestP_Block2_X', 'NearestP_Block2_Y', 'Distance'])
        result_df.to_csv(adj_csv_path, index=False)
        return

    # ── 2. 边界提取 ─────────────────────────────
    log("[2/3] 提取各斑块边界像素...")
    boundary_raster = get_boundary_pixels(raster, nodata_int)
    n_boundary = int(np.sum(boundary_raster > 0))
    log(f"  边界像素总数 : {n_boundary}")

    # ── 3. 最近点对 ─────────────────────────────
    log("[3/3] 计算邻接斑块间最近点对...")
    result_df = find_nearest_boundary_pairs(boundary_raster, adjacency_df, log)

    # 将结果写回 adjacency.csv（覆盖追加最近点信息）
    result_df.to_csv(adj_csv_path, index=False)
    log(f"  ✓ 邻接+最近点 : {adj_csv_path}")
    log(f"  ✓ 有效邻接对 : {len(result_df)}")

    log("─" * 55)
    log("步骤 2 完成。")
    log("下一步: 步骤 3（廊道构建）")
