#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 1：邻接关系计算
====================
输入 : mspa_core_filled.tif（带斑块 ID 的源地栅格）
输出 :
    mspa_core_filled_eucall.tif — 欧氏分配结果栅格
    adjacency.csv               — 邻接关系表 [Block, Adjacent]

优化点：
    - euclidean_allocation 使用 numpy 高级索引，消除 O(n²) 双层循环
    - 用 scipy.ndimage.binary_dilation 替代 cv2.dilate
    - 移除多进程版本，单线程即可
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_dilation

from utils import read_raster, write_raster


# ──────────────────────────────────────────────
# 核心算法
# ──────────────────────────────────────────────

def euclidean_allocation(raster, nodata):
    """
    欧氏分配：将每个背景像素归属到最近的非 nodata 源像素。

    【优化】原代码用双层 Python 循环逐像素赋值，复杂度 O(rows×cols)×循环开销。
    现改用 numpy 高级索引（fancy indexing），一行完成，速度提升百倍以上。

    参数:
        raster : 2D int array，源地斑块 ID（nodata 为背景）
        nodata : 背景值

    返回:
        allocation : 2D int array，每个像素对应最近斑块 ID
    """
    if nodata is None:
        valid_mask = (raster > 0)
    else:
        valid_mask = (raster != int(nodata))

    # distance_transform_edt 在 False（背景）像素上计算距离
    _, indices = distance_transform_edt(~valid_mask, return_indices=True)

    # indices[0] = 最近源像素行号，indices[1] = 列号
    allocation = raster[indices[0], indices[1]]
    return allocation


def is_adjacent(block1_mask, block2_mask):
    """
    判断两个二值掩码是否在 8 连通意义下相邻。
    用 scipy binary_dilation 替代 cv2.dilate。
    """
    struct   = np.ones((3, 3), dtype=bool)
    dilated  = binary_dilation(block1_mask, structure=struct)
    return bool(np.any(dilated & block2_mask))


def compute_adjacency(allocation, log_callback=print):
    """
    通过扫描分配栅格，高效提取相邻斑块对。

    算法：利用 numpy 的移位对比，检测所有 8 连通方向上的相邻斑块 ID 差异。
    复杂度：O(rows * cols)，远快于 O(n^2) 的掩码扩张法。
    """
    rows, cols = allocation.shape
    log_callback(f"  使用栅格扫描法提取邻接关系，尺寸: {rows}x{cols}...")

    adj_pairs = set()

    # 定义检查方向：右、下、右下、左下
    shifts = [
        (slice(None), slice(None, -1), slice(None), slice(1, None)),   # 右
        (slice(None, -1), slice(None), slice(1, None), slice(None)),   # 下
        (slice(None, -1), slice(None, -1), slice(1, None), slice(1, None)), # 右下
        (slice(None, -1), slice(1, None), slice(1, None), slice(None, -1))  # 左下
    ]

    for r1, c1, r2, c2 in shifts:
        v1 = allocation[r1, c1]
        v2 = allocation[r2, c2]

        # 找出值不同且均大于 0 的像素对
        mask = (v1 != v2) & (v1 > 0) & (v2 > 0)
        if np.any(mask):
            # 优化：先用 numpy 提取唯一的 ID 对，再加入 set
            # 这样处理数百万个像素边界时，性能提升巨大
            pairs = np.column_stack((v1[mask], v2[mask]))
            # 排序每一行，确保 (1, 2) 和 (2, 1) 被视为相同
            pairs.sort(axis=1)
            # 获取唯一的斑块 ID 对
            unique_pairs = np.unique(pairs, axis=0)
            for p in unique_pairs:
                adj_pairs.add(tuple(p))

    # 排序输出
    final_pairs = sorted([ (int(p[0]), int(p[1])) for p in adj_pairs ])
    return pd.DataFrame(final_pairs, columns=['Block', 'Adjacent'])


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def run_adjacent(filled_tif, output_dir, log_callback=print):
    """
    执行邻接关系计算。

    参数:
        filled_tif   : mspa_core_filled.tif 路径
        output_dir   : 输出目录
        log_callback : 日志回调函数
    """
    log = log_callback
    log("=" * 55)
    log("步骤 1: 邻接关系计算")
    log("=" * 55)

    alloc_tif = os.path.join(output_dir, 'mspa_core_filled_eucall.tif')
    adj_csv   = os.path.join(output_dir, 'adjacency.csv')

    # ── 1. 读取栅格 ─────────────────────────────
    log("[1/3] 读取源地栅格...")
    raster, transform, crs, nodata = read_raster(filled_tif)
    nodata_int = int(nodata) if nodata is not None else 0

    patch_ids = np.unique(raster[raster > 0])
    log(f"  斑块数量 : {len(patch_ids)}")
    log(f"  栅格尺寸 : {raster.shape}")

    # ── 2. 欧氏分配 ─────────────────────────────
    log("[2/3] 欧氏分配（将背景像素归属到最近斑块）...")
    allocation = euclidean_allocation(raster, nodata_int)

    write_raster(alloc_tif, allocation.astype('int32'), transform, crs, nodata=0)
    log(f"  ✓ 欧氏分配栅格: {alloc_tif}")

    # ── 3. 邻接关系 ─────────────────────────────
    log("[3/3] 计算邻接关系...")
    adj_df = compute_adjacency(allocation, log)
    
    # 将邻接对持久化
    adj_df.to_csv(adj_csv, index=False)
    
    log(f"  ✓ 邻接对数 : {len(adj_df)}")
    if len(adj_df) == 0:
        log("  ⚠ 注意: 邻接对数为 0。可能原因: 只有 1 个斑块，或斑块 ID 均无效。")
    
    log(f"  ✓ 邻接关系保存至 : {adj_csv}")

    # ── 验证：孤立斑块检查 ──────────────────────
    img_ids = set(int(v) for v in patch_ids)
    csv_ids = (set(adj_df['Block'].astype(int).unique())
               | set(adj_df['Adjacent'].astype(int).unique()))
    missing = img_ids - csv_ids
    if missing:
        log(f"  ⚠ 孤立斑块（无邻接关系）: {sorted(missing)}")
    else:
        log("  ✓ 所有斑块均有邻接关系")

    log("─" * 55)
    log("步骤 1 完成。")
    log("下一步: 步骤 2（激活点计算）")
