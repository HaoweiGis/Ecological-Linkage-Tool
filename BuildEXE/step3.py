#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 3：生态廊道构建（最小累积阻力模型 MCR）
=============================================
输入 :
    mspa_core_filled.tif  — 源地栅格（带斑块 ID）
    Connectivity.tif      — 连通性栅格（用于廊道权重属性）
    ResistanceSurface.tif — 阻力面（用于 Dijkstra 路径搜索）
    adjacency.csv         — 邻接关系 + 最近点（步骤 2 输出）
输出 :
    ecological_network.graphml      — 源点图
    ecological_network_act.graphml  — 含激活点图
    ecological_network.shp          — 廊道线矢量
    ecological_network_point.shp    — 节点点矢量

修复：
    - 去除全局变量 node_raster_path 在函数内的隐式引用
    - transform/crs 作为参数传入 pixel_to_geo
"""

import os
import heapq
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import euclidean

from utils import read_raster, pixel_to_geo


# ──────────────────────────────────────────────
# 子栅格提取
# ──────────────────────────────────────────────

def extract_subgrid(grid, p1, p2):
    """
    以两点为中心，提取一个包含两点的正方形子栅格（缓冲 = 欧氏距离）。

    返回:
        subgrid : 切片后的 2D array（视图）
        origin  : (row_min, col_min) 子栅格左上角在全局栅格中的坐标
    """
    r1, c1 = p1
    r2, c2 = p2
    buf = max(1, int(np.ceil(euclidean(p1, p2))))

    r_min = max(0, min(r1, r2) - buf)
    r_max = min(grid.shape[0] - 1, max(r1, r2) + buf)
    c_min = max(0, min(c1, c2) - buf)
    c_max = min(grid.shape[1] - 1, max(c1, c2) + buf)

    return grid[r_min:r_max + 1, c_min:c_max + 1], (r_min, c_min)


def local_to_global(origin, path):
    """将子栅格坐标转换为全局栅格坐标。"""
    r0, c0 = origin
    return [(r + r0, c + c0) for r, c in path]


# ──────────────────────────────────────────────
# Dijkstra 最小累积阻力
# ──────────────────────────────────────────────

def cost_distance(source_mask, cost_raster, eight_conn=True):
    """
    基于 Dijkstra 的最小累积阻力计算。

    参数:
        source_mask  : 2D 布尔/int array，> 0 为源点
        cost_raster  : 2D float array，阻力面（NaN 视为不可通行）
        eight_conn   : True = 8 方向，False = 4 方向

    返回:
        cost_dist  : 累积阻力栅格（np.inf 表示不可达）
        back_dir   : 回溯方向索引（-1 表示源点或不可达）
    """
    rows, cols = cost_raster.shape
    cost_dist  = np.full((rows, cols), np.inf)
    back_dir   = np.full((rows, cols), -1, dtype='int8')
    visited    = np.zeros((rows, cols), dtype=bool)
    pq         = []

    # 初始化源点
    for r, c in zip(*np.where(source_mask > 0)):
        cost_dist[r, c] = 0.0
        heapq.heappush(pq, (0.0, int(r), int(c)))

    if eight_conn:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1,-1), (-1, 1), (1,-1), (1, 1)]
    else:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        cur_cost, r, c = heapq.heappop(pq)
        if visited[r, c]:
            continue
        visited[r, c] = True

        for i, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc]:
                continue

            cell_cost = float(cost_raster[nr, nc])
            if np.isnan(cell_cost) or cell_cost < 0:
                continue  # 不可通行像素

            # 对角线移动乘以 √2
            move_cost = cell_cost if (dr == 0 or dc == 0) else cell_cost * 1.41421356
            new_cost  = cur_cost + move_cost

            if new_cost < cost_dist[nr, nc]:
                cost_dist[nr, nc] = new_cost
                back_dir[nr, nc]  = i
                heapq.heappush(pq, (new_cost, nr, nc))

    return cost_dist, back_dir


def trace_path(cost_dist, back_dir, target_mask, eight_conn=True):
    """
    从目标像素回溯路径到源点。

    参数:
        cost_dist   : cost_distance 返回的累积阻力栅格
        back_dir    : cost_distance 返回的方向栅格
        target_mask : 目标像素掩码（> 0 为目标）
        eight_conn  : 与 cost_distance 保持一致

    返回:
        path : [(row, col), ...] 从源到目标的路径，若无路径则返回 []
    """
    if eight_conn:
        inverse = [(1, 0), (-1, 0), (0, 1), (0,-1),
                   (1, 1), (1,-1), (-1, 1), (-1,-1)]
    else:
        inverse = [(1, 0), (-1, 0), (0, 1), (0,-1)]

    targets = np.argwhere(target_mask > 0)
    if len(targets) == 0:
        return []

    # 找到目标像素中累积阻力最小的
    target_costs = [cost_dist[r, c] for r, c in targets]
    best         = targets[int(np.argmin(target_costs))]
    r, c         = int(best[0]), int(best[1])

    if np.isinf(cost_dist[r, c]):
        return []  # 目标不可达

    # 回溯
    path = []
    max_steps = cost_dist.size  # 防止死循环
    steps = 0
    while cost_dist[r, c] != 0.0 and steps < max_steps:
        path.append((r, c))
        d = int(back_dir[r, c])
        if d < 0:
            break
        dr, dc = inverse[d]
        r, c   = r + dr, c + dc
        steps += 1
    path.append((r, c))

    return path[::-1]


# ──────────────────────────────────────────────
# 廊道构建主逻辑
# ──────────────────────────────────────────────

def _build_corridor(node_i, node_j, core_raster, resistance_raster,
                    connectivity_raster, pi, pj, node_positions,
                    transform, crs):
    """
    计算两个相邻斑块之间的最优廊道（MCR 路径）。

    返回:
        path        : 全局坐标列表，或 None（若路径无效）
        mean_weight : 廊道平均连通性权重
        sum_weight  : 廊道总连通性权重
        from_geo    : 廊道起点地理坐标 (lon, lat)
        to_geo      : 廊道终点地理坐标 (lon, lat)
    """
    source_mask = np.zeros(core_raster.shape, dtype='float32')
    source_mask[core_raster == node_i] = 1.0

    target_mask = np.zeros(core_raster.shape, dtype='float32')
    target_mask[core_raster == node_j] = 1.0

    # 提取子栅格
    src_sub,  origin = extract_subgrid(source_mask,  pi, pj)
    tgt_sub,  _      = extract_subgrid(target_mask,  pi, pj)
    res_sub,  _      = extract_subgrid(resistance_raster, pi, pj)

    cost_dist, back_dir = cost_distance(src_sub, res_sub, eight_conn=True)
    sub_path = trace_path(cost_dist, back_dir, tgt_sub, eight_conn=True)

    if len(sub_path) <= 1:
        return None, None, None, None, None

    path = local_to_global(origin, sub_path)

    # 若廊道经过其他斑块，丢弃
    for r, c in path:
        v = core_raster[r, c]
        if v > 0 and v != node_i and v != node_j:
            return None, None, None, None, None

    # 计算权重（连通性均值）
    conn_vals   = [connectivity_raster[r, c] for r, c in path
                   if not np.isnan(connectivity_raster[r, c])]
    mean_weight = float(np.mean(conn_vals)) if conn_vals else 0.0
    sum_weight  = float(np.sum(conn_vals))  if conn_vals else 0.0

    from_geo = pixel_to_geo(path[0][0],  path[0][1],  transform, crs)
    to_geo   = pixel_to_geo(path[-1][0], path[-1][1], transform, crs)

    return path, mean_weight, sum_weight, from_geo, to_geo


def _build_spine_corridor(src_pos, act_pos, resistance_raster,
                          connectivity_raster, transform, crs):
    """
    计算源地中心点 → 廊道激活点之间的连接廊道（spine line）。

    返回:
        geo_path   : [(lon, lat), ...] 或 None
        mean_weight, sum_weight, length
    """
    if src_pos == act_pos:
        return None, None, None, None

    src_mask = np.zeros(resistance_raster.shape, dtype='float32')
    src_mask[src_pos[0], src_pos[1]] = 1.0

    tgt_mask = np.zeros(resistance_raster.shape, dtype='float32')
    tgt_mask[act_pos[0], act_pos[1]] = 1.0

    res_sub, origin = extract_subgrid(resistance_raster, src_pos, act_pos)
    src_sub, _      = extract_subgrid(src_mask,          src_pos, act_pos)
    tgt_sub, _      = extract_subgrid(tgt_mask,          src_pos, act_pos)

    cost_dist, back_dir = cost_distance(src_sub, res_sub, eight_conn=True)
    sub_path = trace_path(cost_dist, back_dir, tgt_sub, eight_conn=True)

    if len(sub_path) <= 1:
        return None, None, None, None

    path     = local_to_global(origin, sub_path)
    geo_path = [pixel_to_geo(r, c, transform, crs) for r, c in path]

    conn_vals   = [connectivity_raster[r, c] for r, c in path
                   if not np.isnan(connectivity_raster[r, c])]
    mean_weight = float(np.mean(conn_vals)) if conn_vals else 0.0
    sum_weight  = float(np.sum(conn_vals))  if conn_vals else 0.0

    return geo_path, mean_weight, sum_weight, len(path)


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def run_corridor(node_tif, core_tif, connectivity_tif, resistance_tif,
                 adj_csv, output_dir, log_callback=print):
    """
    执行廊道构建。

    参数:
        node_tif         : 保留参数位（兼容旧调用）；可传 None 或不存在的路径，不再依赖该文件
        core_tif         : mspa_core_filled.tif（源地栅格）
        connectivity_tif : Connectivity.tif（连通性，用于权重属性）
        resistance_tif   : ResistanceSurface.tif（阻力面，用于路径搜索）
        adj_csv          : adjacency.csv（步骤 2 输出）
        output_dir       : 输出目录
        log_callback     : 日志回调函数
    """
    log = log_callback
    log("=" * 55)
    log("步骤 3: 生态廊道构建（MCR）")
    log("=" * 55)

    # 输出路径
    out_graphml     = os.path.join(output_dir, 'ecological_network.graphml')
    out_graphml_act = os.path.join(output_dir, 'ecological_network_act.graphml')
    out_shp_line    = os.path.join(output_dir, 'ecological_network.shp')
    out_shp_point   = os.path.join(output_dir, 'ecological_network_point.shp')

    # ── 1. 读取栅格 ─────────────────────────────
    log("读取栅格数据...")
    core_raster,          transform, crs, _ = read_raster(core_tif)
    connectivity_raster,  _,         _,  _ = read_raster(connectivity_tif)
    resistance_raster,    _,         _,  _ = read_raster(resistance_tif)

    # 转 float，无效值设为 NaN
    connectivity_raster = connectivity_raster.astype('float64')
    resistance_raster   = resistance_raster.astype('float64')
    connectivity_raster[connectivity_raster <= 0] = np.nan
    resistance_raster[resistance_raster <= 0]     = np.nan

    # ── 2. 读取邻接关系 ─────────────────────────
    log("读取邻接关系...")
    adj_df = pd.read_csv(adj_csv)
    log(f"  邻接对数 : {len(adj_df)}")

    if len(adj_df) == 0:
        log("  ⚠ 邻接关系表为空，请先成功运行 步骤 1 & 2。")
        return

    # ── 3. 确定节点位置（像素坐标）─────────────
    core_ids = np.unique(core_raster[core_raster > 0]).astype(int)
    if len(core_ids) == 0:
        log("  ⚠ 源地栅格中未找到有效斑块（值 > 0）。")
        return

    centers = center_of_mass(
        np.ones_like(core_raster, dtype=np.uint8),
        labels=core_raster,
        index=core_ids,
    )

    node_positions = {}
    rows, cols = core_raster.shape
    for node_id, center in zip(core_ids, centers):
        r_f, c_f = float(center[0]), float(center[1])
        if np.isnan(r_f) or np.isnan(c_f):
            continue

        r0 = int(round(r_f))
        c0 = int(round(c_f))
        r0 = max(0, min(rows - 1, r0))
        c0 = max(0, min(cols - 1, c0))

        if int(core_raster[r0, c0]) != int(node_id):
            coords = np.argwhere(core_raster == node_id)
            if len(coords) == 0:
                continue
            d = (coords[:, 0].astype('float64') - r_f) ** 2 + (coords[:, 1].astype('float64') - c_f) ** 2
            idx = int(np.argmin(d))
            r0, c0 = int(coords[idx][0]), int(coords[idx][1])

        node_positions[int(node_id)] = (r0, c0)

    log(f"  节点数   : {len(node_positions)}")

    # ── 4. 初始化图结构和输出容器 ────────────────
    G     = nx.Graph()   # 源点图（source → source）
    G_act = nx.Graph()   # 激活点图（含中间节点）

    geom_lines  = []
    attr_lines  = []
    geom_points = []
    attr_points = []

    act_node_id = int(max(node_positions.keys())) + 1 if node_positions else 1
    line_id     = 0
    skipped     = []

    # 添加源节点
    for node, (row, col) in node_positions.items():
        lon, lat = pixel_to_geo(row, col, transform, crs)
        G.add_node(node, lon=lon, lat=lat)
        G_act.add_node(node, lon=lon, lat=lat)
        geom_points.append(Point(lon, lat))
        attr_points.append({"nodeid": node, "sourceid": node, "type": "S"})

    # ── 5. 遍历邻接对，构建廊道 ──────────────────
    total_pairs = int(len(adj_df))
    log(f"\n构建廊道（共 {total_pairs} 对邻接斑块）...")

    def _progress_text(done: int, total: int):
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        frac = done / total
        width = 28
        filled = int(round(frac * width))
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {done}/{total} ({frac * 100:.1f}%)"

    update_every = max(1, total_pairs // 100)
    log(f"  进度: {_progress_text(0, total_pairs)}")

    for i, adj_row in enumerate(adj_df.itertuples(index=False), start=1):
        node_i = int(getattr(adj_row, 'Block'))
        node_j = int(getattr(adj_row, 'Adjacent'))

        pos_i = node_positions.get(node_i)
        pos_j = node_positions.get(node_j)
        if pos_i is None or pos_j is None:
            if i % update_every == 0 or i == total_pairs:
                log(f"  进度: {_progress_text(i, total_pairs)}")
            continue

        pi = (int(getattr(adj_row, 'NearestP_Block1_X')), int(getattr(adj_row, 'NearestP_Block1_Y')))
        pj = (int(getattr(adj_row, 'NearestP_Block2_X')), int(getattr(adj_row, 'NearestP_Block2_Y')))

        path, mean_w, sum_w, from_geo, to_geo = _build_corridor(
            node_i, node_j, core_raster, resistance_raster,
            connectivity_raster, pi, pj, node_positions, transform, crs
        )

        if path is None:
            skipped.append(f"{node_i}--{node_j}")
            if i % update_every == 0 or i == total_pairs:
                log(f"  进度: {_progress_text(i, total_pairs)}")
            continue

        line_id     += 1
        act_id_i     = act_node_id;     act_node_id += 1
        act_id_j     = act_node_id;     act_node_id += 1

        G.add_edge(node_i, node_j, weight=mean_w, sum_weight=sum_w, distance=len(path))

        geo_path = [pixel_to_geo(r, c, transform, crs) for r, c in path]
        geom_lines.append(LineString(geo_path))
        attr_lines.append({
            "fromnode"  : node_i, "tonode"    : node_j,
            "fromX"     : from_geo[0], "fromY": from_geo[1],
            "toX"       : to_geo[0],   "toY"  : to_geo[1],
            "weight"    : mean_w, "sum_weight": sum_w,
            "len"       : len(path), "type"  : "T", "lineid": line_id,
        })

        G_act.add_node(act_id_i, lon=from_geo[0], lat=from_geo[1])
        G_act.add_node(act_id_j, lon=to_geo[0],   lat=to_geo[1])
        G_act.add_edge(act_id_i, act_id_j, weight=mean_w, sum_weight=sum_w, distance=len(path), lineid=line_id)

        geom_points.append(Point(from_geo))
        attr_points.append({"nodeid": act_id_i, "sourceid": node_i, "type": "A"})
        geom_points.append(Point(to_geo))
        attr_points.append({"nodeid": act_id_j, "sourceid": node_j, "type": "A"})

        for src_pos, act_pos, n_id in [
            (pos_i, path[0],  node_i),
            (pos_j, path[-1], node_j),
        ]:
            geo_spine, mw, sw, ln = _build_spine_corridor(
                src_pos, act_pos, resistance_raster,
                connectivity_raster, transform, crs
            )
            if geo_spine:
                line_id += 1
                src_geo  = pixel_to_geo(src_pos[0], src_pos[1], transform, crs)
                act_geo  = pixel_to_geo(act_pos[0], act_pos[1], transform, crs)
                geom_lines.append(LineString(geo_spine))
                attr_lines.append({
                    "fromnode"  : n_id, "tonode"    : n_id,
                    "fromX"     : src_geo[0], "fromY": src_geo[1],
                    "toX"       : act_geo[0], "toY"  : act_geo[1],
                    "weight"    : mw, "sum_weight"  : sw,
                    "len"       : ln, "type"        : "S",
                    "lineid"    : line_id,
                })

        if i % update_every == 0 or i == total_pairs:
            log(f"  进度: {_progress_text(i, total_pairs)}")

    if skipped:
        log(f"\n  ⚠ 跳过 {len(skipped)} 条廊道（路径过短或经过其他斑块）")

    # ── 6. 去重并保存 ────────────────────────────
    log("\n保存结果...")

    gdf_lines = gpd.GeoDataFrame(attr_lines, geometry=geom_lines, crs="EPSG:4326")
    if len(gdf_lines) > 0:
        gdf_lines['_wkb'] = gdf_lines.geometry.apply(lambda g: g.wkb)
        gdf_lines = gdf_lines.loc[
            gdf_lines.groupby('_wkb')['lineid'].idxmin()
        ].drop(columns='_wkb').reset_index(drop=True)

    gdf_points = gpd.GeoDataFrame(attr_points, geometry=geom_points, crs="EPSG:4326")
    if len(gdf_points) > 0:
        gdf_points['_wkb'] = gdf_points.geometry.apply(lambda g: g.wkb)
        gdf_points = gdf_points.loc[
            gdf_points.groupby('_wkb')['nodeid'].idxmin()
        ].drop(columns='_wkb').reset_index(drop=True)

    gdf_lines.to_file(out_shp_line)
    gdf_points.to_file(out_shp_point)
    nx.write_graphml(G,     out_graphml)
    nx.write_graphml(G_act, out_graphml_act)

    log(f"  ✓ 廊道矢量  : {out_shp_line}  ({len(gdf_lines)} 条)")
    log(f"  ✓ 节点矢量  : {out_shp_point} ({len(gdf_points)} 个)")
    log(f"  ✓ 源点网络  : {out_graphml}")
    log(f"  ✓ 激活点网络: {out_graphml_act}")
    log("─" * 55)
    log("步骤 3 完成。生态廊道分析结束！")
