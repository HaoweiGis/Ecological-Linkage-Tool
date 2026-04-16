#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生态廊道构建工具 - 基于最小累积阻力模型
=========================================
优化点：
    - 坐标转换器缓存（避免每像素新建 SpatialReference）
    - 阻力面 NaN 处理（Dijkstra 不可处理 NaN）
    - 步骤 3 廊道构建多进程并行（每对邻接斑块独立计算）
"""

# 执行时间记录（geobase；InputData -> OutputData；2026-04-16；ELT_MAX_PAIRS=200）
# 单进程（ELT_N_WORKERS=1）：649.85 s
# 6 进程（ELT_N_WORKERS=6）：208.34 s（约 3.12x 加速）

import numpy as np
import networkx as nx
from osgeo import gdal, osr
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import LineString, Point
import pandas as pd
import heapq
import os
import sys
import glob
import sqlite3
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

_transformer_cache = {}
_CORE_RASTER = None
_RESISTANCE_RASTER = None
_CONNECTIVITY_RASTER = None
_GT = None
_PROJ_WKT = None
_PROJ_READY = False


def _init_worker(core_raster, resistance_raster, connectivity_raster, gt, proj_wkt):
    global _CORE_RASTER, _RESISTANCE_RASTER, _CONNECTIVITY_RASTER, _GT, _PROJ_WKT
    _CORE_RASTER = np.array(core_raster)
    _RESISTANCE_RASTER = np.array(resistance_raster, dtype='float64')
    _RESISTANCE_RASTER[_RESISTANCE_RASTER <= 0] = np.nan
    _CONNECTIVITY_RASTER = np.array(connectivity_raster, dtype='float64')
    _CONNECTIVITY_RASTER[_CONNECTIVITY_RASTER <= 0] = np.nan
    _GT = gt
    _PROJ_WKT = proj_wkt

def _proj_db_compatible(proj_db_path):
    try:
        con = sqlite3.connect(proj_db_path)
        cur = con.cursor()
        cur.execute("PRAGMA table_info(projected_crs)")
        projected_cols = {row[1] for row in cur.fetchall()}
        cur.execute("PRAGMA table_info(geodetic_crs)")
        geodetic_cols = {row[1] for row in cur.fetchall()}
        con.close()
        return ('area_of_use_auth_name' in projected_cols) and ('area_of_use_auth_name' in geodetic_cols)
    except Exception:
        return False

def _ensure_proj_db():
    global _PROJ_READY
    if _PROJ_READY:
        return

    cand_dirs = []
    env_proj_lib = os.environ.get('PROJ_LIB') or ''
    for part in env_proj_lib.split(';'):
        part = part.strip()
        if part:
            cand_dirs.append(part)

    conda_prefix = os.environ.get('CONDA_PREFIX') or ''
    if conda_prefix:
        cand_dirs.append(os.path.join(conda_prefix, 'Library', 'share', 'proj'))

    for prefix in {sys.prefix, getattr(sys, 'base_prefix', ''), getattr(sys, 'real_prefix', '')}:
        if prefix:
            cand_dirs.append(os.path.join(prefix, 'Library', 'share', 'proj'))

    for root in {os.path.join(os.path.dirname(sys.prefix), 'pkgs'), os.path.join(sys.prefix, 'pkgs'), r'C:\ProgramData\Anaconda3\pkgs'}:
        if root and os.path.isdir(root):
            for db_path in glob.glob(os.path.join(root, 'proj-*', 'Library', 'share', 'proj', 'proj.db')):
                cand_dirs.append(os.path.dirname(db_path))

    chosen = None
    for d in cand_dirs:
        db = os.path.join(d, 'proj.db')
        if os.path.exists(db) and _proj_db_compatible(db):
            chosen = d
            break

    if chosen:
        os.environ['PROJ_LIB'] = chosen
        osr.SetPROJSearchPaths([chosen])
        _PROJ_READY = True

def _get_transformer(src_wkt):
    _ensure_proj_db()
    if src_wkt not in _transformer_cache:
        src = osr.SpatialReference()
        src.ImportFromWkt(src_wkt)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)
        _transformer_cache[src_wkt] = osr.CoordinateTransformation(src, dst)
    return _transformer_cache[src_wkt]

def pixel_to_geo(dataset, col, row):
    gt = dataset.GetGeoTransform()
    px = gt[0] + col * gt[1] + row * gt[2]
    py = gt[3] + col * gt[4] + row * gt[5]
    return px, py

def pixel_to_geo_by_gt(gt, proj_wkt, col, row):
    px = gt[0] + col * gt[1] + row * gt[2]
    py = gt[3] + col * gt[4] + row * gt[5]
    return px, py

def GeoImgR(filename):
    dataset = gdal.Open(filename)
    im_data = np.array(dataset.ReadAsArray())
    im_porj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
    return im_data, im_porj, im_geotrans

def extract_subgrid(grid, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = int(np.ceil(np.sqrt((x1 - x2)**2 + (y1 - y2)**2)))
    x_min = max(0, min(x1, x2) - dist)
    x_max = min(grid.shape[0] - 1, max(x1, x2) + dist)
    y_min = max(0, min(y1, y2) - dist)
    y_max = min(grid.shape[1] - 1, max(y1, y2) + dist)
    return grid[x_min:x_max+1, y_min:y_max+1], (x_min, y_min)

def extract_center_subgrid(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = int(np.ceil(np.sqrt((x1 - x2)**2 + (y1 - y2)**2)))
    x_min = max(0, min(x1, x2) - dist)
    x_max = x_min + 2 * dist
    y_min = max(0, min(y1, y2) - dist)
    y_max = y_min + 2 * dist
    sub1 = np.zeros((2 * dist + 1, 2 * dist + 1))
    sub2 = np.zeros((2 * dist + 1, 2 * dist + 1))
    sub1[x1 - x_min, y1 - y_min] = 1
    sub2[x2 - x_min, y2 - y_min] = 1
    return sub1, sub2, (x_min, y_min)

def restore_path(origin, path):
    ox, oy = origin
    return [(x + ox, y + oy) for x, y in path]

def cost_distance(src_ras, cost_ras, eight=True):
    rows, cols = cost_ras.shape
    cost_dist = np.full((rows, cols), np.inf)
    direction  = np.full((rows, cols), -1, dtype=np.int8)
    visited    = np.zeros((rows, cols), dtype=bool)
    pq = []
    for r in range(rows):
        for c in range(cols):
            if src_ras[r, c] > 0:
                cost_dist[r, c] = 0
                heapq.heappush(pq, (0.0, r, c))
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] if eight else [(-1,0),(1,0),(0,-1),(0,1)]
    codes = [0,1,2,3,4,5,6,7] if eight else [0,1,2,3]
    while pq:
        cur, r, c = heapq.heappop(pq)
        if visited[r, c]:
            continue
        visited[r, c] = True
        for i, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or visited[nr, nc]:
                continue
            cell = float(cost_ras[nr, nc])
            if np.isnan(cell) or cell < 0:
                continue
            move_cost = cell if (dr == 0 or dc == 0) else cell * 1.41421356
            new_cost = cur + move_cost
            if new_cost < cost_dist[nr, nc]:
                cost_dist[nr, nc] = new_cost
                direction[nr, nc]  = codes[i]
                heapq.heappush(pq, (new_cost, nr, nc))
    return cost_dist, direction

def trace_path(cost_dist, direction, target_ras, eight=True):
    targets = np.argwhere(target_ras > 0)
    if len(targets) == 0:
        return []
    costs = [cost_dist[r, c] for r, c in targets]
    best = targets[int(np.argmin(costs))]
    r, c = int(best[0]), int(best[1])
    if np.isinf(cost_dist[r, c]):
        return []
    inverse = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)] if eight else [(1,0),(-1,0),(0,1),(0,-1)]
    path = []
    visited = set()
    for _ in range(cost_dist.size):
        if not (0 <= r < cost_dist.shape[0] and 0 <= c < cost_dist.shape[1]):
            return []
        if (r, c) in visited:
            return []
        visited.add((r, c))
        path.append((r, c))
        d = int(direction[r, c])
        if d < 0:
            break
        dr, dc = inverse[d]
        r, c = r + dr, c + dc
        if not (0 <= r < cost_dist.shape[0] and 0 <= c < cost_dist.shape[1]):
            return []
        if cost_dist[r, c] == 0:
            break
    return path[::-1]

def _is_continuous_path(path, eight=True):
    if len(path) <= 1:
        return False
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        dr = abs(int(r1) - int(r0))
        dc = abs(int(c1) - int(c0))
        if dr == 0 and dc == 0:
            return False
        if eight:
            if dr > 1 or dc > 1:
                return False
        else:
            if dr + dc != 1:
                return False
    return True


def _path_quality(path, resistance_raster):
    if len(path) <= 1:
        return None
    total_cost = 0.0
    total_len = 0.0
    turns = 0
    prev_dir = None
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        r1i, c1i = int(r1), int(c1)
        cell = float(resistance_raster[r1i, c1i])
        if np.isnan(cell) or cell < 0:
            return None
        dr = int(r1) - int(r0)
        dc = int(c1) - int(c0)
        step_len = 1.0 if (dr == 0 or dc == 0) else 1.41421356
        total_len += step_len
        total_cost += cell * step_len
        step_dir = (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))
        if prev_dir is not None and step_dir != prev_dir:
            turns += 1
        prev_dir = step_dir
    mean_cost = float(total_cost / max(1.0, total_len))
    (rs, cs), (re, ce) = path[0], path[-1]
    euclid = float(np.hypot(float(int(re) - int(rs)), float(int(ce) - int(cs))))
    straightness = float(euclid / max(1e-12, total_len))
    return {
        'cost_sum': float(total_cost),
        'cost_mean': mean_cost,
        'turns': int(turns),
        'len_w': float(total_len),
        'straight': straightness,
    }

def _build_single_corridor(args):
    """在子进程中构建单条廊道，返回结果元组（可 pickle）。"""
    (node_i, node_j, pi, pj, pos_i, pos_j) = args

    core_raster = _CORE_RASTER
    res_ras = _RESISTANCE_RASTER
    conn_ras = _CONNECTIVITY_RASTER
    gt = _GT
    proj_wkt = _PROJ_WKT

    core_sub, origin = extract_subgrid(core_raster, pi, pj)
    src_sub = (core_sub == node_i).astype(np.float32)
    tgt_sub = (core_sub == node_j).astype(np.float32)
    res_sub,  _     = extract_subgrid(res_ras, pi, pj)

    cost_dist, direction = cost_distance(src_sub, res_sub, eight=True)
    sub_path = trace_path(cost_dist, direction, tgt_sub, eight=True)
    if len(sub_path) <= 1:
        return None

    path = restore_path(origin, sub_path)
    if not _is_continuous_path(path, eight=True):
        return None

    arr_path = np.array(path)
    if arr_path.size == 0:
        return None
    if np.any(arr_path[:, 0] < 0) or np.any(arr_path[:, 1] < 0):
        return None
    if np.any(arr_path[:, 0] >= core_raster.shape[0]) or np.any(arr_path[:, 1] >= core_raster.shape[1]):
        return None
    if int(core_raster[int(path[0][0]), int(path[0][1])]) != int(node_i):
        return None
    if int(core_raster[int(path[-1][0]), int(path[-1][1])]) != int(node_j):
        return None
    mask_others = np.zeros(core_raster.shape, dtype=bool)
    mask_others[(core_raster != 0) & (core_raster != node_i) & (core_raster != node_j)] = True
    if np.any(mask_others[arr_path[:, 0], arr_path[:, 1]]):
        return None


    q = _path_quality(path, res_ras)
    if q is None:
        return None
    conn_vals = [float(conn_ras[r, c])
                 for r, c in path
                 if 0 <= r < conn_ras.shape[0]
                 and 0 <= c < conn_ras.shape[1]
                 and 0 <= c < conn_ras.shape[1]
                 and not np.isnan(conn_ras[r, c])]
    mean_w = float(np.mean(conn_vals)) if conn_vals else 0.0
    sum_w  = float(np.sum(conn_vals))  if conn_vals else 0.0

    spine_results = []

    for src_pos, act_pos, nid in [
        (pos_i, path[0],  node_i),
        (pos_j, path[-1], node_j),
    ]:
        res_sub2, origin2 = extract_subgrid(res_ras, src_pos, act_pos)
        src_sub2 = np.zeros(res_sub2.shape, dtype=np.float32)
        tgt_sub2 = np.zeros(res_sub2.shape, dtype=np.float32)
        r0 = int(src_pos[0] - origin2[0]); c0 = int(src_pos[1] - origin2[1])
        r1 = int(act_pos[0] - origin2[0]); c1 = int(act_pos[1] - origin2[1])
        if 0 <= r0 < src_sub2.shape[0] and 0 <= c0 < src_sub2.shape[1]:
            src_sub2[r0, c0] = 1.0
        if 0 <= r1 < tgt_sub2.shape[0] and 0 <= c1 < tgt_sub2.shape[1]:
            tgt_sub2[r1, c1] = 1.0

        cd2, dir2 = cost_distance(src_sub2, res_sub2, eight=True)
        sp2 = trace_path(cd2, dir2, tgt_sub2, eight=True)
        if len(sp2) > 1:
            gpath2 = restore_path(origin2, sp2)
            if not _is_continuous_path(gpath2, eight=True):
                continue
            conn2 = [float(conn_ras[r, c])
                     for r, c in gpath2
                     if 0 <= r < conn_ras.shape[0]
                     and 0 <= c < conn_ras.shape[1]
                     and not np.isnan(conn_ras[r, c])]
            mw2 = float(np.mean(conn2)) if conn2 else 0.0
            sw2 = float(np.sum(conn2))  if conn2 else 0.0
            spine_results.append({
                'fromnode': nid, 'tonode': nid,
                'weight': mw2, 'sum_weight': sw2,
                'len': len(sp2), 'type': 'S',
                'path': gpath2,
            })

    return {
        'node_i': node_i, 'node_j': node_j,
        'mean_w': mean_w, 'sum_w': sum_w, 'path_len': len(path),
        'path': path,
        'quality': q,
        'spine_results': spine_results,
    }

def build_ecological_network(node_raster, core_raster,
                             connectivity_raster, resistance_raster,
                             adjacency_df, dataset, n_workers=None):
    if n_workers is None:
        env_workers = os.environ.get('ELT_N_WORKERS', '').strip()
        if env_workers.isdigit():
            n_workers = max(1, int(env_workers))
        else:
            n_workers = min(8, max(1, mp.cpu_count() - 1))

    proj_wkt = dataset.GetProjection()
    gt = dataset.GetGeoTransform()

    unique_nodes = np.unique(node_raster[~np.isnan(node_raster) & (node_raster > 0)]).astype(int)
    node_positions = {}
    for node in unique_nodes:
        pos = np.argwhere(node_raster == node)
        if len(pos) > 0:
            node_positions[int(node)] = (int(pos[0][0]), int(pos[0][1]))

    G = nx.Graph()
    G_act = nx.Graph()
    for node, (r, c) in node_positions.items():
        lon, lat = pixel_to_geo(dataset, c, r)
        G.add_node(node, lon=lon, lat=lat)
        G_act.add_node(node, lon=lon, lat=lat)

    core_rows, core_cols = core_raster.shape
    res_rows, res_cols = resistance_raster.shape
    conn_rows, conn_cols = connectivity_raster.shape

    pairs = []
    for _, row in adjacency_df.iterrows():
        node_i = int(row['Block'])
        node_j = int(row['Adjacent'])
        pos_i = node_positions.get(node_i)
        pos_j = node_positions.get(node_j)
        if pos_i is None or pos_j is None:
            continue
        pi = (int(row['NearestP_Block1_X']), int(row['NearestP_Block1_Y']))
        pj = (int(row['NearestP_Block2_X']), int(row['NearestP_Block2_Y']))
        pairs.append((node_i, node_j, pi, pj, pos_i, pos_j))

    env_max_pairs = os.environ.get('ELT_MAX_PAIRS', '').strip()
    if env_max_pairs.isdigit():
        max_pairs = max(1, int(env_max_pairs))
        pairs = pairs[:max_pairs]

    print(f"构建廊道（共 {len(pairs)} 对邻接斑块，使用 {n_workers} 进程）...")

    results = []
    skipped = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(core_raster, resistance_raster, connectivity_raster, gt, proj_wkt),
    ) as executor:
        for res in tqdm(
            executor.map(_build_single_corridor, pairs, chunksize=max(1, len(pairs) // (n_workers * 4))),
            total=len(pairs), desc="廊道构建（并行）"
        ):
            if res is None:
                skipped += 1
            else:
                results.append(res)

    geom_lines, attr_lines = [], []
    geom_points, attr_points = [], []
    act_node_id = int(max(node_positions.keys())) + 1 if node_positions else 1
    line_id = 0

    for node, (r, c) in node_positions.items():
        lon, lat = pixel_to_geo(dataset, c, r)
        geom_points.append(Point(lon, lat))
        attr_points.append({"nodeid": node, "sourceid": node, "type": 'S'})

    for res in results:
        line_id += 1
        act_i = act_node_id; act_node_id += 1
        act_j = act_node_id; act_node_id += 1

        G.add_edge(res['node_i'], res['node_j'],
                   weight=res['mean_w'], sum_weight=res['sum_w'],
                   distance=res['path_len'],
                   cost_sum=float(res.get('quality', {}).get('cost_sum', 0.0)),
                   cost_mean=float(res.get('quality', {}).get('cost_mean', 0.0)),
                   turns=int(res.get('quality', {}).get('turns', 0)),
                   len_w=float(res.get('quality', {}).get('len_w', 0.0)),
                   straight=float(res.get('quality', {}).get('straight', 0.0)))

        pix_path = res['path']
        geo_path = [pixel_to_geo(dataset, c, r) for r, c in pix_path]
        fp = geo_path[0]
        tp = geo_path[-1]
        geom_lines.append(LineString(geo_path))
        attr_lines.append({
            "fromnode": res['node_i'], "tonode": res['node_j'],
            "fromX": fp[0], "fromY": fp[1],
            "toX": tp[0], "toY": tp[1],
            "weight": res['mean_w'], "sum_weight": res['sum_w'],
            "len": res['path_len'], "type": 'T', "lineid": line_id,
            "cost_sum": float(res.get('quality', {}).get('cost_sum', 0.0)),
            "cost_mean": float(res.get('quality', {}).get('cost_mean', 0.0)),
            "turns": int(res.get('quality', {}).get('turns', 0)),
            "len_w": float(res.get('quality', {}).get('len_w', 0.0)),
            "straight": float(res.get('quality', {}).get('straight', 0.0)),
        })

        G_act.add_node(act_i, lon=fp[0], lat=fp[1])
        G_act.add_node(act_j, lon=tp[0], lat=tp[1])
        G_act.add_edge(
            act_i, act_j,
            weight=res['mean_w'],
            sum_weight=res['sum_w'],
            distance=res['path_len'],
            lineid=line_id,
            cost_sum=float(res.get('quality', {}).get('cost_sum', 0.0)),
            cost_mean=float(res.get('quality', {}).get('cost_mean', 0.0)),
            turns=int(res.get('quality', {}).get('turns', 0)),
            len_w=float(res.get('quality', {}).get('len_w', 0.0)),
            straight=float(res.get('quality', {}).get('straight', 0.0)),
        )
        geom_points.extend([Point(fp), Point(tp)])
        attr_points.extend([
            {"nodeid": act_i, "sourceid": res['node_i'], "type": 'A'},
            {"nodeid": act_j, "sourceid": res['node_j'], "type": 'A'},
        ])

        for sp in res['spine_results']:
            line_id += 1
            pix_spine = sp['path']
            geo_spine = [pixel_to_geo(dataset, c, r) for r, c in pix_spine]
            from_geo = geo_spine[0]
            to_geo = geo_spine[-1]
            geom_lines.append(LineString(geo_spine))
            attr_lines.append({
                "fromnode": sp['fromnode'], "tonode": sp['tonode'],
                "fromX": from_geo[0], "fromY": from_geo[1],
                "toX": to_geo[0], "toY": to_geo[1],
                "weight": sp['weight'], "sum_weight": sp['sum_weight'],
                "len": sp['len'], "type": sp['type'], "lineid": line_id,
            })

    if skipped:
        print(f"跳过 {skipped} 条廊道（路径过短或穿过其他斑块）")

    gdf_lines = gpd.GeoDataFrame(attr_lines, crs=dataset.GetProjection(), geometry=geom_lines)
    if len(gdf_lines) > 0:
        gdf_lines['_wkb'] = gdf_lines.geometry.apply(lambda g: g.wkb)
        gdf_lines = gdf_lines.loc[
            gdf_lines.groupby('_wkb')['lineid'].idxmin()
        ].drop(columns='_wkb').reset_index(drop=True)

    gdf_points = gpd.GeoDataFrame(attr_points, crs=dataset.GetProjection(), geometry=geom_points)
    if len(gdf_points) > 0:
        gdf_points['_wkb'] = gdf_points.geometry.apply(lambda g: g.wkb)
        gdf_points = gdf_points.loc[
            gdf_points.groupby('_wkb')['nodeid'].idxmin()
        ].drop(columns='_wkb').reset_index(drop=True)

    return G, G_act, gdf_lines, gdf_points

if __name__ == '__main__':
    print("="*60)
    print("生态廊道构建工具 - 最小累积阻力模型（并行版）")
    print("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir  = os.path.join(base_dir, 'InputData')
    output_dir  = os.path.join(base_dir, 'OutputData')
    os.makedirs(output_dir, exist_ok=True)

    node_raster_path    = os.path.join(output_dir, 'mspa_core_center.tif')
    mspa_core_path      = os.path.join(output_dir, 'mspa_core_filled.tif')
    connectivity_path   = os.path.join(input_dir,  'Connectivity.tif')
    resistance_path     = os.path.join(input_dir,  'ResistanceSurface.tif')
    adj_csv_path        = os.path.join(output_dir, 'adjacency_parallel.csv')

    shp_line  = os.path.join(output_dir, 'ecological_network.shp')
    shp_point = os.path.join(output_dir, 'ecological_network_point.shp')

    print("读取栅格数据...")
    node_ds = gdal.Open(node_raster_path)
    node_raster, _, _  = GeoImgR(node_raster_path)
    core_raster,  _, _ = GeoImgR(mspa_core_path)
    conn_raster,  _, _ = GeoImgR(connectivity_path)
    res_raster,   _, _ = GeoImgR(resistance_path)

    adjacency_df = pd.read_csv(adj_csv_path)
    print(f"邻接斑块对数: {len(adjacency_df)}")

    print("构建生态网络...")
    G, G_act, gdf_lines, gdf_points = build_ecological_network(
        node_raster, core_raster, conn_raster, res_raster,
        adjacency_df, node_ds
    )
    del node_ds

    print("保存结果...")
    nx.write_graphml(G,     os.path.join(output_dir, "ecological_network.graphml"))
    nx.write_graphml(G_act, os.path.join(output_dir, "ecological_network_act.graphml"))
    gdf_lines.to_file(shp_line)
    print(f"  廊道矢量: {shp_line}  ({len(gdf_lines)} 条)")
    gdf_points.to_file(shp_point)
    print(f"  节点矢量: {shp_point} ({len(gdf_points)} 个)")
    print("完成！")
