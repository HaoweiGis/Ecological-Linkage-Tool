# 执行时间记录（geobase；InputData -> OutputData；2026-04-16）
# 本脚本：26.56 s（19 线程）

import numpy as np
from osgeo import gdal
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion
from concurrent.futures import ThreadPoolExecutor

def GeoImgR(filename):
    dataset = gdal.Open(filename)
    im_porj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
    im_data = np.array(dataset.ReadAsArray())
    nodata_value = dataset.GetRasterBand(1).GetNoDataValue()
    if im_data.ndim == 2:
        im_data = im_data[np.newaxis, :, :]
    del dataset
    return im_data, im_porj, im_geotrans, nodata_value

def get_internal_boundaries(raster, nodata_value=0):
    unique_labels = np.unique(raster)
    boundaries = np.zeros_like(raster, dtype=np.int32)
    struct = np.ones((3, 3), dtype=bool)
    for label in unique_labels:
        if label == nodata_value:
            continue
        patch = (raster == label)
        eroded_patch = binary_erosion(patch, structure=struct)
        boundary = patch & ~eroded_patch
        boundaries[boundary] = int(label)
    return boundaries

def _nearest_pair_for_row(row, boundary_positions):
    b1 = int(row.Block)
    b2 = int(row.Adjacent)
    p1 = boundary_positions.get(b1)
    p2 = boundary_positions.get(b2)
    if p1 is None or len(p1) == 0 or p2 is None or len(p2) == 0:
        return None
    dists = cdist(p1, p2)
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    return {
        'Block': b1,
        'Adjacent': b2,
        'NearestP_Block1_X': int(p1[idx[0]][0]),
        'NearestP_Block1_Y': int(p1[idx[0]][1]),
        'NearestP_Block2_X': int(p2[idx[1]][0]),
        'NearestP_Block2_Y': int(p2[idx[1]][1]),
        'Distance': float(dists[idx]),
    }


def find_nearest_pairs(boundary_raster, adjacency_df, n_workers=1):
    unique_labels = np.unique(boundary_raster)
    unique_labels = unique_labels[unique_labels != 0]
    boundary_positions = {
        int(lbl): np.argwhere(boundary_raster == lbl)
        for lbl in unique_labels
    }
    rows = list(adjacency_df.itertuples(index=False))
    if n_workers is None or n_workers < 2:
        records = []
        for row in tqdm(rows, total=len(rows), desc="最近点对"):
            rec = _nearest_pair_for_row(row, boundary_positions)
            if rec is not None:
                records.append(rec)
        return pd.DataFrame(records)

    records = []
    with ThreadPoolExecutor(max_workers=int(n_workers)) as executor:
        for rec in tqdm(
            executor.map(lambda r: _nearest_pair_for_row(r, boundary_positions), rows),
            total=len(rows),
            desc=f"最近点对（{int(n_workers)}线程）",
        ):
            if rec is not None:
                records.append(rec)
    return pd.DataFrame(records)

if __name__ == '__main__':
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'OutputData')
    os.makedirs(output_dir, exist_ok=True)

    input_raster_path = os.path.join(output_dir, 'mspa_core_filled.tif')
    adj_csv_path      = os.path.join(output_dir, 'adjacency_parallel.csv')

    print("读取源地栅格...")
    line_raster, im_porj, im_geotrans, nodata_value = GeoImgR(input_raster_path)

    print("提取边界像素...")
    boundary_raster = get_internal_boundaries(line_raster[0], nodata_value)

    print("读取邻接关系...")
    adjacency_df = pd.read_csv(adj_csv_path)

    print("计算邻接斑块间的最近点对...")
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    result_df = find_nearest_pairs(boundary_raster, adjacency_df, n_workers=n_workers)

    result_df.to_csv(adj_csv_path, index=False)
    print(f"邻接+最近点已保存: {adj_csv_path}")
    print(f"有效邻接对: {len(result_df)}")
