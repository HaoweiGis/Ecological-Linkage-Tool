# 执行时间记录（geobase；InputData -> OutputData；2026-04-16）
# 本脚本：1.56 s

import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import distance_transform_edt

def read_raster(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    nodata_value = band.GetNoDataValue()
    return array, dataset, nodata_value

def write_raster(file_path, array, dataset, nodata_value):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(
        file_path, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Int32,
        options=["TILED=YES", "COMPRESS=LZW"]
    )
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(nodata_value)
    out_band.FlushCache()
    del out_dataset

def euclidean_allocation(raster_array, nodata_value):
    valid_mask = (raster_array != nodata_value).astype(int)
    _, indices = distance_transform_edt(1 - valid_mask, return_indices=True)
    allocation = raster_array[indices[0], indices[1]]
    return allocation

def compute_adjacency(allocation, nodata_value=0, validate=False):
    pair_chunks = []
    shifts = [
        (slice(None),      slice(None, -1),  slice(None),      slice(1, None)),
        (slice(None, -1),  slice(None),      slice(1, None),   slice(None)),
        (slice(None, -1),  slice(None, -1),  slice(1, None),   slice(1, None)),
        (slice(None, -1),  slice(1, None),   slice(1, None),   slice(None, -1)),
    ]
    for r1, c1, r2, c2 in shifts:
        v1 = allocation[r1, c1]
        v2 = allocation[r2, c2]
        mask = (v1 != v2) & (v1 > 0) & (v2 > 0)
        if nodata_value is not None:
            mask &= (v1 != nodata_value) & (v2 != nodata_value)
        if np.any(mask):
            pairs = np.column_stack((v1[mask], v2[mask]))
            pairs = np.sort(pairs, axis=1)
            pair_chunks.append(pairs.astype(np.int64, copy=False))

    if not pair_chunks:
        return []

    all_pairs = np.vstack(pair_chunks)
    unique_pairs = np.unique(all_pairs, axis=0)

    if validate:
        if np.any(unique_pairs[:, 0] == unique_pairs[:, 1]):
            raise ValueError("self-loop adjacency pair detected")
        if unique_pairs.shape[0] != len({(int(a), int(b)) for a, b in unique_pairs}):
            raise ValueError("duplicate adjacency pair detected")

    return [(int(a), int(b)) for a, b in unique_pairs]

def save_to_csv(adjacency_pairs, output_path):
    df = pd.DataFrame(adjacency_pairs, columns=['Block', 'Adjacent'])
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    import os

    base_dir = r'D:\2_HaoweiPapers\1_SOCIAndEco\Ecological-Linkage-Tool-main'
    output_dir = os.path.join(base_dir, 'OutputData')
    os.makedirs(output_dir, exist_ok=True)

    input_raster_path   = os.path.join(output_dir, 'mspa_core_filled.tif')
    output_allocation    = os.path.join(output_dir, 'mspa_core_filled_eucall.tif')
    adj_csv_path        = os.path.join(output_dir, 'adjacency_parallel.csv')

    print("读取栅格...")
    raster_array, dataset, nodata_value = read_raster(input_raster_path)
    nodata_int = int(nodata_value) if nodata_value is not None else 0

    print("欧氏分配...")
    allocation = euclidean_allocation(raster_array, nodata_int)

    print("保存分配栅格...")
    write_raster(output_allocation, allocation, dataset, 0)

    print("计算邻接关系（向量化扫描）...")
    adjacency_pairs = compute_adjacency(allocation, nodata_int, validate=True)
    print(f"  邻接对数: {len(adjacency_pairs)}")

    save_to_csv(adjacency_pairs, adj_csv_path)
    print(f"邻接关系计算完成，结果已保存为CSV文件: {adj_csv_path}")

    patch_ids = set(int(v) for v in np.unique(allocation) if v > 0)
    csv_ids = set()
    for b, a in adjacency_pairs:
        csv_ids.add(b); csv_ids.add(a)
    missing = patch_ids - csv_ids
    if missing:
        print(f"孤立斑块（无邻接）: {sorted(missing)}")
    else:
        print("所有斑块均有邻接关系")
