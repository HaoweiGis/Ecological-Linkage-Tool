import numpy as np
from osgeo import gdal, osr
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
from scipy.ndimage import binary_erosion
from multiprocessing import Pool, cpu_count

def GeoImgR(filename):
    dataset = gdal.Open(filename)
    im_porj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
    im_data = np.array(dataset.ReadAsArray())
    nodata_value = dataset.GetRasterBand(1).GetNoDataValue()
    if len(im_data.shape) == 2:
        im_data = im_data[np.newaxis, :, :]
    del dataset
    return im_data, im_porj, im_geotrans, nodata_value

def GeoImgW(filename, im_data, im_geotrans, im_porj, nodata, driver='GTiff'):
    im_shape = im_data.shape
    driver = gdal.GetDriverByName(driver)
    datetype = gdal.GDT_UInt32
    dataset = driver.Create(filename, im_shape[2], im_shape[1], im_shape[0], datetype,
                            options=["TILED=YES", "COMPRESS=LZW"])
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_porj)
    for band_num in range(im_shape[0]):
        img = im_data[band_num, :, :]
        raster_band = dataset.GetRasterBand(band_num + 1)
        raster_band.SetNoDataValue(nodata)
        raster_band.WriteArray(img)
    del dataset

# 获取边界像素
def get_internal_boundaries(raster, nodata_value):
    unique_labels = np.unique(raster)
    boundaries = np.zeros_like(raster)
    structure = np.ones((3, 3), dtype=int)
    for label in unique_labels:
        if label == nodata_value:
            continue
        patch = (raster == label)
        eroded_patch = binary_erosion(patch, structure)
        boundary = patch & ~eroded_patch
        boundaries[boundary] = label
    return boundaries

# 读取邻接关系
def read_adjacency_csv(csv_path):
    return pd.read_csv(csv_path)

# ******************************************************************Single thread
# 获取相邻斑块之间的最近点
def get_near_points_single(boundary_raster, adjacency_df):
    unique_labels = np.unique(boundary_raster)
    unique_labels = unique_labels[unique_labels != 0]
    
    boundary_positions = {label: np.argwhere(boundary_raster == label) for label in unique_labels}
    
    new_raster = np.zeros(boundary_raster.shape, dtype=np.uint32)
    nearest_pairs = []
    
    for _, row in tqdm(adjacency_df.iterrows(), total=len(adjacency_df), desc="Processing Rows"):
        block1 = row['Block']
        block2 = row['Adjacent']
        pixels1 = boundary_positions.get(block1, [])
        pixels2 = boundary_positions.get(block2, [])
        
        # 使用 scipy.spatial.distance.cdist 加速计算距离矩阵
        dists = distance.cdist(pixels1, pixels2)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_dist = dists[min_idx]

        nearest_pair = (tuple(pixels1[min_idx[0]]), tuple(pixels2[min_idx[1]]))
        nearest_pairs.append((block1, block2, nearest_pair[0][0], nearest_pair[0][1], nearest_pair[1][0], nearest_pair[1][1], min_dist))
        
        if nearest_pair[0] and nearest_pair[1]:
            new_raster[nearest_pair[0]] = block1
            new_raster[nearest_pair[1]] = block2
    
    # 将最近点对添加到 adjacency_df
    nearest_pairs_df = pd.DataFrame(nearest_pairs, columns=['Block', 'Adjacent', 'NearestP_Block1_X','NearestP_Block1_Y', 'NearestP_Block2_X', 'NearestP_Block2_Y', 'Distance'])
    adjacency_df = pd.concat([adjacency_df, nearest_pairs_df[['NearestP_Block1_X','NearestP_Block1_Y', 'NearestP_Block2_X', 'NearestP_Block2_Y', 'Distance']]], axis=1)

    return new_raster,adjacency_df

# ******************************************************************Parallel
def process_row(args):
    index, row, boundary_positions = args
    block1 = row['Block']
    block2 = row['Adjacent']
    pixels1 = boundary_positions.get(block1, [])
    pixels2 = boundary_positions.get(block2, [])
    
    # if len(pixels1) == 0 or len(pixels2) == 0:
    #     return index, None  # 跳过没有像素的块
    
    dists = distance.cdist(pixels1, pixels2)
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    min_dist = dists[min_idx]

    nearest_pair = (tuple(pixels1[min_idx[0]]), tuple(pixels2[min_idx[1]]))
    return index, (block1, block2, nearest_pair[0][0], nearest_pair[0][1], nearest_pair[1][0], nearest_pair[1][1], min_dist)

def get_near_points_parallel(boundary_raster, adjacency_df):
    unique_labels = np.unique(boundary_raster)
    unique_labels = unique_labels[unique_labels != 0]
    
    boundary_positions = {label: np.argwhere(boundary_raster == label) for label in unique_labels}
    
    new_raster = np.zeros(boundary_raster.shape, dtype=np.uint32)

    # 设置并行处理
    num_workers = cpu_count()
    with Pool(num_workers) as pool:
        args = ((index, row, boundary_positions) for index, row in adjacency_df.iterrows())
        results = list(tqdm(pool.imap(process_row, args), total=len(adjacency_df), desc="Processing Rows"))

    nearest_pairs = [(index, pair) for index, pair in results if pair is not None]  # 移除 None 值
    
    # 确保最近点对和原始数据对应
    nearest_pairs_df = pd.DataFrame([pair for index, pair in nearest_pairs], 
                                    columns=['Block', 'Adjacent', 'NearestP_Block1_X','NearestP_Block1_Y', 'NearestP_Block2_X', 'NearestP_Block2_Y', 'Distance'],
                                    index=[index for index, pair in nearest_pairs])

    # 更新 new_raster
    for _, pair in nearest_pairs:
        block1, block2, x1, y1, x2, y2, min_dist = pair
        new_raster[x1, y1] = block1
        new_raster[x2, y2] = block2

    return new_raster, nearest_pairs_df


if __name__ == '__main__':
    # GeoTIFF文件路径
    input_raster_path = r'InputData\mspa_core_filled.tif'
    adj_csv_path =r'OutputData/adjacency_parallel.csv'
    actP_raster_path = r'OutputData/mspa_core_activate_parallel.tif'

    line_raster, im_porj, im_geotrans, nodata_value = GeoImgR(input_raster_path)
    boundary_raster = get_internal_boundaries(line_raster[0], nodata_value)
    adjacency_df = read_adjacency_csv(adj_csv_path)

    # 选择是否并行
    new_raster,adjacency_df = get_near_points_single(boundary_raster, adjacency_df)
    # new_raster,adjacency_df = get_near_points_parallel(boundary_raster, adjacency_df)
    adjacency_df.to_csv(adj_csv_path, index=False)

    GeoImgW(actP_raster_path, new_raster[np.newaxis, :, :], im_geotrans, im_porj, nodata=0, driver='GTiff')
    print("ActivatePoints计算完成，结果已保存为栅格文件。")

    # screen -L -Logfile 2_adj_single.log python 2_GetActivatePoint.py
    # screen -L -Logfile 2_adj_parallel.log python 2_GetActivatePoint.py