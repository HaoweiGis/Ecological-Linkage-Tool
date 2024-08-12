import numpy as np
from osgeo import gdal
from scipy.ndimage import distance_transform_edt
import cv2
import rasterio
import pandas as pd
import tqdm
from multiprocessing import Pool, cpu_count

# 读取栅格数据
def read_raster(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    nodata_value = band.GetNoDataValue()
    return array, dataset, nodata_value

# 保存栅格数据
def write_raster(file_path, array, dataset, nodata_value):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(file_path, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Int32)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(nodata_value)
    out_band.FlushCache()

# 欧几里得分配算法
def euclidean_allocation(raster_array, nodata_value):
    # 创建源点掩码
    valid_mask = (raster_array != nodata_value).astype(int)
    
    # 计算欧几里得距离
    distances, indices = distance_transform_edt(1 - valid_mask, return_indices=True)
    
    # 分配区域
    allocation = np.full_like(raster_array, nodata_value)
    for i in range(allocation.shape[0]):
        for j in range(allocation.shape[1]):
            allocation[i, j] = raster_array[indices[0, i, j], indices[1, i, j]]
    
    return distances, allocation

# 判断是否相邻
def is_adjacent(block1, block2):
    # 转换为8位无符号整数类型
    block1_uint8 = block1.astype(np.uint8)
    block2_uint8 = block2.astype(np.uint8)
    struct = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    dilated_block1 = cv2.dilate(block1_uint8, struct, iterations=1)
    return np.any(dilated_block1 & block2_uint8)

# ******************************************************************Single thread
# 计算邻接关系
def compute_adjacency_single(raster_array, nodata_value):
    unique_values = np.unique(raster_array)
    unique_values = unique_values[unique_values != nodata_value]
    
    adjacency_list = []
    
    for value1 in tqdm.tqdm(unique_values):
        block1 = (raster_array == value1)
        adjacent_blocks = []
        
        for value2 in unique_values:
            if value1 >= value2: #计算单向连通
                continue
            block2 = (raster_array == value2)
            if is_adjacent(block1, block2):
                adjacent_blocks.append(value2)
    
        adjacency_list.append((value1, adjacent_blocks))
    
    return adjacency_list

# ******************************************************************Parallel
# 计算某个值的邻接关系
def compute_adjacency_for_value(args):
    value1, unique_values, raster_array, nodata_value = args
    block1 = (raster_array == value1)
    adjacent_blocks = []

    for value2 in unique_values:
        if value1 >= value2: #计算单向连通
            continue
        block2 = (raster_array == value2)
        if is_adjacent(block1, block2):
            adjacent_blocks.append(value2)

    return (value1, adjacent_blocks)

# 计算邻接关系
def compute_adjacency_parallel(raster_array, nodata_value):
    unique_values = np.unique(raster_array)
    unique_values = unique_values[unique_values != nodata_value]
    
    args = [(value1, unique_values, raster_array, nodata_value) for value1 in unique_values]

    with Pool(processes=cpu_count()) as pool:
        adjacency_list = pool.map(compute_adjacency_for_value, args)

    return adjacency_list

# 读取邻接关系
def read_adjacency_csv(csv_path):
    return pd.read_csv(csv_path)

# 保存为CSV文件
def save_to_csv(adjacency_list, output_path):
    data = []
    for block, adjacents in adjacency_list:
        for adj in adjacents:
            data.append([block, adj])
    
    df = pd.DataFrame(data, columns=['Block', 'Adjacent'])
    df.to_csv(output_path, index=False)

# ******************************************************************用于验证的相关函数
# 读取影像文件并获取唯一值
def get_image_unique_values(image_path):
    with rasterio.open(image_path) as src:
        image_data = src.read(1)  # 读取第一波段数据
        unique_values = np.unique(image_data)
    return unique_values

# 读取CSV文件并获取['Block', 'Adjacent']列中的唯一值
def get_csv_unique_values(csv_path):
    df = pd.read_csv(csv_path)
    unique_values_block = df['Block'].unique()
    unique_values_adjacent = df['Adjacent'].unique()
    combined_unique_values = np.unique(np.concatenate((unique_values_block, unique_values_adjacent)))
    return combined_unique_values

# 查找在影像唯一值中存在而在CSV文件中缺失的值
def find_missing_values(image_unique_values, csv_unique_values):
    missing_values = np.setdiff1d(image_unique_values, csv_unique_values)
    return missing_values


# # 主程序
if __name__ == '__main__':
    input_raster_path = r'InputData\mspa_core_filled.tif'
    output_distance_path = r'OutputData\mspa_core_filled_eucdis.tif'
    output_allocation_path = r'OutputData\mspa_core_filled_eucall.tif'
    adj_csv_path =r'OutputData\adjacency_parallel.csv'

    # ******************************************************************计算邻接关系 step1
    # 读取输入栅格
    raster_array, dataset, nodata_value = read_raster(input_raster_path)

    # ******************************************************************空间分配
    # 计算欧几里得距离和分配区域
    distances, allocation = euclidean_allocation(raster_array, nodata_value)

    # 保存输出栅格
    # write_raster(output_distance_path, distances, dataset, nodata_value)
    write_raster(output_allocation_path, allocation, dataset, nodata_value)
    # print("欧几里得分配完成。")

    # ******************************************************************计算邻接关系 step2
    # 计算邻接关系
    # adjacency_list = compute_adjacency_single(allocation,nodata_value)
    adjacency_list = compute_adjacency_parallel(allocation,nodata_value)
    # 保存邻接关系为CSV文件
    save_to_csv(adjacency_list, adj_csv_path)
    print("邻接关系计算完成，结果已保存为CSV文件。")

    # ******************************************************************验证是否存在漏掉的斑块
    # 获取影像唯一值和CSV唯一值
    image_unique_values = get_image_unique_values(input_raster_path)
    csv_unique_values = get_csv_unique_values(adj_csv_path)
    # 查找缺失的值
    missing_values = find_missing_values(image_unique_values, csv_unique_values)
    # 打印缺失的值
    print("缺失的值:", missing_values)

    # screen -L -Logfile 1_adj_parallel.log python 1_AdjacentCore.py
    # screen -L -Logfile 1_adj_single.log python 1_AdjacentCore.py