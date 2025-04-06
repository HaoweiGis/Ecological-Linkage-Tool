from tkinter import Button
import numpy as np
from osgeo import gdal,ogr,osr
import argparse
import pandas as pd
import os,re,tqdm
import os.path as osp
from osgeo import gdalconst
import rasterio
import tqdm
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString

def GeoImgR(filename):
    dataset = gdal.Open(filename)
    im_porj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
    im_data = np.array(dataset.ReadAsArray())
    del dataset
    return im_data, im_porj, im_geotrans

def GeoImgW(filename,im_data, im_geotrans, im_porj,nodata, driver='GTiff'):
    im_shape = im_data.shape
    driver = gdal.GetDriverByName(driver)
    if "int8" in im_data.dtype.name:
        datetype = gdal.GDT_Byte
    elif "int16" in im_data.dtype.name:
        datetype = gdal.GDT_UInt16
    elif "int32" in im_data.dtype.name:
        datetype = gdal.GDT_UInt32
    else :
        datetype = gdal.GDT_Float32
    datetype = gdal.GDT_Float32
    # driver.Create weight hight
    dataset = driver.Create(filename, im_shape[2], im_shape[1], im_shape[0], datetype,
    options=["TILED=YES", "COMPRESS={0}".format("LZW")])
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_porj)
    for band_num in range(im_shape[0]):
        img = im_data[band_num,:,:]
        band_num = band_num + 1
        raster_band = dataset.GetRasterBand(band_num)
        raster_band.SetNoDataValue(nodata)
        # raster_band.SetDescription(bandNames[band_num-1])
        raster_band.WriteArray(img)
    del dataset

def pixel_geo_register(infile,outfile,reffile,methods,nodata=0):
    '''
    infile:输入文件
    outfile:输出文件
    reffile:参考文件
    methods:重采样方法
            gdalconst.GRA_NearestNeighbour：near
            gdalconst.GRA_Bilinear:bilinear
            gdalconst.GRA_Cubic:cubic
            gdalconst.GRA_CubicSpline:cubicspline
            gdalconst.GRA_Lanczos:lanczos
            gdalconst.GRA_Average:average
            gdalconst.GRA_Mode:mode
    '''
    in_ds  = gdal.Open(infile, gdalconst.GA_ReadOnly)
    ref_ds = gdal.Open(reffile, gdalconst.GA_ReadOnly)

    in_trans = in_ds.GetGeoTransform()
    in_proj = in_ds.GetProjection()
    ref_trans = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()

    band_ref = ref_ds.GetRasterBand(1)
    
    x = ref_ds.RasterXSize 
    y = ref_ds.RasterYSize
    
    driver= gdal.GetDriverByName('GTiff')
    output = driver.Create(outfile, x, y, 1, gdalconst.GDT_Float32,options=["TILED=YES", "COMPRESS={0}".format("LZW")])
    # output = driver.Create(outfile, x, y, 1, gdalconst.GDT_Byte,options=["TILED=YES", "COMPRESS={0}".format("LZW")])
    output.SetGeoTransform(ref_trans)
    output.SetProjection(ref_proj)
    # raster_band = output.GetRasterBand(1)
    # raster_band.SetNoDataValue(nodata)
    gdal.ReprojectImage(in_ds, output, in_proj, ref_proj, methods)
    
    in_ds = None
    ref_ds = None
    driver  = None
    output = None

if __name__ == "__main__":

    # ********************************************************计算生物流
    # 文件路径
    patch_raster_path = r"D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\1_HuLine\EcologialNetInput\Huline300m\MSPA_Core1105\MSPA_Core.tif"
    habitat_quality_raster_path = r"D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\all_species_maxent\habitat_quality_mask.tif"
    biodiversity_raster_path = r"D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\all_species_maxent\ailurus_ful_pro_nor.tif"

    # 读取斑块数据
    with rasterio.open(patch_raster_path) as patch_src:
        patch_data = patch_src.read(1)  # 读取第一个波段
        patch_nodata = patch_src.nodata
        pixel_size = patch_src.transform[0]  # 像元分辨率（假设方形像元）

    # 读取生境质量数据
    with rasterio.open(habitat_quality_raster_path) as habitat_src:
        habitat_data = habitat_src.read(1)

    # 读取生物多样性数据
    with rasterio.open(biodiversity_raster_path) as biodiversity_src:
        biodiversity_data = biodiversity_src.read(1)

    # 无效值设置
    habitat_nodata = -9999
    biodiversity_nodata = -9999

    # 获取唯一斑块ID
    patch_ids = np.unique(patch_data)
    if patch_nodata is not None:
        patch_ids = patch_ids[patch_ids != patch_nodata]  # 去除无数据值

    # 初始化结果列表
    results = []

    # 像元面积（单位：km²）
    pixel_area_km2 = (pixel_size / 1000) ** 2

    # 遍历每个斑块ID
    for patch_id in tqdm.tqdm(patch_ids):
        # 创建布尔掩膜
        mask = (patch_data == patch_id)
        
        # 提取对应的生境质量和生物多样性数据
        habitat_values = habitat_data[mask]
        biodiversity_values = biodiversity_data[mask]
        
        # 排除无效值
        valid_mask = (habitat_values != habitat_nodata) & (biodiversity_values != biodiversity_nodata)
        habitat_values = habitat_values[valid_mask]
        biodiversity_values = biodiversity_values[valid_mask]
        
        # 计算统计值
        if len(habitat_values) > 0 and len(biodiversity_values) > 0:
            habitat_mean = np.nanmean(habitat_values)  # 生境质量均值
            biodiversity_mean = np.nanmean(biodiversity_values)  # 生物多样性均值
            
            # 计算斑块面积（单位：km²）
            patch_area_km2 = np.sum(mask) * pixel_area_km2
            
            # 计算Pot_Bio_flow和Disp_attr
            log_area = np.log(patch_area_km2) if patch_area_km2 > 0 else 0
            pot_bio_flow = biodiversity_mean * log_area
            disp_attr = biodiversity_mean * habitat_mean * log_area
            
            # 将结果保存到列表
            results.append({
                "Patch_ID": patch_id,
                "Patch_Area_km2": patch_area_km2,
                "Habitat_Quality_Mean": habitat_mean,
                "Biodiversity_Mean": biodiversity_mean,
                "Pot_Bio_flow": pot_bio_flow,
                "Disp_attr": disp_attr
            })

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 对Disp_attr字段进行归一化
    if "Disp_attr" in df.columns:
        disp_attr_min = df["Disp_attr"].min()
        disp_attr_max = df["Disp_attr"].max()
        if disp_attr_max > disp_attr_min:  # 避免除零错误
            df["Disp_attr_Nor"] = (df["Disp_attr"] - disp_attr_min) / (disp_attr_max - disp_attr_min)
        else:
            df["Disp_attr_Nor"] = 0  # 如果最大值等于最小值，则所有值归一化为0

    # 保存为CSV
    
    output_csv_path = r"D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\all_species_maxent\patch_statistics_with_area_ailurus_ful.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"结果已保存到 {output_csv_path}")



    # # ********************************************************计算结合廊道生物流
    # 文件路径
    shp_path = r"D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\1_HuLine\Huline_corridor\huline300mV2\ecological_network1107_mcr_crs.shp"
    csv_path = r"D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\all_species_maxent\patch_statistics_with_area_ailurus_ful.csv"
    output_shp_path = r"D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\1_HuLine\Huline_corridor\huline300mV2\ecological_network1107_mcr_crs_ailurus_ful.shp"

    # 读取廊道线数据
    shp_data = gpd.read_file(shp_path)

    # 读取 patch statistics 数据
    patch_data = pd.read_csv(csv_path)

    # 创建 Patch_ID 为索引的字典，便于快速查找
    patch_dict = patch_data.set_index("Patch_ID").to_dict("index")

    # 初始化新字段
    shp_data["len"] = 0.0  # 单位：km
    shp_data["Disp_prob"] = 0.0
    shp_data["flow_f2t"] = 0.0
    shp_data["flow_t2f"] = 0.0

    # 遍历每条线，计算相关字段
    for idx, row in shp_data.iterrows():
        fromnode = row["fromnode"]
        tonode = row["tonode"]
        lineid = row["lineid"]
        geometry = row["geometry"]
        
        # 计算线长度（单位：km，假设投影单位为米）
        line_length_km = geometry.length / 1000.0  # 从米转换为千米
        shp_data.at[idx, "len"] = line_length_km

        # 计算 Disp_prob
        if line_length_km > 0:
            disp_prob = np.exp(-0.05*np.log(line_length_km))  # 修正后的公式
        else:
            disp_prob = 0
        shp_data.at[idx, "Disp_prob"] = disp_prob

        # 获取 fromnode 和 tonode 的 patch 属性
        if fromnode in patch_dict and tonode in patch_dict:
            from_patch = patch_dict[fromnode]
            to_patch = patch_dict[tonode]
            
            from_pot_bio_flow = from_patch["Pot_Bio_flow"]
            to_pot_bio_flow = to_patch["Pot_Bio_flow"]
            from_disp_attr_nor = from_patch["Disp_attr_Nor"]
            to_disp_attr_nor = to_patch["Disp_attr_Nor"]

            # 计算流量
            min_pot_bio_flow = min(from_pot_bio_flow, to_pot_bio_flow)
            flow_f2t = min_pot_bio_flow * disp_prob * to_disp_attr_nor
            flow_t2f = min_pot_bio_flow * disp_prob * from_disp_attr_nor
            
            # 更新字段
            shp_data.at[idx, "flow_f2t"] = flow_f2t
            shp_data.at[idx, "flow_t2f"] = flow_t2f

    # 保存结果为新的 SHP 文件
    shp_data.to_file(output_shp_path)

    print(f"结果已保存到 {output_shp_path}")


