import geopandas as gpd
import math
import numpy as np

def calculate_azimuth(p1, p2):
    """计算两个点之间的方位角（0-360度范围内）。"""
    angle = math.atan2(p2.x - p1.x, p2.y - p1.y)
    return (math.degrees(angle) + 360) % 360

def calculate_average_direction(line):
    """计算一条线的平均方向。"""
    angles = [
        calculate_azimuth(line.interpolate(i / (len(line.coords) - 1)),
                          line.interpolate((i + 1) / (len(line.coords) - 1)))
        for i in range(len(line.coords) - 1)
    ]
    # 通过单位圆上的平均向量计算平均方向
    x = np.mean([math.cos(math.radians(a)) for a in angles])
    y = np.mean([math.sin(math.radians(a)) for a in angles])
    return math.degrees(math.atan2(y, x)) % 360

def get_direction_category(average_di, tolerance=1):
    """根据平均方向分类方向类别，允许一定的误差范围。"""
    rounded_di = round(average_di)  # 将方向四舍五入到整数
    if abs(rounded_di % 360 - 0) <= tolerance or abs(rounded_di % 360 - 180) <= tolerance:
        return 'NS'
    elif abs(rounded_di % 360 - 90) <= tolerance or abs(rounded_di % 360 - 270) <= tolerance:
        return 'EW'
    elif abs(rounded_di % 360 - 45) <= tolerance or abs(rounded_di % 360 - 225) <= tolerance:
        return 'NESW'
    elif abs(rounded_di % 360 - 135) <= tolerance or abs(rounded_di % 360 - 315) <= tolerance:
        return 'SENW'
    else:
        return 'Unknown'

def calculate_flow_and_sources(row):
    """
    根据flow_f2t和flow_t2f确定流向方向，以及动态字段，并标记flow_type。
    """
    if row['flow_f2t'] > row['flow_t2f']:
        flow_direction = 'from_to'
        from_source = row['fromnode']
        to_source = row['tonode']
        from_flow = row['flow_f2t']
        to_flow = row['flow_t2f']
        flow_type = 'source_source'
    elif row['flow_f2t'] < row['flow_t2f']:
        flow_direction = 'to_from'
        from_source = row['tonode']
        to_source = row['fromnode']
        from_flow = row['flow_t2f']
        to_flow = row['flow_f2t']
        flow_type = 'source_source'
    else:
        # flow_f2t == flow_t2f
        flow_direction = 'edge_source'
        from_source = row['fromnode']
        to_source = row['tonode']
        from_flow = row['flow_f2t']
        to_flow = row['flow_t2f']
        flow_type = 'edge_source'
    
    return flow_direction, from_source, to_source, from_flow, to_flow, flow_type

def assign_specific_direction(row, direction_map):
    """
    根据 direction_category 和 flow_direction，计算具体方向。
    """
    direction_category = row['direction_category']
    flow_direction = row['flow_direction']

    if flow_direction == 'edge_source':
        # 直接使用 direction_category 作为 edge_source 的方向
        return direction_category
    else:
        # 处理 source_source 情况，按照原逻辑基于 flow_direction 和 direction_category 确定
        if flow_direction == 'from_to':
            main_direction = 'forward'
        else:
            main_direction = 'reverse'

        if direction_category == 'NS':
            return 'N2S' if main_direction == 'forward' else 'S2N'
        elif direction_category == 'EW':
            return 'W2E' if main_direction == 'forward' else 'E2W'
        elif direction_category == 'NESW':
            return 'SW2NE' if main_direction == 'forward' else 'NE2SW'
        elif direction_category == 'SENW':
            return 'NW2SE' if main_direction == 'forward' else 'SE2NW'
        else:
            return 'Unknown'

# 加载包含方向属性的shp文件
input_shp = r'D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\1_HuLine\Huline_corridor\huline300mV2\ecological_network1107_mcr_crs_ailurus_ful.shp'
output_shp = r'D:\2_HaoweiPapers\4_ecologicalSFlow\2_output\1_HuLine\Huline_corridor\huline300mV2\ecological_network1107_mcr_crs_ailurus_ful_dir.shp'

gdf = gpd.read_file(input_shp)

# 添加average_direction
gdf['average_direction'] = gdf.geometry.apply(calculate_average_direction)

# 添加方向分类
gdf['direction_category'] = gdf['average_direction'].apply(get_direction_category)

# 检查未知方向
if (gdf['direction_category'] == 'Unknown').any():
    print("警告：部分线的方向无法分类，请检查数据！")

# 添加流向和动态字段，包括flow_type
flow_results = gdf.apply(calculate_flow_and_sources, axis=1)
gdf['flow_direction'], gdf['from_source'], gdf['to_source'], gdf['from_flow'], gdf['to_flow'], gdf['flow_type'] = zip(*flow_results)

# 创建 direction_map 存储每个 source_source 的方向
direction_map = {}

# 先计算并存储所有 source_source 类型的 specific_direction
for idx, row in gdf.iterrows():
    if row['flow_type'] == 'source_source':
        # 计算 source_source 的 specific_direction
        direction_map[row['from_source']] = assign_specific_direction(row, direction_map)

# 为所有 'edge_source' 类型的行添加具体方向
gdf['specific_direction'] = gdf.apply(assign_specific_direction, axis=1, direction_map=direction_map)

# 导出为新的shapefile
gdf.to_file(output_shp, driver='ESRI Shapefile')
print(f"新文件已保存到: {output_shp}")
