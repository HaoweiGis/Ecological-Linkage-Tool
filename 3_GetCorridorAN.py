import numpy as np
import networkx as nx
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import label
from tqdm import tqdm
from scipy.spatial import distance
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import Point
import pandas as pd
import heapq

def GeoImgR(filename):
    dataset = gdal.Open(filename)
    im_data = np.array(dataset.ReadAsArray())
    im_porj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
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
    # datetype = gdal.GDT_Byte
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

def extract_subgrid(grid, point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # 计算两点之间的欧氏距离
    euclidean_distance = int(np.ceil(distance.euclidean(point1, point2)))

    # 计算子栅格的边界
    min_x = max(0, min(x1, x2) - euclidean_distance)
    max_x = min(grid.shape[0] - 1, max(x1, x2) + euclidean_distance)
    min_y = max(0, min(y1, y2) - euclidean_distance)
    max_y = min(grid.shape[1] - 1, max(y1, y2) + euclidean_distance)

    subgrid = grid[min_x:max_x + 1, min_y:max_y + 1]
    subgrid_origin = (min_x, min_y)

    return subgrid, subgrid_origin


def extract_centerP(grid, point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # 计算两点之间的欧氏距离
    euclidean_distance = int(np.ceil(distance.euclidean(point1, point2)))

    # 计算子栅格的边界
    min_x = max(0, min(x1, x2) - euclidean_distance)
    max_x = min(grid.shape[0] - 1, max(x1, x2) + euclidean_distance)
    min_y = max(0, min(y1, y2) - euclidean_distance)
    max_y = min(grid.shape[1] - 1, max(y1, y2) + euclidean_distance)

    subgrid = grid[min_x:max_x + 1, min_y:max_y + 1]
    new_point1 = (point1[0] - min_x, point1[1] - min_y)
    new_point2 = (point2[0] - min_x, point2[1] - min_y)
    subgrid1 = np.zeros(subgrid.shape)
    subgrid2 = np.zeros(subgrid.shape)
    subgrid1[new_point1] = 1
    subgrid2[new_point2] = 1
    subgrid_origin = (min_x, min_y)

    return subgrid1, subgrid2 , subgrid_origin

def restore_path_to_global(subgrid_origin, path):
    origin_x, origin_y = subgrid_origin
    global_path = [(x + origin_x, y + origin_y) for x, y in path]
    return global_path

def restore_subgrid_to_original(grid, subgrid, subgrid_origin):
    origin_x, origin_y = subgrid_origin
    subgrid_rows, subgrid_cols = subgrid.shape

    grid[origin_x:origin_x + subgrid_rows, origin_y:origin_y + subgrid_cols] = subgrid
    return grid

def pixel_to_geo(dataset, x, y):
    transform = dataset.GetGeoTransform()
    px = transform[0] + x * transform[1] + y * transform[2]
    py = transform[3] + x * transform[4] + y * transform[5]

    source = osr.SpatialReference()
    source.ImportFromWkt(dataset.GetProjection())
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)  # WGS84
    transform = osr.CoordinateTransformation(source, target)
    lat, lon, _  = transform.TransformPoint(px, py)
    return lon, lat

def create_sparse_graph(connectivity_raster):
    rows, cols = connectivity_raster.shape
    graph = nx.DiGraph()
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(connectivity_raster[i, j]):
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                    if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(connectivity_raster[ni, nj]):
                        graph.add_edge((i, j), (ni, nj), weight=connectivity_raster[ni, nj])
    return graph

def convert_to_int_tuple(np_tuple):
    return tuple(map(int, np_tuple))

# 读取邻接关系
def read_adjacency_csv(csv_path):
    return pd.read_csv(csv_path)

def cost_distance_with_direction(source_ras, cost_ras, eight_directions=True):
    rows, cols = cost_ras.shape
    cost_dist = np.full((rows, cols), np.inf)
    direction = np.full((rows, cols), -1)
    visited = np.zeros((rows, cols), dtype=bool)
    
    # Priority queue for Dijkstra algorithm
    pq = []

    # Initialize source points
    for r in range(rows):
        for c in range(cols):
            if source_ras[r, c] > 0:
                cost_dist[r, c] = 0
                heapq.heappush(pq, (0, r, c))
    
    # Dijkstra's algorithm
    if eight_directions:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        direction_codes = [0, 1, 2, 3, 4, 5, 6, 7]  # Assigning direction codes for each move
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction_codes = [0, 1, 2, 3]  # Assigning direction codes for each move
    
    while pq:
        current_cost, r, c = heapq.heappop(pq)
        
        if visited[r, c]:
            continue
        
        visited[r, c] = True
        
        for i, (dr, dc) in enumerate(directions):
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                move_cost = cost_ras[nr, nc] if dr == 0 or dc == 0 else cost_ras[nr, nc] * np.sqrt(2)
                new_cost = current_cost + move_cost
                
                if new_cost < cost_dist[nr, nc]:
                    cost_dist[nr, nc] = new_cost
                    direction[nr, nc] = direction_codes[i]
                    heapq.heappush(pq, (new_cost, nr, nc))
    
    return cost_dist, direction

def get_mcrpath(cost_dist, direction, target_ras, eight_directions):
    target_points = np.argwhere(target_ras > 0)
    path = []

    # Directions and their inverses
    if eight_directions:
        inverse_directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        inverse_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    # Find the target point with the minimum cost
    min_cost = np.inf
    min_point = None
    for r, c in target_points:
        if cost_dist[r, c] < min_cost:
            min_cost = cost_dist[r, c]
            min_point = (r, c)
    
    # Backtrack from the target point to the source point
    if min_point:
        r, c = min_point
        while cost_dist[r, c] != 0:
            path.append((r, c))
            direction_code = direction[r, c]
            dr, dc = inverse_directions[int(direction_code)]
            r, c = r + dr, c + dc
        path.append((r, c))  # Add the source point
    
    return path[::-1]  # Reverse the path to start from the source

def geodataframes_to_graph(points_gdf, lines_gdf):
    # 创建一个空的无向图
    G = nx.Graph()

    # 添加点要素作为节点
    for idx, point in points_gdf.iterrows():
        # 获取点的经纬度
        lat, lon = point.geometry.y, point.geometry.x
        # 展开属性字典并添加 lat 和 lon
        node_attributes = {f"{k}": str(v) if isinstance(v, dict) else v for k, v in point.items() if k != 'geometry'}
        node_attributes['lat'] = lat
        node_attributes['lon'] = lon
        G.add_node(idx, **node_attributes)

    # 添加线要素作为边
    for idx, line in lines_gdf.iterrows():
        # 确保线要素的起点和终点存在于点要素中
        start_point = line.geometry.coords[0]
        end_point = line.geometry.coords[-1]
        
        # 找到点要素中的起点和终点的索引
        start_node = None
        end_node = None
        
        for point_idx, point in points_gdf.iterrows():
            if Point(start_point).equals(point.geometry):
                start_node = point_idx
            if Point(end_point).equals(point.geometry):
                end_node = point_idx
                
        # 如果找到了起点和终点的匹配点，则添加边
        if start_node is not None and end_node is not None:
            # 不保留线的几何信息，只保留其他属性
            edge_attributes = {f"{k}": str(v) if isinstance(v, dict) else v for k, v in line.items() if k != 'geometry'}
            G.add_edge(start_node, end_node, **edge_attributes)
    return G

def create_graph_from_rasters(node_raster, core_raster,connectivity_raster, resistance_raster,adjacency_df):
    unique_nodes = np.unique(node_raster[~np.isnan(node_raster) & (node_raster > 0)])

    dataset = gdal.Open(node_raster_path)
    geometry_list = []
    attribute_list = []
    point_geometry_list = []
    point_attribute_list = []
    
    # 创建图
    G = nx.Graph()
    G1 = nx.Graph()
    
    # 添加节点
    node_positions = {}
    for node in unique_nodes:
        pos = convert_to_int_tuple(np.argwhere(node_raster == node)[0])
        lon, lat = pixel_to_geo(dataset, pos[1], pos[0])
        G.add_node(node, lon=lon, lat=lat)
        G1.add_node(node, lon=lon, lat=lat)
        node_positions[node] = pos
        point_geometry_list.append((lon,lat))
        point_attribute_list.append({"nodeid": node,"sourceid": node ,"type":'S'})
    # line_positions = {label: np.argwhere(line_raster == label) for label in unique_nodes}
    
    nique_values_block = adjacency_df['Block'].unique()
    G1nodeId = len(node_positions)
    G1lineId = 0
    for node_i in tqdm(nique_values_block, total=len(nique_values_block), desc="Processing Rows"):
        Adjacent_block = adjacency_df.loc[adjacency_df['Block'] == node_i, 'Adjacent']
        source_ras = np.zeros(connectivity_raster.shape)
        source_ras[np.where(core_raster == node_i)] = 1
        for node_j in Adjacent_block:
            target_ras = np.zeros(connectivity_raster.shape)
            target_ras[np.where(core_raster == node_j)] = 1
            condition = (adjacency_df['Block'] == node_i) & (adjacency_df['Adjacent'] == node_j)
            pi_light = np.array(adjacency_df.loc[condition, ['NearestP_Block1_X', 'NearestP_Block1_Y']])[0]
            pj_light = np.array(adjacency_df.loc[condition, ['NearestP_Block2_X', 'NearestP_Block2_Y']])[0]

            source_rassub, source_origin = extract_subgrid(source_ras, pi_light, pj_light)
            target_rassub, _ = extract_subgrid(target_ras, pi_light, pj_light)
            connectivity_sub, _ = extract_subgrid(resistance_raster, pi_light, pj_light)

            cost_dist, direction = cost_distance_with_direction(source_rassub, connectivity_sub, True)
            subpath = get_mcrpath(cost_dist, direction, target_rassub, True)
            path = restore_path_to_global(source_origin, subpath)

            if path == []:
                eightConn.append(str(node_i) + "--" +str(node_j))
                continue

            updataMask = np.zeros(core_raster.shape)
            updataMask[np.where((core_raster!=0)&(core_raster!=node_i)&(core_raster!=node_j))] = 1
            maskIndex = np.array([updataMask[p[0], p[1]] for p in path])
            sourceN = np.sum(maskIndex)

            if sourceN == 0: #当廊道不经过其它斑块时保留此廊道
                pathList = [connectivity_raster[p[0], p[1]] for p in path]
                mean_weight = np.mean(pathList)
                sum_weight = np.sum(pathList)
                min_len = len(path)
                
                # 计算源点对应的地理坐标
                fromPointS = pixel_to_geo(dataset, node_positions[node_i][1], node_positions[node_i][0])
                toPointS = pixel_to_geo(dataset, node_positions[node_j][1], node_positions[node_j][0])
                # 计算activateP对应的地理坐标
                fromPoint = pixel_to_geo(dataset, path[0][1], path[0][0])
                toPoint = pixel_to_geo(dataset, path[-1][1], path[-1][0])

                actId_pi = G1nodeId + 1
                actId_pj = G1nodeId + 2
                G1nodeId = actId_pj

                lineId_L = G1lineId + 1
                lineId_Li = G1lineId + 2
                lineId_Lj = G1lineId + 3
                G1lineId = lineId_Lj

                # 保存对应源点的OD Graph和相应属性
                G.add_edge(node_i, node_j, weight=mean_weight, sum_weight=sum_weight, distance = min_len,fromx = fromPointS[0], fromy = fromPointS[1], tox = toPointS[0], toy =toPointS[1])

                # 保存踏脚石之间的廊道数据为Shp数据
                min_geopath = [pixel_to_geo(dataset, mp[1], mp[0]) for mp in path]
                line_geom = LineString(min_geopath)
                geometry_list.append(line_geom)
                attribute_list.append({"fromnode": node_i,"tonode":node_j,"fromX": fromPoint[0], "fromY": fromPoint[1], "toX": toPoint[0], "toY": toPoint[1], "weight": mean_weight, "sum_weight":sum_weight, "len": min_len,"type":'T',"lineid": lineId_L})
                G1.add_node(actId_pi, lon=fromPoint[0], lat=fromPoint[1])
                G1.add_node(actId_pj, lon=toPoint[0], lat=toPoint[1])
                G1.add_edge(actId_pi, actId_pj, weight=mean_weight, sum_weight=sum_weight, distance = min_len,fromx = fromPoint[0], fromy = fromPoint[1], tox = toPoint[0], toy =toPoint[1], lineid = lineId_L)

                # 保存踏脚石数据为Shp数据
                point_geometry_list.append(fromPoint)
                point_attribute_list.append({"nodeid": actId_pi,"sourceid": node_i,"type":'A'})
                point_geometry_list.append(toPoint)
                point_attribute_list.append({"nodeid": actId_pj,"sourceid": node_j,"type":'A'})

                # 补充源点到踏脚石的廊道数据
                source_rassub, target_rassub, source_origin = extract_centerP(source_ras, node_positions[node_i], path[0])
                connectivity_sub, _ = extract_subgrid(resistance_raster, node_positions[node_i], path[0])

                cost_dist, direction = cost_distance_with_direction(source_rassub, connectivity_sub, True)
                subpath = get_mcrpath(cost_dist, direction, target_rassub, True)
                path1 = restore_path_to_global(source_origin, subpath)
                # path = dijkstra(connectivity_weight_core_raster, node_positions[node_i], pi_light)
                # _, path = nx.single_source_dijkstra(weightgraph, source=tuple(node_positions[node_i]), target=tuple(pi_light), weight='weight')
                geopath = [pixel_to_geo(dataset, cp[1], cp[0]) for cp in path1]
                line_geom = LineString(geopath)
                geometry_list.append(line_geom)
                pathList = [connectivity_raster[p[0], p[1]] for p in path1]
                mean_weight = np.mean(pathList)
                sum_weight = np.sum(pathList)
                lenc = len(path)
                attribute_list.append({"fromnode": node_i,"tonode":node_i,"fromX": fromPointS[0], "fromY": fromPointS[1], "toX": fromPoint[0], "toY": fromPoint[1], "weight": mean_weight, "sum_weight":sum_weight, "len": lenc,"type":'S',"lineid": lineId_Li})
                G1.add_edge(node_i, actId_pi, weight=mean_weight, sum_weight=sum_weight, distance = min_len,fromx = fromPointS[0], fromy = fromPointS[1], tox = fromPoint[0], toy =fromPoint[1], lineid = lineId_Li)

                source_rassub, target_rassub, source_origin = extract_centerP(source_ras, node_positions[node_j], path[-1])
                connectivity_sub, _ = extract_subgrid(resistance_raster, node_positions[node_j], path[-1])
                cost_dist, direction = cost_distance_with_direction(source_rassub, connectivity_sub, True)
                subpath = get_mcrpath(cost_dist, direction, target_rassub, True)
                path2 = restore_path_to_global(source_origin, subpath)
                geopath = [pixel_to_geo(dataset, cp[1], cp[0]) for cp in path2]
                line_geom = LineString(geopath)
                geometry_list.append(line_geom)
                pathList = [connectivity_raster[p[0], p[1]] for p in path2]
                mean_weight = np.mean(pathList)
                sum_weight = np.sum(pathList)
                lenc = len(path)
                attribute_list.append({"fromnode": node_j,"tonode":node_j,"fromX": toPointS[0], "fromY": toPointS[1], "toX": toPoint[0], "toY": toPoint[1], "weight": mean_weight, "sum_weight":sum_weight, "len": lenc,"type":'S',"lineid": lineId_Lj})
                G1.add_edge(node_j, actId_pj, weight=mean_weight, sum_weight=sum_weight, distance = min_len,fromx = toPointS[0], fromy = toPointS[1], tox = toPoint[0], toy =toPoint[1], lineid = lineId_Lj)

    # 保存踏脚石之间的廊道数据为Shp数据
    gdf = gpd.GeoDataFrame(attribute_list, crs="EPSG:4326", geometry=geometry_list)
    gdf['wkb_geometry'] = gdf['geometry'].apply(lambda geom: geom.wkb)
    gdf_unique = gdf.loc[gdf.groupby('wkb_geometry')['lineid'].idxmin()]
    gdf_unique = gdf_unique.drop(columns='wkb_geometry')

    # 保存踏脚石数据为Shp数据
    dfpoint = pd.DataFrame(point_geometry_list, columns=['longitude', 'latitude'])
    geometrypoint = [Point(xy) for xy in zip(dfpoint['longitude'], dfpoint['latitude'])]
    gdfpoint = gpd.GeoDataFrame(point_attribute_list, crs="EPSG:4326", geometry=geometrypoint)
    gdfpoint['wkb_geometry'] = gdfpoint['geometry'].apply(lambda geom: geom.wkb)
    gdfpoint_unique = gdfpoint.loc[gdfpoint.groupby('wkb_geometry')['nodeid'].idxmin()]
    gdfpoint_unique = gdfpoint_unique.drop(columns='wkb_geometry')

    G2 = geodataframes_to_graph(gdfpoint_unique, gdf_unique)

    return G, G2 , gdf_unique, gdfpoint_unique

def save_graph_to_graphml(graph, file_path):
    nx.write_graphml(graph, file_path)


if __name__ == '__main__':

    # node_raster_path景观源点数据路径；mspa_core_path景观Core区数据；
    # line_raster_path景观边缘数据（目的是减少计算，数据量不大时可以与mspa_core_path一致）
    node_raster_path = r'InputData\mspa_core_filled_center.tif'
    mspa_core_path = r'InputData\mspa_core_filled.tif'
    node_raster,im_porj, im_geotrans = GeoImgR(node_raster_path)
    core_raster,_,_ = GeoImgR(mspa_core_path)
  
    # 加载连通性数据和阻力面数据
    connectivity_raster_path = r'InputData\cum_update.tif'
    resistance_raster_file = r'InputData\cum_weight_core0714.tif'
    connectivity_raster,_,_ = GeoImgR(connectivity_raster_path)
    resistance_raster,_,_ = GeoImgR(resistance_raster_file)

    adj_csv_path = r'OutputData/adjacency_parallel.csv'
    adjacency_df = read_adjacency_csv(adj_csv_path)

    # 输出文件路径
    graphml_file_path = r'OutputData/ecological_network0812_mcr.graphml'
    graphml_file_path1 = r'OutputData/ecological_network0812_mcr_act.graphml'
    shpline_file_path = r'OutputData/ecological_network0812_mcr.shp'
    shppoint_file_path = r'OutputData/ecological_network0812_mcr_point.shp'

    # ****************************************************************创建生态网络
    eightConn = []
    ecological_network,ecological_network_act,corrdors,point  = create_graph_from_rasters(node_raster, core_raster,connectivity_raster, resistance_raster,adjacency_df)
    print(eightConn)

    # 保存图结构到GraphML文件
    save_graph_to_graphml(ecological_network, graphml_file_path)
    save_graph_to_graphml(ecological_network_act, graphml_file_path1)
    corrdors.to_file(shpline_file_path)
    point.to_file(shppoint_file_path)

    # screen -L -Logfile getcorridor_mcr0804.log python 3_GetCorridor.py