# ELT 生态廊道分析工具 — 打包方案

## 一、现有代码问题清单

| 文件 | 问题 | 严重程度 |
|------|------|---------|
| `1_AdjacentCore.py` L107-108 | 悬空 `return` 语句（无函数体，死代码） | 高 |
| `3_GetCorridorAN.py` L318 | `create_graph_from_rasters` 内引用全局变量 `node_raster_path` | 高 |
| `1_AdjacentCore.py` `euclidean_allocation` | 双重 Python 循环，O(n²)，大栅格极慢 | 高 |
| 所有文件 | `base_dir` 硬编码绝对路径 | 高 |
| `1_AdjacentCore.py` | 并行版 `compute_adjacency_parallel` 与实际调用的单线程版并存 | 中 |
| `2_GetActivateP.py` | 同上，并行/单线程并存 | 中 |
| `0_PreprocessTool.py` | `from tkinter import Button` 导入但未使用 | 低 |

---

## 二、依赖替换策略

### 目标：减少 C 扩展依赖，简化打包

| 库 | 当前用途 | 替换方案 | 说明 |
|----|---------|---------|------|
| `cv2` (OpenCV) | `cv2.dilate` 形态学膨胀 | `scipy.ndimage.binary_dilation` | 完全等价，scipy 打包更成熟 |
| `osgeo.gdal` / `osgeo.osr` | 栅格读写、坐标转换 | `rasterio` + `pyproj` | rasterio 是 GDAL 的 Pythonic 封装，有官方 Windows wheel |
| `geopandas` | 矢量数据读写（仅输出 SHP） | **保留**，但统一通过 rasterio/pyproj 处理投影 | fiona>=1.9 有 PyPI 预编译 wheel，无需系统 GDAL |
| `multiprocessing.Pool` | 并行计算 | **移除**，统一单线程 | 消除 PyInstaller 的 `freeze_support` 兼容问题 |

**关键结论：** GDAL 仍是底层依赖（rasterio/geopandas 都用），但通过 PyPI wheel 安装，
无需用户手动配置系统 GDAL，打包时用 `--collect-all` 即可。

---

## 三、新代码架构

```
BuildEXE/
├── main.py          # tkinter GUI 入口 + 流程编排
├── step0.py         # 预处理（栅格→矢量→过滤→栅格化）
├── step1.py         # 邻接关系计算（欧氏分配 + 邻接矩阵）
├── step2.py         # 激活点计算（边界提取 + 最近点对）
├── step3.py         # 廊道构建（Dijkstra MCR + 网络图）
├── utils.py         # 共享工具（栅格 I/O、坐标转换）
├── requirements.txt # 依赖清单
├── build.spec       # PyInstaller 打包配置（后续补充）
└── PLAN.md          # 本文件
```

### 模块间数据流

```
InputData/
  EcologicalSource.tif ──→ [step0] ──→ OutputData/
  Connectivity.tif      ──→ [step3]      mspa_core_filled.tif
  ResistanceSurface.tif ──→ [step3]      mspa_core_center.tif
                                          mspa_core.shp
                                          mspa_core_center.shp
                                      ──→ [step1] ──→
                                          adjacency.csv（邻接关系）
                                          mspa_core_filled_eucall.tif
                                      ──→ [step2] ──→
                                          adjacency.csv（追加最近点列）
                                          mspa_core_activate.tif
                                      ──→ [step3] ──→
                                          ecological_network.graphml
                                          ecological_network_act.graphml
                                          ecological_network.shp
                                          ecological_network_point.shp
```

---

## 四、关键优化点

### 4.1 `euclidean_allocation` 向量化
```python
# 原代码 O(n²) — 双重循环逐像素赋值
for i in range(allocation.shape[0]):
    for j in range(allocation.shape[1]):
        allocation[i, j] = raster_array[indices[0, i, j], indices[1, i, j]]

# 新代码 O(n) — numpy 高级索引，速度提升数百倍
allocation = raster_array[indices[0], indices[1]]
```

### 4.2 `cv2.dilate` → `scipy.ndimage.binary_dilation`
```python
# 原代码
dilated = cv2.dilate(block1.astype(np.uint8), struct, iterations=1)

# 新代码（等价，无 cv2 依赖）
from scipy.ndimage import binary_dilation
dilated = binary_dilation(block1, structure=struct)
```

### 4.3 修复全局变量引用（step3）
```python
# 原代码：函数内直接用全局变量
dataset = gdal.Open(node_raster_path)  # ← 隐式依赖外部变量

# 新代码：从函数参数 transform/crs 传入，用 pyproj 做坐标转换
lon, lat = pixel_to_geo(row, col, transform, crs)
```

### 4.4 统一路径管理
- 所有步骤函数接受 `input_path` 和 `output_dir` 参数
- GUI 负责路径拼接，代码内不再有硬编码路径

---

## 五、打包策略

### 依赖安装（目标环境）
```
pip install numpy scipy rasterio geopandas shapely networkx pandas tqdm pyproj
```
> 不再需要 `opencv-python`（cv2）和直接的 `GDAL` 包

### PyInstaller 打包命令
```bash
pyinstaller --name ELT_Tool \
  --onedir \
  --windowed \
  --collect-all rasterio \
  --collect-all fiona \
  --collect-all pyproj \
  --collect-all shapely \
  --collect-all geopandas \
  --hidden-import pandas \
  --hidden-import networkx \
  --hidden-import scipy.ndimage \
  --hidden-import scipy.spatial \
  main.py
```

> 注意：`--onedir`（文件夹模式）比 `--onefile` 更适合含 GDAL 数据的地理空间工具，
> 启动速度快，GDAL/PROJ 数据文件路径不会因解压临时目录而失效。

### 运行时 GDAL/PROJ 路径（需在 main.py 头部添加）
```python
import sys, os
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后设置 GDAL/PROJ 数据路径
    bundle = sys._MEIPASS
    os.environ.setdefault('GDAL_DATA', os.path.join(bundle, 'rasterio', 'gdal_data'))
    os.environ.setdefault('PROJ_DATA', os.path.join(bundle, 'pyproj', 'proj_data'))
```

### 替代方案：conda-pack（更稳健但体积更大）
```bash
conda create -n elt_env python=3.10
conda activate elt_env
conda install -c conda-forge rasterio geopandas networkx tqdm
conda install pyinstaller
conda pack -n elt_env -o elt_env.tar.gz
```

---

## 六、测试清单

- [ ] step0：单独运行，检查输出 SHP 和 TIF 文件是否正常
- [ ] step1：检查邻接 CSV 行数，验证孤立斑块警告
- [ ] step2：检查邻接 CSV 新增列（NearestP_*、Distance）
- [ ] step3：检查 SHP 廊道数量和 GraphML 节点/边数
- [ ] GUI：文件选择、日志输出、各步骤按钮
- [ ] 打包后：在无 Python 环境的 Windows 机器上运行完整流程
