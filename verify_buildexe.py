import sys
import os
import time

# 将 BuildEXE 目录添加到 sys.path 以便导入模块
buildexe_path = r"D:\2_HaoweiPapers\1_SOCIAndEco\Ecological-Linkage-Tool-main\BuildEXE"
if buildexe_path not in sys.path:
    sys.path.insert(0, buildexe_path)

# 设置 PROJ 数据路径，避免 GDAL/PROJ 报错
os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\envs\geobase\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\ProgramData\Anaconda3\envs\geobase\Library\share\gdal'

from step0 import run_preprocess
from step1 import run_adjacent
from step2 import run_activate
from step3 import run_corridor

# ── 配置路径 ───────────────────────────────────
input_dir = r"D:\2_HaoweiPapers\1_SOCIAndEco\Ecological-Linkage-Tool-main\InputData"
eco_source = os.path.join(input_dir, "EcologicalSource.tif")
connectivity = os.path.join(input_dir, "Connectivity.tif")
resistance = os.path.join(input_dir, "ResistanceSurface.tif")

output_dir = r"D:\2_HaoweiPapers\1_SOCIAndEco\Ecological-Linkage-Tool-main\OutputData_BuildEXE"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ── 运行流程 ───────────────────────────────────
def main():
    start_time = time.time()
    log("开始 BuildEXE 版本验证运行...")
    
    try:
        # 步骤 0: 预处理
        run_preprocess(eco_source, output_dir, min_area_km2=1.0, log_callback=log)
        
        # 步骤 1: 邻接关系计算
        filled_tif = os.path.join(output_dir, "mspa_core_filled.tif")
        run_adjacent(filled_tif, output_dir, log_callback=log)
        
        # 步骤 2: 激活点计算
        adj_csv = os.path.join(output_dir, "adjacency.csv")
        run_activate(filled_tif, adj_csv, output_dir, log_callback=log)
        
        # 步骤 3: 廊道构建
        run_corridor(None, filled_tif, connectivity, resistance, adj_csv, output_dir, log_callback=log)
        
        end_time = time.time()
        log(f"验证运行全部完成！总耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        log(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
