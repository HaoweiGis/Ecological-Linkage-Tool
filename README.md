# Ecological Linkage Tool
 Ecological Linkage Tool

This is a tool for calculating ecological linkage between different landscape elements (i.e., habitat).

The Ecological Linkage Tool (ELT) can identify ecological corridors both within and outside irregular ecological sources, activates nodes and stepping stones, and construct an intact ecological network (Figure). The workflow is as follows: First, Euclidean allocation is applied to ecological sources. Pixels are assigned to the nearest sources areas based on Euclidean distance, and an adjacency table of sources was generated. Second, the adjacency table is updated by adding the nearest neighboring points of adjacent sources. Third, the Minimum Cumulative Resistance (MCR) model was executed. The scope of MCR execution was optimized using the neighboring points, generating ecological corridors and activation points. These were then transformed into an ecological network. ELT was developed using the open-source tools (e.g., Geospatial Data Abstraction Library, GDAL), unlike Linkage Mapper (McRae and Kavanagh, 2011) (https://linkagemapper.org/), does not rely on ArcPy and supports parallel computing.

## Download

Ecological Linkage Tool (ELT): https://doi.org/10.5281/zenodo.19602332

ELT comprises three key codes: 1_AdjacentCore.py, 2_GetActivateP.py, and 3_GetCorridorAN.py. 

![image](https://github.com/HaoweiGis/Ecological-Linkage-Tool/blob/main/InputData/Framework.jpg)

##  OtherTools
ELT-Direction.py implements corridor direction computation, classifying corridors into east-west, north-south, southeast-northwest, and northeast-southwest directions based on their average azimuth angles. These calculations are performed in a projected coordinate system using the Albers equal-area conic projection, which is well-suited for mid-latitude regions.

ELT-BiologicalFlow.py enables the computation of dynamic biological flows, where potential biological flows arise from species migration between habitat patches, influenced by habitat area, biodiversity, and habitat quality (Lu et al., 2024). Drawing on island biogeography theory, large-scale species habitats can serve as "species pools" for surrounding habitats (MacArthur and Wilson, 1963).

![image](https://github.com/HaoweiGis/Ecological-Linkage-Tool/blob/main/InputData/Framework2.JPG)

##  Recommended Citations

If you use **ELT** in your research, please cite:

> Mu H, Guo S, Zhang X, et al. An enhanced ecological network for spatial planning considering spatial conflicts and structural resilience[J]. Geography and Sustainability, 2026: 100420. (https://doi.org/10.1016/j.geosus.2026.100420)

If you use **ELT-Direction** (corridor direction analysis) and **ELT-BiologicalFlow** (biological flow simulation), please cite:

> Mu H, Guo S, Pan K, et al. Revealing the dynamic biological flow between eastern and western China from the perspective of ecological network[J]. Environmental Impact Assessment Review, 2026, 116: 108138. (https://doi.org/10.1016/j.eiar.2025.108138)

If you use **Connectivity.tif** (omnidirectional connectivity data) in this repository, please cite:

> Mu H, Guo S, Zhang X, et al. Moving in the landscape: Omnidirectional connectivity dynamics in China from 1985 to 2020[J]. Environmental Impact Assessment Review, 2025, 110: 107721. (https://doi.org/10.1016/j.eiar.2024.107721)

##  Contact Information
If you have any query for this work, please directly contact me.

E-mail: haoweimu@smail.nju.edu.cn

WeChat: HaoweiNJU
