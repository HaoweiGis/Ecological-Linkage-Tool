# Ecological Linkage Tool
 Ecological Linkage Tool

This is a tool for calculating ecological linkage between different landscape elements (i.e., habitat).

The Ecological Linkage Tool (ELT) can identify ecological corridors both within and outside irregular ecological sources, activates nodes and stepping stones, and construct an intact ecological network (Figure). The workflow is as follows: First, Euclidean allocation is applied to ecological sources. Pixels are assigned to the nearest sources areas based on Euclidean distance, and an adjacency table of sources was generated. Second, the adjacency table is updated by adding the nearest neighboring points of adjacent sources. Third, the Minimum Cumulative Resistance (MCR) model was executed. The scope of MCR execution was optimized using the neighboring points, generating ecological corridors and activation points. These were then transformed into an ecological network. ELT was developed using the open-source tools (e.g., Geospatial Data Abstraction Library, GDAL), unlike Linkage Mapper (McRae and Kavanagh, 2011) (https://linkagemapper.org/), does not rely on ArcPy and supports parallel computing.

ELT comprises three key codes: 1_AdjacentCore.py, 2_GetActivateP.py, and 3_GetCorridorAN.py. 

![image](https://github.com/HaoweiGis/Ecological-Linkage-Tool/blob/main/InputData/Framework.jpg)
The article submitted to Geography and Sustainability

ELT-Direction.py implements corridor direction computation, classifying corridors into east-west, north-south, southeast-northwest, and northeast-southwest directions based on their average azimuth angles. These calculations are performed in a projected coordinate system using the Albers equal-area conic projection, which is well-suited for mid-latitude regions.

ELT-BiologicalFlow.py enables the computation of dynamic biological flows, where potential biological flows arise from species migration between habitat patches, influenced by habitat area, biodiversity, and habitat quality (Lu et al., 2024). Drawing on island biogeography theory, large-scale species habitats can serve as "species pools" for surrounding habitats (MacArthur and Wilson, 1963).

![image](https://github.com/HaoweiGis/Ecological-Linkage-Tool/blob/main/InputData/Framework2.JPG)

##  Contact Information
If you have any query for this work, please directly contact me.
E-mail: haoweimu@smail.nju.edu.cn
WeChat: HaoweiNJU