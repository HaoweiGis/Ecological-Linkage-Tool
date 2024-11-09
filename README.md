# Ecological Linkage Tool
 Ecological Linkage Tool

This is a tool for calculating ecological linkage between different landscape elements (i.e., habitat).

The Ecological Linkage Tool (ELT) follows three main steps (Figure). First, Euclidean allocation is applied to ecological sources. Pixels are assigned to the nearest sources areas based on Euclidean distance, and an adjacency table of sources was generated. Second, the adjacency table is updated by adding the nearest neighboring points of adjacent sources. Third, the Minimum Cumulative Resistance (MCR) model was executed. The scope of MCR execution was optimized using the neighboring points, generating ecological corridors and activation points. These were then transformed into an ecological network. ELT was developed using the open-source tools (e.g., Geospatial Data Abstraction Library, GDAL), unlike Linkage Mapper (McRae and Kavanagh, 2011) (https://linkagemapper.org/), does not rely on ArcPy and supports parallel computing. 

![image](https://github.com/HaoweiGis/Ecological-Linkage-Tool/blob/main/InputData/Framework.jpg)