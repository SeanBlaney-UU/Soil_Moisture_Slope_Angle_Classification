# Soil Moisture Landform Classification

A tool to allow computation statistics with regard to high resolution of soil moisture values and slope angle values. 

## Installation

Use the package manager pip to install Soil Moisture Landform Classification.

```
pip install git+ssh://git@github.com/SeanBlaney-UU/Soil_Moisture_Landform_Classification
```

## Usage

The program requires two raster file inputs and uses default filenames unless the input files are provided by command line arguments. 
The first requirement is a Digital Elevation Model provided as a GeoTIFF file complete with a coordinate reference system in a projected grid.
The second requirement is GeoTIFF containing high resolution soil moisture data complete. Should the soil moisture GeoTIFF file 
be provided with a coordinate reference system that is different to the digital elevation model, it will be reprojected on the fly to
allow the two datasets to be samples at common coordinates.

The two required raster files should be located at:

```
data_files\DTM_Rev1_Clip.tif
data_files\soil_moisture.tif
```

Deliverables that are created whilst processing are saved to:

```
\exports
```

If the export folder does not exist it will be created.

## Dependencies


