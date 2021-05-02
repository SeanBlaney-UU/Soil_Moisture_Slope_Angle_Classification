# Soil Moisture Slope Angle Classification

A tool to allow computation statistics with regard to high resolution soil moisture data and slope angle values.

## Features

This tool was created to allow the production of statistics to provide a greater understanding of the distribution of 
soil moisture value in relation to slope angle. Given a grid projected digital elevation model, the following plots 
and csv files are created:

* plots.png - An overview of individual plots created during execution.
* dtm_subplot.png - A colourised plot of the digital elevation model.
* histogram_subplot.png - A histogram plot of the distribution of elevation heights on a per cell basis.
* slope_subplot.png - A colourised plot of slope angle, created using GDAL from the digital elevation model and overplotted with the soil moisture data to allow a visual confirmation of the correct transformation of the soil moisture data.
* soil_moisture_subplot.png - A colourised plot of the soil moisture values on a per cell basis.
* soil_moisture_reprojected.png - A GeoTIFF of the reprojected soil moisture data.
* slope.tif - A GeoTIFF of the slope angle data generated during execution.
* categorised_count.csv - A csv containing the count of soil moisture values in intervals of 5% between 10% and 60% gravimetric water content.
* statistics.csv - A CSV containing the easting and northing of the sample points, soil moisture value, slope angle value and soil moisture interval.

## Installation

Use the package manager pip to install Soil Moisture Slope Angle Classification.

```
pip install git+ssh://git@github.com/SeanBlaney-UU/Soil_Moisture_Slope_Angle_Classification
```

Once cloned to a local directory, cd into the project directory and create the following two directories within the root folder of the project:

```
data_files\
exports\
```

Input rasters should be placed in `data_files\`.

## Usage

The program requires two raster file inputs and uses default filenames unless the input files are provided by command line arguments. 
The first requirement is a Digital Elevation Model provided as a GeoTIFF file complete with a coordinate reference system in a projected grid.
The second requirement is GeoTIFF containing high resolution soil moisture data complete. Should the soil moisture GeoTIFF file 
be provided with a coordinate reference system that is different to the digital elevation model, it will be reprojected on the fly to
allow the two datasets to be sampled at common coordinates.

The two required raster files should be located at:

```
data_files\DTM_Rev1_Clip.tif
data_files\SM_Resample_Bilinear.tif
```

Deliverables that are created whilst processing are saved to:

```
exports\
```

If the export folder does not exist it will be created.

## Dependencies

[Matplotlib](https://matplotlib.org/)

[pandas](https://pandas.pydata.org/)

[Rasterio](https://rasterio.readthedocs.io/en/latest/)

[NumPy](https://numpy.org/)

[OSGeo GDAL](https://github.com/OSGeo/gdal)


## License

*GNU General Public License v3.0*

