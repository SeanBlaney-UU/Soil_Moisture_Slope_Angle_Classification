import os
import timeit

import numpy as np

import rasterio as rio
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
from rasterio.plot import show_hist
from rasterio.transform import xy
from rasterio.sample import sample_gen
from shapely.ops import cascaded_union
from shapely.geometry.polygon import Polygon
from cartopy.feature import ShapelyFeature
import matplotlib.patches as mpatches

from rasterio.plot import show

import earthpy.plot as ep
import earthpy.spatial as es

from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

from osgeo import gdal

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from datetime import datetime
from operator import itemgetter

import geopandas

# check if I can refine this import
import math

'''
User Flags - could make these kwargs?
'''
verbose_messages = True
ignore_negative_dtm_elevations = True

'''
End User Flags
'''


'''
File Directories
'''
dem_location = os.path.join("data_files",
                            "DTM_Rev1_Clip.tif")

sm_location = os.path.join("data_files",
                            "SM_Resample_Bilinear.tif")

export_folder_location = os.path.join("exports")
'''
End File Directories
'''





def order_of_magnitude(max_value, min_value):
    '''
    Returns the magnitude of difference between min_value and max_value supplied
    :param max_value: maximum value of grid dataset
    :param min_value: minimum value of grid dataset
    :return: Order of magnitude difference between values supplied.
    '''
    mag_max_value = math.floor(math.log(max_value, 10))
    mag_min_value = math.floor(math.log(min_value, 10))
    return (mag_max_value - mag_min_value)


# Record the time to provide execution time later
startTime = datetime.now()




with rio.open(dem_location) as dataset:

    input_crs = dataset.crs

    # if verbose flag is set, print the dataset filename
    if (verbose_messages): print("Loading: " + dataset.name)

    # if verbose flag is set, print the dataset profile (including projection info)
    if (verbose_messages) : print(dataset.profile)

    # assign the DEM plot extent to a variable, and;
    # if verbose flag is set, print the dataset extent
    dem_plot_ext = plotting_extent(dataset)
    if (verbose_messages) : print(plotting_extent(dataset))

    # for completeness, assign extents as per EGM722
    xmin, ymin, xmax, ymax = dataset.bounds
    if (verbose_messages):
        print(xmin, ymin, xmax, ymax)
        if ((dem_plot_ext[0] == xmin) and (dem_plot_ext[1] == ymin) and
            (dem_plot_ext[2] == xmax) and (dem_plot_ext[3] == ymax)):
            print("EGM722 extents matches documentation method.")
        else:
            print("EGM722 extents does not match documentation method.")

        print(xmin)
        print(dem_plot_ext[0])


    # check EGM722 method matches my method

    dtm_pre_arr = dataset.read(1, masked=True)
    dtm_pre_arr[dtm_pre_arr < 0] = np.nan

    print("DTM Min Value: {}".format(dtm_pre_arr.min()))
    print("DTM Max Value: {}".format(dtm_pre_arr.max()))

    magnitude_of_values = order_of_magnitude(dtm_pre_arr.max(), dtm_pre_arr.min())
    if (magnitude_of_values > 2):
        print("Order of magnitude difference: {}".format(order_of_magnitude(dtm_pre_arr.max(), dtm_pre_arr.min())))
    else:
        print("Elevation magnitude of change acceptable.")
        print("Difference of the magnitude of values: {}".format(magnitude_of_values))


    fig, ((ax_dtm, ax_hist), (ax_slope, ax_soil_moisture)) = plt.subplots(2, 2, figsize=(14, 8))

    show(dtm_pre_arr, with_bounds=True,
         cmap='terrain',
         transform=dataset.transform,
         ax=ax_dtm,
         title='Digital Elevation Model')

    show_hist(dtm_pre_arr, bins=20, lw=0.0, stacked=False, alpha=0.3, histtype='stepfilled', title="Elevation Distribution", ax=ax_hist)

    gdal.DEMProcessing(os.path.join(export_folder_location, "slope.tif"),
                       srcDS=dem_location,
                       processing='slope')



    with rio.open(os.path.join(export_folder_location, "slope.tif")) as slope_dataset:
        #slope = slope_dataset.read(1)

        slope_data = slope_dataset.read(1, masked=True)
        slope_data[slope_data < 0] = np.nan

        show(slope_data, cmap='Reds',
             transform=dataset.transform,
             ax=ax_slope,
             title='Slope')

    #with rio.open(sm_location) as sm_dataset:
    #    #slope = slope_dataset.read(1)

    #    soil_moisture = sm_dataset.read(1, masked=True)
    #    soil_moisture[soil_moisture < 0] = np.nan
#
#        show(soil_moisture, cmap='RdYlGn',
#             transform=dataset.transform,
#             ax=ax_soil_moisture,
#             title='Soil Moisture')



### Now open the soil moisture dataset


src_file = sm_location
dst_file = os.path.join(export_folder_location, "soil_moisture_reprojected.tif")
dst_crs = input_crs

with rasterio.open(src_file) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(dst_file, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)


    # Open the reprojected soil moisture tif back up
    with rio.open(dst_file) as sm_dataset:
        #slope = slope_dataset.read(1)

        soil_moisture = sm_dataset.read(1, masked=True)
        soil_moisture[soil_moisture < 0] = np.nan


        # plot the soil moisture
        show(soil_moisture, cmap='RdYlGn_r',
             transform = sm_dataset.transform,
             ax = ax_soil_moisture,
             title = 'Soil Moisture')

        # plot the reprojected soil moisture tif on top of the slope plot to confirm correct transform
        show(soil_moisture, cmap='RdYlGn_r',
             transform= sm_dataset.transform,
             ax=ax_slope,
             title = 'Soil Moisture')


# Calculate some statistics
# Start with the soil moisture data

        print("Soil moisture dataset profile: ", sm_dataset.profile)

        xcols = range(0, sm_dataset.width + 1)
        ycols = range(0, sm_dataset.height + 1)

        xs, ys = xy(sm_dataset.transform, xcols, ycols)

        # make a tuple from the coordinates as the rasterio sample() method requires pairs of coordinates
        sm_pixel_coordinates = tuple(zip(xs, ys))

        sm_samples = sample_gen(dataset=sm_dataset,xy=sm_pixel_coordinates, indexes=1, masked=True)

        for i in range(1000):
            print(next(sm_samples))

        sm_samples_with_coord = list(zip(sm_pixel_coordinates, sm_samples))

        # TODO - consider list comprehension
        # sm_samples_with_coord[0][0][0] = Easting
        # sm_samples_with_coord[0][0][1] = Northing
        # sm_samples_with_coord[0][1][0] = Soil Moisture Value
        valid_samples = list()
        for x in sm_samples_with_coord:
            if not np.ma.is_masked(sm_samples_with_coord[0][1][0]):
                valid_samples.append([sm_samples_with_coord[0][0][0],
                                     sm_samples_with_coord[0][0][1],
                                     sm_samples_with_coord[0][1][0]])

        for x in sm_samples_with_coord:
            if not np.ma.is_masked(x[1][0]):
                valid_samples.append([x[0][0],
                                     x[0][1],
                                     x[1][0]])

        print(valid_samples)

        # Print the minimum and maximum soil moisture value in the valid samples
        np.set_printoptions(suppress=True)
        print("Minimum soil moisture value: {}".format(np.amin(valid_samples, axis=0)[2]))
        print("Minimum soil moisture value: {}".format(np.amax(valid_samples, axis=0)[2]))



        # TODO
        # use sample() to get slope values and soil moisture values
        # convert soil moisture values to int?
        # convert soil moisture values to categories (5%)
        # refactor repetitive code into respective methods
        # clean up imports
        # calculate soil moisture statistics on slope values normalised for area
        # clean up plots, add legend etc


## Finally - save the plots!


fig.savefig(os.path.join(export_folder_location, "plots.png"))

# ((ax_dtm, ax_hist), (ax_slope, ax_soil_moisture))
extent = ax_dtm.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(os.path.join(export_folder_location, 'dtm_subplot.png'),
            bbox_inches=extent.expanded(1.25, 1.25))
extent = ax_hist.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(os.path.join(export_folder_location, 'histogram_subplot.png'),
            bbox_inches=extent.expanded(1.25, 1.25))
extent = ax_slope.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(os.path.join(export_folder_location, 'slope_subplot.png'),
            bbox_inches=extent.expanded(1.25, 1.25))
extent = ax_soil_moisture.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(os.path.join(export_folder_location, 'soil_moisture_subplot.png'),
            bbox_inches=extent.expanded(1.25, 1.25))

# calculate the script execution time to assist with refactoring later.
print("Script execution time: {}".format(datetime.now() - startTime))