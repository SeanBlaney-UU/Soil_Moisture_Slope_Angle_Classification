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
    if (verbose_messages): print(dataset.profile)

    # assign the DEM plot extent to a variable, and;
    # if verbose flag is set, print the dataset extent
    dem_plot_ext = plotting_extent(dataset)
    if (verbose_messages): print(plotting_extent(dataset))

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

    show_hist(dtm_pre_arr, bins=20, lw=0.0, stacked=False, alpha=0.3, histtype='stepfilled',
              title="Elevation Distribution", ax=ax_hist)

    gdal.DEMProcessing(os.path.join(export_folder_location, "slope.tif"),
                       srcDS=dem_location,
                       processing='slope')

    with rio.open(os.path.join(export_folder_location, "slope.tif")) as slope_dataset:
        # slope = slope_dataset.read(1)

        slope_data = slope_dataset.read(1, masked=True)
        slope_data[slope_data < 0] = np.nan

        show(slope_data, cmap='Reds',
             transform=dataset.transform,
             ax=ax_slope,
             title='Slope')

    # with rio.open(sm_location) as sm_dataset:
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
        # slope = slope_dataset.read(1)

        soil_moisture = sm_dataset.read(1, masked=True)
        soil_moisture[soil_moisture < 0] = np.nan

        # plot the soil moisture
        show(soil_moisture, cmap='RdYlGn_r',
             transform=sm_dataset.transform,
             ax=ax_soil_moisture,
             title='Soil Moisture')

        # plot the reprojected soil moisture tif on top of the slope plot to confirm correct transform
        show(soil_moisture, cmap='RdYlGn_r',
             transform=sm_dataset.transform,
             ax=ax_slope,
             title='Soil Moisture')

        # Calculate some statistics
        # Start with the soil moisture data

        print("Soil moisture dataset profile: ", sm_dataset.profile)

        xcols = range(0, sm_dataset.width + 1)
        ycols = range(0, sm_dataset.height + 1)

        xs, ys = xy(sm_dataset.transform, xcols, ycols)

        # make a tuple from the coordinates as the rasterio sample() method requires pairs of coordinates
        sm_pixel_coordinates = tuple(zip(xs, ys))
        print("sm_pixel_coordinates count: {}".format(len(sm_pixel_coordinates)))

        sm_samples = sample_gen(dataset=sm_dataset, xy=sm_pixel_coordinates, indexes=1, masked=True)

        # print("sm_samples count: {}".format(sum(1 for _ in sm_samples)))

        sm_samples_with_coord = list(zip(sm_pixel_coordinates, sm_samples))

        print("sm_samples_with_coord count: {}".format(len(sm_samples_with_coord)))

        # TODO - consider list comprehension
        # sm_samples_with_coord[0][0][0] = Easting
        # sm_samples_with_coord[0][0][1] = Northing
        # sm_samples_with_coord[0][1][0] = Soil Moisture Value
        valid_samples = list()

        for x in sm_samples_with_coord:
            if not np.ma.is_masked(x[1][0]):
                valid_samples.append([x[0][0],
                                      x[0][1],
                                      x[1][0]])

        print("valid_samples count: {}".format(len(valid_samples)))

        categorised_sm_values = {"10 - 15%": 0,
                                 "15 - 20%": 0,
                                 "20 - 25%": 0,
                                 "25 - 30%": 0,
                                 "30 - 35%": 0,
                                 "35 - 40%": 0,
                                 "40 - 45%": 0,
                                 "45 - 50%": 0,
                                 "50 - 55%": 0,
                                 "55 - 60%": 0}

        for x in valid_samples:
            if 10 < x[2] <= 15: categorised_sm_values["10 - 15%"] = categorised_sm_values["10 - 15%"] + 1
            if 15 < x[2] <= 20: categorised_sm_values["15 - 20%"] = categorised_sm_values["15 - 20%"] + 1
            if 20 < x[2] <= 25: categorised_sm_values["20 - 25%"] = categorised_sm_values["20 - 25%"] + 1
            if 25 < x[2] <= 30: categorised_sm_values["25 - 30%"] = categorised_sm_values["25 - 30%"] + 1
            if 30 < x[2] <= 35: categorised_sm_values["30 - 35%"] = categorised_sm_values["30 - 35%"] + 1
            if 35 < x[2] <= 40: categorised_sm_values["35 - 40%"] = categorised_sm_values["35 - 40%"] + 1
            if 40 < x[2] <= 45: categorised_sm_values["40 - 45%"] = categorised_sm_values["40 - 45%"] + 1
            if 45 < x[2] <= 50: categorised_sm_values["45 - 50%"] = categorised_sm_values["45 - 50%"] + 1
            if 50 < x[2] <= 55: categorised_sm_values["50 - 55%"] = categorised_sm_values["50 - 55%"] + 1
            if 55 < x[2] <= 60: categorised_sm_values["55 - 60%"] = categorised_sm_values["55 - 60%"] + 1

        print("Categorised soil moisture values: ", categorised_sm_values)

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




#############################################
# Begin Slope Statistics ####################
#############################################

src_file = sm_location
dst_file = os.path.join(export_folder_location, "slope.tif")
dst_crs = input_crs

# Open the slope tif back up
with rio.open(dst_file) as slope_dataset:
    # slope = slope_dataset.read(1)

    slope = slope_dataset.read(1, masked=True)
    slope[slope < 0] = np.nan

    # Calculate some statistics
    # Now with the slope data

    print("Slope dataset profile: ", sm_dataset.profile)

    xcols = range(0, slope_dataset.width + 1)
    ycols = range(0, slope_dataset.height + 1)

    xs, ys = xy(slope_dataset.transform, xcols, ycols)

    # make a tuple from the coordinates as the rasterio sample() method requires pairs of coordinates
    slope_pixel_coordinates = tuple(zip(xs, ys))
    print("slope_pixel_coordinates count: {}".format(len(slope_pixel_coordinates)))

    # sample the raster values at the coordinates provided
    slope_samples = sample_gen(dataset=slope_dataset, xy=slope_pixel_coordinates, indexes=1, masked=True)

    # print("sm_samples count: {}".format(sum(1 for _ in sm_samples)))

    slope_samples_with_coord = list(zip(slope_pixel_coordinates, slope_samples))

    print("slope_samples_with_coord count: {}".format(len(slope_samples_with_coord)))

    # TODO - consider list comprehension
    # sm_samples_with_coord[0][0][0] = Easting
    # sm_samples_with_coord[0][0][1] = Northing
    # sm_samples_with_coord[0][1][0] = Soil Moisture Value
    valid_samples = list()

    for x in slope_samples_with_coord:
        if not np.ma.is_masked(x[1][0]):
            valid_samples.append([x[0][0],
                                  x[0][1],
                                  x[1][0]])

    print("valid_samples count: {}".format(len(valid_samples)))

#############################################
# End Slope Statistics ######################
#############################################



# Finally - save the plots!


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
