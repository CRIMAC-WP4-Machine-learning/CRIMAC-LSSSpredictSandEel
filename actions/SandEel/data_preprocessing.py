import numpy as np
import iris
from iris.coords import DimCoord
from iris.analysis import AreaWeighted, Linear
from iris.cube import Cube
# import torch

def preprocess_data(Sv, frequencies, sampledistance, z):
    """
    Get data ready for model
    """
    # Regrid data in range
    new_sample_distance = 0.18
    Sv = regrid_data(Sv, sampledistance, new_sample_distance=new_sample_distance)

    # Crop data in range to get 256 x 256 patch
    range_vector = np.arange(0, Sv.shape[-1], new_sample_distance)
    range_start_idx = np.argmin(np.abs(range_vector - z))
    range_start_idx = np.min([range_start_idx, Sv.shape[-1] - 256])
    Sv = Sv[:, :, range_start_idx:range_start_idx + 256]

    # Select the frequency channels the model is trained on: [18, 38, 120, 200]
    Sv, frequencies = select_channels(Sv, frequencies)

    # TODO check this
    # Move axis to get correct shape: [frequency_channels, range, time]
    Sv = np.moveaxis(Sv, -1, 1)
    return Sv, frequencies


def select_channels(sv, frequencies):
    """
    Select frequency channels model has been trained on
    """
    # For some reason the frequencies retrieved does not completely match frequencies displayed in LSSS
    frequencies = np.array(frequencies)
    freq_idxs = []
    for f in [18000, 38000, 120000, 200000]:
        freq_idxs.append(np.argmin(np.abs(frequencies - f)))
    freq_idxs = np.array(freq_idxs)

    return sv[freq_idxs, :, :], frequencies[freq_idxs]


def db_with_limits(data, limit_low=-75, limit_high=0):
    # remove nan / inf
    data[np.invert(np.isfinite(data))] = 0

    # Decibel transform, and set limits
    data = db(data)
    data[data > limit_high] = limit_high
    data[data < limit_low] = limit_low
    return data

def db(data, eps=1e-10):
    """ Decibel (log) transform """
    return 10 * np.log10(data + eps)

def regrid_data(data, sampledistance, new_sample_distance=0.18):
    """
    Placeholder linear regridding algoritm - should probably use something else
    """
    horizontal_dim = np.arange(0, data[0].shape[0], 1)
    horizontal_dim_coord = DimCoord(horizontal_dim, standard_name='projection_x_coordinate', units='s')

    data_regridder = Linear()
    max_depths = np.array([data_freq.shape[1] * sampledistance[i] for i, data_freq in enumerate(data)])
    max_length = max(np.ceil(max_depths/new_sample_distance))

    output = np.full((len(data), 256, np.ceil(max_length).astype(int)), np.nan)
    for i, data_freq in enumerate(data):
        old_resolution = sampledistance[i]
        max_depth = max_depths[i]

        old_vertical_dim = np.arange(0, max_depth, old_resolution)
        old_vertical_dim_coord = DimCoord(old_vertical_dim, standard_name='projection_y_coordinate', units='meter')
        new_vertical_dim = np.arange(0, max_depth, new_sample_distance)
        new_vertical_dim_coord = DimCoord(new_vertical_dim, standard_name='projection_y_coordinate', units='meter')

        old_dims = [(horizontal_dim_coord, 0), (old_vertical_dim_coord, 1)]
        new_dims = [(horizontal_dim_coord, 0), (new_vertical_dim_coord, 1)]

        data_regridded = regrid(data_freq, old_dims, new_dims, regridder=data_regridder)

        # Need a better regridding to ensure data is on the same grid with same length. For now, just select 256 patch
        output[i, :, :len(new_vertical_dim)] = data_regridded


    return output

def regrid(data, old_dims, new_dims, regridder=None):
    """
    :param data: data to be regridded, 2D or 3D
    :param old_dims: old data dimensions (list of Iris DimCoord)
    :param new_dims: new data dimensions (list of Iris DimCoord)
    :param regridder: iris regrid algorithm
    :return:
    """
    orig_cube = Cube(data, dim_coords_and_dims=old_dims)
    grid_cube = Cube(np.zeros([coord[0].shape[0] for coord in new_dims]), dim_coords_and_dims=new_dims)

    try:
        orig_cube.coord('projection_y_coordinate').guess_bounds()
        orig_cube.coord('projection_x_coordinate').guess_bounds()
        grid_cube.coord('projection_y_coordinate').guess_bounds()
        grid_cube.coord('projection_x_coordinate').guess_bounds()
    except ValueError:
        pass

    if regridder is None:
        regridder = Linear()
    regrid = orig_cube.regrid(grid_cube, regridder)
    return regrid.data

