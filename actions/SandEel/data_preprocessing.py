import numpy as np
import iris
from iris.coords import DimCoord
from iris.analysis import AreaWeighted, Linear
from iris.cube import Cube
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
# import torch

def preprocess_data(sv, datrange):
    """
    Get data ready for model
    """
    
    # Regrid data in range
    new_sample_distance = 0.18
    Sv, ping_vector, depth_vector, frequencies = regrid_data(
        sv, datrange, new_sample_distance=new_sample_distance)

    # range_vector = np.arange(0, Sv.shape[-1], new_sample_distance)
    # range_start_idx = np.argmin(np.abs(range_vector - z))
    # range_start_idx = np.min([range_start_idx, Sv.shape[-1] - 256])
    
    #Sv2[:, :, range_start_idx:range_start_idx + 256]

    # Select the frequency channels the model is trained on: [18, 38, 120, 200]
    Sv, frequencies = select_channels(Sv, frequencies)

    ########----Ingrid-----#######
    # Crop data in range to get 256 x 256 patch (HACK!)
    Sv2 = np.zeros((len(frequencies), 256, 256), dtype=np.float)
    Sv2[:] = -82 #  np.nan
    n_p = min([len(ping_vector), 256])
    n_d = min([len(depth_vector), 256])
    Sv2[:, 0:n_p, 0:n_d] = Sv[:, 0:n_p, 0:n_d]

    # TODO check this
    # Move axis to get correct shape: [frequency_channels, range, time]
    Sv2 = np.moveaxis(Sv2, -1, 1)
    #######----Ingrid-----#######
    
    return Sv2, frequencies, depth_vector, ping_vector


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


def regrid_data(sv, datrange, new_sample_distance=0.18):
    # Nearest neighghbour regridding algoritm

    # Get the frequency vector
    freq = [_sv['frequency'] for _sv in sv[0]['channels']]
    
    # Time vector
    # time = [_sv['time'] for _sv in sv]
    pingNumber = [_sv['pingNumber'] for _sv in sv]

    # transducer depth is not available, and this is an approximation
    offset = sv[0]['channels'][len(freq)-1]['sampleDistance']*sv[0][
        'channels'][len(freq)-1]['offset']
    tr_depth = datrange['minDepth'] - offset

    # horizontal_dim = pingNumber
    new_vertical_dim = np.arange(datrange['minDepth'], datrange['maxDepth'],
                                 new_sample_distance)
    
    # Initialize output (channel, ping, range)
    output = np.full((len(freq), len(pingNumber), len(new_vertical_dim)),
                     np.nan)

    # Loop over channel
    for i, data_freq in enumerate(freq):
        # Loop over ping
        for j, _sv in enumerate(sv):
            sv_sub = _sv['channels'][i]['sv']
            d0 = tr_depth + _sv['channels'][i]['sampleDistance']*_sv[
                'channels'][i]['offset']
            # _pingnumber = datrange['pingNumber'] + j
            sd = _sv['channels'][i]['sampleDistance']
            old_vertical_dim = np.arange(0, len(sv_sub))*sd + d0
            
            len(old_vertical_dim)
            len(sv_sub)

            f = interpolate.interp1d(old_vertical_dim, sv_sub,
                                     axis=0,
                                     bounds_error=False,
                                     kind='nearest',
                                     fill_value=(sv_sub[0], sv_sub[-1]))
            # use interpolation function returned by `interp1d`
            ynew = f(new_vertical_dim)
            output[i, j, :len(new_vertical_dim)] = ynew

    return output, pingNumber, new_vertical_dim, freq

'''
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
'''
