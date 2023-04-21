# --- Begin generated header ---
import sys
import matplotlib.pyplot as plt
import json
import os
import requests
import numpy as np
import pickle
from scipy.ndimage import binary_opening, binary_closing, label, generate_binary_structure
try:
    from actions.SandEel.data_preprocessing import preprocess_data
    from actions.SandEel.model import load_pretrained_model, get_predictions
except:
    from data_preprocessing import preprocess_data
    from model import load_pretrained_model, get_predictions

baseUrl = 'http://127.0.0.1:' + os.environ.get('LSSS_SERVER_PORT', '8000')
input = json.loads(os.environ.get('LSSS_INPUT', '{}'))

# Path to pre-trained model
checkpoint_path_all = [
    'C:\\Users\\utseth\\Documents\\Projects\\COGMAR\\Data\\model\\paper_v2_heave_2.pt',
    '/mnt/c/DATAscratch/crimac-scratch/modelweights/paper_v2_heave_2.pt',
    './paper_v2_heave_2.pt', './actions/SandEel/paper_v2_heave_2.pt']
checkpoint_path = [_path for _path in checkpoint_path_all if os.path.isfile(_path)][0]


def get(path, params=None):
    url = baseUrl + path
    response = requests.get(url, params=params)
    if response.status_code == 200:
        if response.headers['Content-Type'] == 'application/json':
            return response.json()
        return response.text
    raise ValueError(url + ' returned status code ' + str(response.status_code) + ': ' + response.text)


def post(path, params=None, json=None, data=None):
    url = baseUrl + path
    response = requests.post(url, params=params, json=json, data=data)
    if response.status_code == 200:
        if response.headers['Content-Type'] == 'application/json':
            return response.json()
        return response.text
    if response.status_code == 204:
        return None
    raise ValueError(url + ' returned status code ' + str(response.status_code) + ': ' + response.text)


#
# Get the location where we would like to run the classifier
#

# Get the data from LSSS, this works if you click on the echogram first
# If connection refused error, ensure LSSS scripting server is turned on
# Application configuration -> LSSS server -> Server active. Access level (lower left corner) = Administrator mode
zoom = get('/lsss/module/PelagicEchogramModule/zoom')

# /lsss/module/{EchogramModuleId}/current-echogram-point
# If not, just get the centre pixel for the test data:
if len(zoom) == 2:
    npings = zoom[1]['pingNumber']-zoom[0]['pingNumber']
    datrange = {'pingCount': npings, 'sv': True,
                'pingNumber': zoom[0]['pingNumber'],
                'minDepth': zoom[0]['z'], 'maxDepth': zoom[1]['z']}
else:
    raise ValueError('You need to open the echogram and zoom to the region of interest.')

# There is a school here:
# pingNumber = int(18450-256/2)
# z = 28
    
#
# Get sv data
#
    
# Horizontal size og patch
#    all: boolean. All these options (except pulseCompressed and complex)
#    angles: boolean. Alongship and athwartship angles [degree]
#    sv: boolean. Sv [dB]
#    tsc: boolean. Beam compensated TS [dB]
#    tsu: boolean. Uncompensated TS [dB]
#    pulseCompressed: boolean. Complex sample data after pulse compression, if available
#    complex: boolean. Complex sample data before pulse compression, if available
#    nmea: boolean. NMEA sentences
#
# Optionally limit vertical extent:
#    minDepth: float. Minimum depth [m]
#    maxDepth: float. Maximum depth [m]

sv = get('/lsss/data/pings', params=datrange)

# TODO: The regridding seems to select the wrong data. Needs attention.
data, freq, _depth, pingNumber = preprocess_data(sv, datrange)

# Plot test
if True:
    fig, axs = plt.subplots(nrows=len(freq), figsize=(6, 10))
    for i in range(len(freq)):
        axs[i].imshow(data[i], aspect='auto')
        axs[i].set_title(freq[i])
    plt.tight_layout()
    plt.show(block=False)

# Post data input to LSSS for testing
if False:
    # This post a square school box to vizualize the data that is used by the algortihm
    # You cannot post a school into another, so this is not compatible with posting
    # schools from the algorithm.
    school_test = [{'pingNumber': int(min(pingNumber)), 'z': min(_depth)},
                   {'pingNumber': int(min(pingNumber)), 'z': max(_depth)},
                   {'pingNumber': int(max(pingNumber)), 'z': max(_depth)},
                   {'pingNumber': int(max(pingNumber)), 'z': min(_depth)}]
    
    school3 = post('/lsss/module/PelagicEchogramModule/school-boundary',
                   json = school_test)

# Load pretrained model and get predictions
model = load_pretrained_model(checkpoint_path)
preds = get_predictions(model, data)

# Assign all predictions above threshold as "sandeel" (consider argmax instead ...)
prediction_threshold = 0.5
preds_binary = (preds > prediction_threshold).astype(np.uint8)

# Morphological opening and closing to remove small predictions (this should be refined)
kernel = np.ones((5, 5), np.uint8)
preds_binary = binary_opening(preds_binary, kernel)
preds_binary = binary_closing(preds_binary, kernel)

# Plot test
if True:
    fig, axs = plt.subplots(nrows=len(freq), figsize=(6, 10))
    for i in range(len(freq)):
        axs[i].imshow(preds_binary, aspect='auto')
        axs[i].set_title(freq[i])
    plt.tight_layout()
    plt.show(block=True)

# Get the connected components (schools)
s = generate_binary_structure(2, 2)
labelled_preds, num_schools = label(preds_binary, structure=s)

# Get bounding box for each school
for i in range(1, num_schools+1):
    # Get bounding box
    idxs = np.argwhere(labelled_preds == i)
    print(idxs.shape)
    xs = idxs[:, 0]
    ys = idxs[:, 1]
    # bbox = [min(xs), min(ys), max(xs), max(ys)]
    # Get the unique pings
    pi = np.unique(ys)
    ma = []
    mi = []
    for _ys in np.unique(ys):
        ma.append(max(xs[ys == _ys]))
        mi.append(min(xs[ys == _ys]))
    # Combine max an min values
    ysi = np.append(pi, pi[::-1])  # Ping
    xsi = np.append(ma, mi[::-1])  # Depth
    school = []
    for i, _ysi in enumerate(ysi):
        school.append({'pingNumber': int(_ysi+pingNumber[0]),
                       'z': _depth[xsi[i]]})
    # Post school
    posted_school = post('/lsss/module/PelagicEchogramModule/school-boundary',
                         json = school)


