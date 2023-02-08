# --- Begin generated header ---
import sys
import matplotlib.pyplot as plt
import json
import os
import requests
import numpy as np
import pickle
from scipy.ndimage import binary_opening, binary_closing, label, generate_binary_structure

from actions.SandEel.data_preprocessing import preprocess_data
from actions.SandEel.model import load_pretrained_model, get_predictions

baseUrl = 'http://127.0.0.1:' + os.environ.get('LSSS_SERVER_PORT', '8000')
input = json.loads(os.environ.get('LSSS_INPUT', '{}'))

# Path to pre-trained model
#checkpoint_path = 'C:\\Users\\utseth\\Documents\\Projects\\COGMAR\\Data\\model\\paper_v2_heave_2.pt'
checkpoint_path = '/mnt/c/DATAscratch/crimac-scratch/modelweights/paper_v2_heave_2.pt'


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
centre = get('/lsss/module/PelagicEchogramModule/current-echogram-point')


# If not, just get the centre pixel for the test data:
if not len(centre) == 0:
    pingNumber = centre['pingNumber']
    #z = centre['z']
    # centre = {'time': '2021-04-27T14:48:39.039Z', 'pingNumber': 18629,
    #          'vesselDistance': 699.663, 'z': 33.77759}
else:
    pingNumber = int(18629-256/2)
    z = 34

# There is a school here:
pingNumber = int(18450-256/2)
z = 40

    
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

npings = 256
datrange = {'pingCount': npings, 'sv': True, 'pingNumber': pingNumber}
sv = get('/lsss/data/pings', params=datrange)

# Get the frequency vector
freq = [_sv['frequency'] for _sv in sv[0]['channels']]

# Time vector
time = [_sv['time'] for _sv in sv]
pingNumber = [_sv['pingNumber'] for _sv in sv]

# sample disance per channel
sampledistance = [_sv['sampleDistance'] for _sv in sv[0]['channels']]
transduceroffset = [_sv['offset'] for _sv in sv[0]['channels']]


data = []
depth = []
for i, _freq in enumerate(freq):
    dum = np.array([_sv['channels'][i]['sv'] for _sv in sv])
    data.append(dum)

    # Length of range vector
    depth.append(np.arange(dum.shape[1])*sampledistance[i])

# Regrid the data, ensure correct dimensions
data, freq, _depth = preprocess_data(data, freq, sampledistance, z)

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
    bbox = [min(xs), min(ys), max(xs), max(ys)]


# TODO POST SCHOOLS TO LSSS

# Plotting
fig, axs = plt.subplots(nrows=5, figsize=(6, 10))
for i in range(len(freq)):
    axs[i].imshow(preds_binary, aspect='auto')
    axs[i].set_title(freq[i])
plt.tight_layout()
plt.show()

# interp depth and ys

pn = np.array(pingNumber)[ys]
school = []
for i, _pn in enumerate(pn):
    school.append({'pingNumber': int(_pn), 'z': _depth[xs[i]]})

# This is how you post a school to LSSS:
#school_test = [{'pingNumber': 18400, 'z': 30},
#               {'pingNumber': 18400, 'z': 50},
#               {'pingNumber': 18500, 'z': 50},
#               {'pingNumber': 18500, 'z': 30}]

school2 = post('/lsss/module/PelagicEchogramModule/school-boundary',
              json = school)

# Set interpretation
school['id']


