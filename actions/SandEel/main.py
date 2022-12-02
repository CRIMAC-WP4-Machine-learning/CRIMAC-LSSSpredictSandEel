# --- Begin generated header ---
import sys
import matplotlib.pyplot as plt
import json
import os
import requests
import numpy as np

baseUrl = 'http://127.0.0.1:' + os.environ.get('LSSS_SERVER_PORT', '8000')
input = json.loads(os.environ.get('LSSS_INPUT', '{}'))


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
centre = get('/lsss/module/PelagicEchogramModule/current-echogram-point')

# If not, just get the centre pixel for the test data:
if not len(centre) == 0:
    pingNumber = centre['pingNumber']
    z = centre['z']
    # centre = {'time': '2021-04-27T14:48:39.039Z', 'pingNumber': 18629,
    #          'vesselDistance': 699.663, 'z': 33.77759}
else:
    pingNumber = int(18629-256/2)
    z = 34

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

# sample disance per channel
sampledistance = [_sv['sampleDistance'] for _sv in sv[0]['channels']]
transduceroffset = [_sv['offset'] for _sv in sv[0]['channels']]


dat = []
for i, _freq in enumerate(freq):
    print(i)
    dum = np.array([_sv['channels'][i]['sv'] for _sv in sv])
    # Length of range vector
    depth = np.arange(dum.shape[1])*sampledistance[i] + 
len(depth)
    
    print(dum.shape)
    dat = dat.append([dum])

plt.figure()
plt.imagesc(dat[0])
plt.show()

# This is how you post a school to LSSS:
school = [{'time': '2021-04-27T14:47:00Z', 'z': 10},
          {'time': '2021-04-27T14:47:00Z', 'z': 20},
          {'time': '2021-04-27T14:48:00Z', 'z': 20},
          {'time': '2021-04-27T14:48:00Z', 'z': 10}]

school = post('/lsss/module/PelagicEchogramModule/school-boundary', json = school)

# Set interpretation
school['id']




