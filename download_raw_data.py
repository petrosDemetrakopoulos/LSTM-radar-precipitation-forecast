
from pandas.plotting import register_matplotlib_converters
import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np
from matplotlib.animation import FuncAnimation
from datetime import timedelta
import h5py
from matplotlib.pyplot import imshow
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import urllib.parse
import urllib.request
import json
import shutil
import os
import pygmt


# KNMI operational test key from https://developer.dataplatform.knmi.nl/get-started#make-api-calls
key = os.getenv('KNMI_API_KEY')
def getRadarData(key, tstamp,dirloc):
    url = 'https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_reflectivity_composites/versions/2.0/files/RAD_NL25_PCP_NA_'+tstamp+'.h5/url'
    headers = {'Authorization': key}

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
       meta = response.read()
    
    realurl=json.loads(meta)["temporaryDownloadUrl"]
    req = urllib.request.Request(realurl)
    fname=tstamp+".hf5"
    print(fname)
    isExist = os.path.exists(dirloc)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dirloc)
   
    with urllib.request.urlopen(req) as response:
        with open(dirloc+fname, 'wb') as location:
            shutil.copyfileobj(response, location)

def get_files_for_specific_timestamps(tstamps_list, dirloc):
    files = []
    for tstamp in tstamps_list:
        files.append(dirloc+tstamp+".hf5")
        getRadarData(key, tstamp, dirloc)
    return files

def get_data_of_n_previous_hours(hours):
    now=datetime.datetime.utcnow()
    now = now - datetime.timedelta(hours=hours, minutes=5)
    now -= datetime.timedelta(minutes=now.minute%5)

    now.strftime("%Y%m%d%H%M")
    files = []
    start=now
    for n in range(0,hours*12): # data avaialble every 5 minutes, so 12 times per hour
        tstamp=start.strftime("%Y%m%d%H%M")
        files.append(tstamp+".hf5")
        getRadarData(key, tstamp)
        start += datetime.timedelta(minutes=5)
    return files

def morning_filenames_for_day(year, month, day):
    prefix = year+month+day
    return     [prefix+'0805',prefix+'0810',prefix+'0815',prefix+'0820',prefix+'0825',prefix+'0830',
                prefix+'0835',prefix+'0840',prefix+'0845',prefix+'0850',prefix+'0855',prefix+'0900',
                prefix+'0905',prefix+'0910',prefix+'0915',prefix+'0920',prefix+'0925',prefix+'0930',
                prefix+'0935',prefix+'0940',prefix+'0945',prefix+'0950',prefix+'0955',prefix+'1000',
                prefix+'1005',prefix+'1010',prefix+'1015',prefix+'1020',prefix+'1025',prefix+'1030',
                prefix+'1035',prefix+'1040',prefix+'1045',prefix+'1050',prefix+'1055',prefix+'1100']

def afternoon_filenames_for_day(year, month, day):
    prefix = year+month+day
    return     [prefix+'2005',prefix+'2010',prefix+'2015',prefix+'2020',prefix+'2025',prefix+'2030',
                prefix+'2035',prefix+'2040',prefix+'2045',prefix+'2050',prefix+'2055',prefix+'2100',
                prefix+'2105',prefix+'2110',prefix+'2115',prefix+'2120',prefix+'2125',prefix+'2130',
                prefix+'2135',prefix+'2140',prefix+'2145',prefix+'2150',prefix+'2155',prefix+'2200',
                prefix+'2205',prefix+'2210',prefix+'2215',prefix+'2220',prefix+'2225',prefix+'2230',
                prefix+'2235',prefix+'2240',prefix+'2245',prefix+'2250',prefix+'2255',prefix+'2300']

#files = get_data_of_n_preevious_hours(3)
year = '2020'
month = '03'
day = '30'
tstamps_list = afternoon_filenames_for_day(year,month,day)
dirloc = './data/raw/' + year +'-'+month + '-' + day + '-2005/'
files = get_files_for_specific_timestamps(tstamps_list, dirloc)
