import PIL
import os, sys
import glob
import numpy as np
import h5py
from matplotlib.pyplot import imshow
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import xarray as xr
import rioxarray as rio
import pygmt
from PIL import Image

def raw_to_images():
    fig, ax = plt.subplots()
    print()
    for filename in glob.iglob('./data/raw/' + '**/*.hf5', recursive=True):
        # open raw radar data
        ax.clear()
        img = h5py.File(filename)
        original_image = np.array(img["image1"]["image_data"]).astype(float)
        cmap=np.array(img["visualisation1"]["color_palette"])
        knmimap=ListedColormap(cmap/256.0)

        original_image[original_image == 255.0] = np.nan
        original_image[original_image == 0.0] = np.nan
        masked_image = np.ma.array(original_image, mask=np.isnan(original_image))

        my_cmap = plt.cm.get_cmap('viridis')
        my_cmap.set_bad('white', 0)
        day = filename[11:26]
        path_to_image_day = './data/images/'+day
        raw_name = filename[-16:].replace('.hf5','.png')
        if not os.path.exists(path_to_image_day):
            os.makedirs(path_to_image_day)
        full_path = path_to_image_day + '/' + raw_name
        plt.imsave(full_path, masked_image, cmap=my_cmap)
        print("saved " + full_path)

def resize_images():
    for filename in glob.iglob('./data/images/' + '**/*.png', recursive=True):
        print("resizing " + filename)
        im = Image.open(filename)
        #im1 = im.crop((left, top, right, bottom))
        im1 = im.resize((70,77))
        im1.save(filename,"PNG")

raw_to_images()
resize_images()