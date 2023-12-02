import os
import math
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import cv2


try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

MODEL_PATH = './model_saved'
model = load_model(MODEL_PATH)

# larger possible dpi: 382x350
def create_dataset_from_raw(directory_path, resize_to):
    resize_width = resize_to[0]
    resize_height = resize_to[1]
    batch_names = [directory_path + name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    dataset = np.zeros(shape=(len(batch_names),36,resize_height,resize_width)) # (samples, filters, rows = height, cols = width)

    for batch_idx,batch in enumerate(batch_names):
        files = [x for x in os.listdir(batch) if x != '.DS_Store']
        files.sort()
        crn_batch = np.zeros(shape=(36, resize_height, resize_width))
        for (idx,raster) in enumerate(files):
            fn = batch + '/' + raster
            img = h5py.File(fn)
            original_image = np.array(img["image1"]["image_data"]).astype(float)
            img = Image.fromarray(original_image)
            # note that here it is (width, heigh) while in the tensor is in (rows = height, cols = width)
            img = img.resize(size=(resize_width, resize_height))
            original_image = np.array(img)
            original_image = original_image / 255.0
            crn_batch[idx] = original_image
        dataset[batch_idx] = crn_batch
        print("Importing batch:" + str(batch_idx+1))
    return dataset

def split_data_xy(data):
    x = data[:, 0 : 18, :, :]
    y = data[:, 18 : 36, :, :]
    return x, y

def RMSE(y_true,y_pred):
    return math.sqrt(np.square(np.subtract(y_true,y_pred)).mean())

def plot_both(y_true, y_pred):
    # Plot the original frames.
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(y_true[idx]), cmap="viridis")
        ax.set_title(f"Frame {idx + 18}")
        ax.axis("off")

    # Plot the predicted frames.
    for idx, ax in enumerate(axes[1]):
        predicted = y_pred[idx].reshape((344,315))
        ax.imshow(predicted, cmap="viridis")
        ax.set_title(f"Frame {idx + 18}")
        ax.axis("off")

    plt.show()

val_dataset = create_dataset_from_raw('./data/raw_validation/', resize_to=(315,344))
val_dataset = np.expand_dims(val_dataset, axis=-1)
val_x, val_y = split_data_xy(val_dataset)

# calculate RMSE between ground truth and predicted frames
results = []
for i in range(val_x.shape[0]):
    crn_datapoint = val_x[i]
    new_prediction = model.predict(np.expand_dims(crn_datapoint, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)

    rmse = RMSE(val_y[i], new_prediction)
    results.append(rmse)

print("Average RMSE: " + str(np.average(results)))
