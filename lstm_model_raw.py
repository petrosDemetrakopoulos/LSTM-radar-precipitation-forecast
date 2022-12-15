import glob
import os

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from IPython import display
from keras.layers import BatchNormalization, ConvLSTM2D
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv3D
from keras.models import Sequential
from PIL import Image
from tensorflow import keras
import sklearn.model_selection as sk

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
# larger possible dpi: 382x350
def create_dataset_from_raw(directory_path):
    batch_names = [directory_path + name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    dataset = np.zeros(shape=(len(batch_names),36,77,70)) # (samples, filters, rows = height, cols = width)

    for batch_idx,batch in enumerate(batch_names):
        files = [x for x in os.listdir(batch) if x != '.DS_Store']
        files.sort()
        crn_batch = np.zeros(shape=(36, 77, 70)) 
        for (idx,raster) in enumerate(files):
            fn = batch + '/' + raster
            img = h5py.File(fn)
            original_image = np.array(img["image1"]["image_data"]).astype(float)
            img = Image.fromarray(original_image)
            img = img.resize(size=(70, 77)) # note that here it is (width, heigh) while in the tensor is in (rows = height, cols = width)
            original_image = np.array(img)
            # set background pixels to 0.0
            original_image[original_image == 255.0] = 255.0
            original_image[original_image == 0.0] = 255.0
            original_image = original_image / 255.0
            crn_batch[idx] = original_image
        dataset[batch_idx] = crn_batch

    return dataset

#X_train, X_test, y_train, y_test = sk.train_test_split(features,labels,test_size=0.33, random_state = 42)

def create_shifted_frames(data):
    x = data[:, 0 : 18, :, :]
    y = data[:, 18 : 36, :, :]
    return x, y

dataset = create_dataset_from_raw('./data/raw/')
dataset = np.expand_dims(dataset, axis=-1)
dataset_x, dataset_y = create_shifted_frames(dataset)
X_train, X_val, y_train, y_val = sk.train_test_split(dataset_x,dataset_y,test_size=0.2, random_state = 42)

def create_model():
    model = Sequential()

    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                    input_shape=(None,*dataset.shape[2:]),
                    padding='same',activation=LeakyReLU(alpha=0.01), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                    padding='same',activation=LeakyReLU(alpha=0.01), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1),
                    padding='same',activation=LeakyReLU(alpha=0.01), return_sequences=True))
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same', data_format='channels_last'))
    return model

model = create_model()

model.compile(loss='binary_crossentropy', optimizer='adadelta')
print(model.summary())
# Define modifiable training hyperparameters.
epochs = 25
batch_size = 1

#Fit the model to the training data.
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=1,
)
model.save('./model_saved')

# Select a random example from the validation dataset.
example = train_dataset[np.random.choice(range(len(validation_dataset)), size=1)[0]]

# Pick the first/last 18 frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]
# reconstructed_model = keras.models.load_model("model_saved")

# Predict a new set of 10 frames.
for _ in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 10, figsize=(20, 8))
my_cmap = plt.cm.get_cmap('viridis')
originals = []
predicted = []
# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap=my_cmap)
    ax.set_title(f"Ground Truth {idx}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap=my_cmap)
    ax.set_title(f"Pred. Frame {idx}")
    ax.axis("off")

# Display the figure.
plt.show()

def updateOriginals(frame):
    # clear the axis each frame
    ax.clear()
    ax.set_xlabel("KM")
    ax.set_ylabel("KM")
    ax.grid()
    
    ax.imshow(np.squeeze(frame), cmap=my_cmap)
    # replot things
    ax.set_title("Original data")
    
    return ax.plot()

fig, ax = plt.subplots()
def updatePredicted(frame):
    # clear the axis each frame
    ax.clear()
    ax.set_xlabel("KM")
    ax.set_ylabel("KM")
    ax.grid()
    
    ax.imshow(np.squeeze(frame), cmap=my_cmap)
    # replot things
    ax.set_title("Predicted data")
    
    return ax.plot()

def save_animation_original():
    animation_originals = animation.FuncAnimation(fig, updateOriginals, frames=original_frames, interval=200)
    animation_originals.save('originals.gif', writer='imagemagick', fps=5)

def save_animation_predicted():
    animation_originals = animation.FuncAnimation(fig, updatePredicted, frames=new_frames, interval=200)
    animation_originals.save('predicted.gif', writer='imagemagick', fps=5)

save_animation_predicted()
save_animation_original()