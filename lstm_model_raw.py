import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import h5py
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization
from PIL import Image
import glob
from IPython import display
import matplotlib.animation as animation

try:
    # Disable all GPUS due to issue with the BatchNormalization() layer
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

def create_dataset_from_raw(directory_path):
    batch_names = [directory_path + name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    dataset = np.zeros(shape=(len(batch_names),36,170,156))
    # for fn in file_list:
    #     print(fn)
    for batch_idx,batch in enumerate(batch_names):
        files = [x for x in os.listdir(batch) if x != '.DS_Store']
        crn_batch = np.zeros(shape=(36, 170, 156))
        for (idx,raster) in enumerate(files):
            fn = batch + '/' + raster
            img = h5py.File(fn)
            original_image = np.array(img["image1"]["image_data"]).astype(float)
            img = Image.fromarray(original_image)
            img = img.resize(size=(156, 170))
            original_image = np.array(img)
            # set background pixels to 255.0
            original_image[original_image == 255.0] = 255.0
            original_image[original_image == 0.0] = 255.0
            crn_batch[idx] = original_image / 255.0
        dataset[batch_idx] = crn_batch
    return dataset

train_dataset = create_dataset_from_raw('./data/raw/')
validation_dataset = create_dataset_from_raw('./data/raw_validation/')
train_dataset = np.expand_dims(train_dataset, axis=-1)
validation_dataset = np.expand_dims(validation_dataset, axis=-1)

def create_shifted_frames(data):
    x = data[:, 0 : 17, :, :]
    y = data[:, 18 : 35, :, :]
    return x, y

x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(validation_dataset)

def create_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=128, kernel_size=(7, 7),
                    input_shape=(None,*train_dataset.shape[2:]),
                    padding='same', activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=128, kernel_size=(5, 5),
                    padding='same',activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3),
                    padding='same',activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=128, kernel_size=(1, 1),
                    padding='same',activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same', data_format='channels_last'))
    return model

model = create_model()

model.compile(loss='binary_crossentropy', optimizer='adadelta')
print(model.summary())
# Define modifiable training hyperparameters.
epochs = 25
batch_size = 5

#Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=1,
)
model.save('./model_saved')

# Select a random example from the validation dataset.
example = train_dataset[np.random.choice(range(len(train_dataset)), size=1)[0]]

# Pick the first/last ten frames from the example.
frames = example[:18, ...]
original_frames = example[18:, ...]
#reconstructed_model = keras.models.load_model("model_saved")

# Predict a new set of 10 frames.
for _ in range(18):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 18, figsize=(20, 8))
my_cmap = plt.cm.get_cmap('viridis')
my_cmap.set_bad('white', 0)
originals = []
predicted = []
# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap=my_cmap)
    ax.set_title(f"Original Frame {idx}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[18:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap=my_cmap)
    ax.set_title(f"Predicted Frame {idx}")
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
    ax.set_title("Original radar data")

def updatePredicted(frame):
    # clear the axis each frame
    ax.clear()
    ax.set_xlabel("KM")
    ax.set_ylabel("KM")
    ax.grid()
    
    ax.imshow(np.squeeze(frame), cmap=my_cmap)
    # replot things
    ax.set_title("Predicted radar data")

def save_animations():
    fig, ax = plt.subplots()
    animation_originals = animation.FuncAnimation(fig, updateOriginals, frames=original_frames, interval=200)
    animation_originals.save('originals.gif', writer='imagemagick', fps=2)

    animation_predicted = animation.FuncAnimation(fig, updatePredicted, frames=new_frames, interval=200)
    animation_predicted.save('predicted.gif', writer='imagemagick', fps=2)

save_animations()