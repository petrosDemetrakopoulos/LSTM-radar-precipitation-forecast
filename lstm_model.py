import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization
import PIL.Image

from IPython import display

batch_size = 36
#img_height = 219
#img_width = 200
img_height = 77
img_width = 70
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


data_dir = './data/images/train'
val_dir = './data/images/validation'
train_files_list = sorted([x for x in os.listdir(data_dir) if x != '.DS_Store'])
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0,
  labels=None,
  shuffle=False,
  seed=123,
  batch_size=36,
  image_size=(img_height, img_width))

val_ds = tf.keras.utils.image_dataset_from_directory(
  val_dir,
  labels=None,
  validation_split=0,
  shuffle=False,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=36)

# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    x = data[:, 0 : 34, :, :]
    y = data[:, 1 : 35, :, :]
    return x, y

for el in train_ds:
    print(el.shape)
print("------")
for el in val_ds:
    print(el.shape)
train_np = np.stack(list(train_ds))
val_np = np.stack(list(val_ds))
train_np[train_np == 0] = 255
val_np[val_np == 0] = 255
train_np = train_np / 255.0
val_np = val_np / 255.0
# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_np)
x_val, y_val = create_shifted_frames(val_np)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

# Construct a figure on which we will visualize the images.
fig, axes = plt.subplots(4, 5, figsize=(10, 8))

# Plot each of the sequential images for one random data example.
# data_choice = np.random.choice(range(len(train_np)), size=1)[0]
# for idx, ax in enumerate(axes.flat):
#     print(train_np[data_choice][idx])
#     ax.imshow(np.squeeze(train_np[data_choice][idx]), cmap="gray")
#     ax.set_title(f"Frame {idx + 1}")
#     ax.axis("off")

# # Print information and display the figure.
# print(f"Displaying frames for example {data_choice}.")
# plt.show()


model = Sequential()

model.add(ConvLSTM2D(filters=70, kernel_size=(3, 3),
                   input_shape=(None,*train_np.shape[2:]),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=70, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=70, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=70, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(Conv3D(filters=3, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
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
example = train_np[np.random.choice(range(len(train_ds)), size=1)[0]]

# Pick the first/last ten frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]
#reconstructed_model = keras.models.load_model("model_saved")

# Predict a new set of 10 frames.
for _ in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
my_cmap = plt.cm.get_cmap('viridis')
my_cmap.set_bad('white', 0)
# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]))
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]))
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()