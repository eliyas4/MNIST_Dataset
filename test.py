import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Set your image directory
img_dir = 'data'

# Get all file names
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

# Initialize lists
images = []
filenames = []

# Loop through files and preprocess
for file in img_files:
    img_path = os.path.join(img_dir, file)
    img = Image.open(img_path).convert('L')        # convert to grayscale
    img = img.resize((28, 28))                      # resize (MNIST‐like)
    img_array = np.array(img) / 255.0               # normalize to [0,1]
    images.append(img_array)
    filenames.append(file)                          # keep track of filenames

# Stack into a single numpy array of shape (n, 28, 28)
images = np.stack(images)  # shape = (num_images, 28, 28)

# Load the CSV of labels; assume it has the column ['character']
# (If your CSV’s “filename” column matches exactly the .jpg names, you could
# join by filename; here the original code zipped in order, so we’ll just
# assume labels_df['character'] is aligned with images in the same order.)
labels_df = pd.read_csv('chinese_mnist.csv')
print("labels_df:\n", labels_df.head())

# If you need to convert characters to integer indices:
unique_chars = sorted(labels_df['character'].unique())
char_to_idx = {c: i for i, c in enumerate(unique_chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Visual check
plt.imshow(images[1228], cmap='gray')
plt.title(f"Example Image (label = {labels_df.loc[1228,'character']})")
plt.axis('off')
plt.show()

# Create a list of (image, label) tuples
data = list(zip(images, labels_df['character']))

# Shuffle in place (set seed if you want reproducibility)
random.seed(42)
random.shuffle(data)

# Total number of examples
N = len(data)

# Compute split indices for 80/10/10
n_train = int(0.80 * N)
n_val   = int(0.10 * N)   # we’ll take the next 10% for validation
# (the remaining N - n_train - n_val will be for test)

# Now slice
train_data = data[:n_train]
val_data   = data[n_train : n_train + n_val]
test_data  = data[n_train + n_val : ]

print(f"Total examples: {N}")
print(f" → Train: {len(train_data)}")
print(f" → Val:   {len(val_data)}")
print(f" → Test:  {len(test_data)}")

# Example access
img_train0, label_train0 = train_data[0]
img_val0,   label_val0   = val_data[0]
img_test0,  label_test0  = test_data[0]

print("Train[0] shape & label:", img_train0.shape, label_train0)
print("Val[0]   shape & label:", img_val0.shape,   label_val0)
print("Test[0]  shape & label:", img_test0.shape,  label_test0)

# If you need them as separate numpy arrays:
X_train = np.stack([pair[0] for pair in train_data])
y_train = np.array([pair[1] for pair in train_data])

X_val   = np.stack([pair[0] for pair in val_data])
y_val   = np.array([pair[1] for pair in val_data])

X_test  = np.stack([pair[0] for pair in test_data])
y_test  = np.array([pair[1] for pair in test_data])

print("Shapes: ")
print(" X_train:", X_train.shape, " y_train:", y_train.shape)
print(" X_val:  ", X_val.shape,   " y_val:  ", y_val.shape)
print(" X_test: ", X_test.shape,  " y_test: ", y_test.shape)
