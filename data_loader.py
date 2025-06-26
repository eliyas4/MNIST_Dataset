import os
import re
from PIL import Image
import numpy as np
import pandas as pd

class MNISTLoader:
    def __init__(self, image_directory='data', labels_file='chinese_mnist.csv'):
        self.image_directory = image_directory
        self.labels_file = labels_file


    def load_data(self):
        # Gather all image filenames in the directory
        image_files = [f for f in os.listdir(self.image_directory) if os.path.isfile(os.path.join(self.image_directory, f))]


        # Load the CSV into a DataFrame
        labels_df = pd.read_csv(self.labels_file)
        required_cols = {'suite_id', 'sample_id', 'code', 'character'}
        if not required_cols.issubset(labels_df.columns):
            missing = required_cols - set(labels_df.columns)
            raise ValueError(f"Labels CSV is missing these columns: {missing}")
        

        labels_df.set_index(['suite_id', 'sample_id', 'code'], inplace=True)
        # Now you can do: labels_df.loc[(s_id, samp_id, c), 'character']


        # Build char_to_idx / idx_to_char from all unique characters
        unique_chars = sorted(labels_df['character'].unique())
        print("unique_chars", unique_chars)
        char_to_idx = {c: i for i, c in enumerate(unique_chars)}
        print("char_to_idx", char_to_idx)
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        print("idx_to_char", idx_to_char)


        data: list[tuple[np.ndarray, int]] = []


        # This gives you a multiIndex which allows you to search for a file based on a particular combination of suite_id, sample_id, and code. 
        pattern = re.compile(r'^input_(\d+)_(\d+)_(\d+)\.(?:png|jpe?g|bmp)$', re.IGNORECASE)


        for fname in image_files:
            match = pattern.fullmatch(fname)
            if not match:
                raise ValueError(
                    f"Filename '{fname}' does not match 'input_<suite>_<sample>_<code>.<ext>'."
                )

            suite_id   = int(match.group(1))
            sample_id  = int(match.group(2))
            code       = int(match.group(3))

            # Look up the character in the CSV via the MultiIndex
            try:
                char = labels_df.loc[(suite_id, sample_id, code), 'character']
            except KeyError:
                raise KeyError(
                    f"No row found in CSV for (suite_id={suite_id}, sample_id={sample_id}, code={code})."
                )

            label_idx = char_to_idx[char]

            # 7. Load & preprocess the image: grayscale → resize → normalize
            img_path = os.path.join(self.image_directory, fname)
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            arr = np.array(img, dtype=np.float32) / 255.0

            # Add the one hot label method so that each datapoints lable is a one hot lable. 
            one_hot_label = np.zeros((len(char_to_idx), 1))
            one_hot_label[label_idx] = 1.0
            # print("label_idx:", label_idx, "one_hot_label shape:", one_hot_label.shape)
            # print("one_hot_label sample:\n", one_hot_label.T)
            data.append((arr.reshape((28 * 28, 1)), one_hot_label))


        # Split the full data into 80/10/10 for training, validation, and testing.
        n = len(data)
        n_train = int(0.80 * n)
        n_val   = int(0.10 * n) 

        train_data = data[:n_train]
        val_data   = data[n_train : n_train + n_val]
        test_data  = data[n_train + n_val : ]

        print("this is what your data looks like:")
        print(f"{'training data':<17} - {len(train_data)}")
        print(f"{'validation data':<17} - {len(val_data)}")
        print(f"{'testing data':<17} - {len(test_data)}")

        return (train_data, val_data, test_data)
