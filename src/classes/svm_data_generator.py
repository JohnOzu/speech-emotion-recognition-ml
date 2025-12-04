import os
import numpy as np
from tqdm import tqdm

class SVMDataGenerator:

    # Generator that loads audio features on the fly for SVM training.

    def __init__(self, file_paths, labels, batch_size=32, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(file_paths)
        self.indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # Return number of batches per epoch
        return int(np.ceil(self.n_samples / self.batch_size))

    def __iter__(self):
        # Make generator iterable
        return self

    def __next__(self):
        # Get next batch
        if self._batch_idx >= len(self):
            raise StopIteration

        batch = self.next_batch()
        return batch

    def next_batch(self):
        # Load and return the next batch of features
        start_idx = self._batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]

        # Load features for this batch
        X_batch = []
        y_batch = []

        for idx in batch_indices:
            # Load feature file
            features = np.load(self.file_paths[idx])

            # Flatten features
            features_flatten = features.flatten()

            X_batch.append(features_flatten)
            y_batch.append(self.labels[idx])

        self._batch_idx += 1

        return np.array(X_batch), np.array(y_batch)

    def reset(self):
        # Reset generator to start
        self._batch_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_all_batches(self):
        # Generator function to yield all batches

        self.reset()

        for _ in range(len(self)):
            yield self.next_batch()

    def load_all_data(self):
        """Load all data into memory (needed for traditional SVM)"""
        print("Loading all features into memory for SVM training...")
        X_all = []
        
        for i in tqdm(range(len(self.file_paths)), desc="Loading features"):
            features = np.load(self.file_paths[i])
            X_all.append(features.flatten())
        
        return np.array(X_all), self.labels