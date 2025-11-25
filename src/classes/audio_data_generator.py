from tensorflow.keras.utils import Sequence
import numpy as np
import os

class AudioDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.file_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = []
        y_batch = []
        
        for idx in batch_indices:
            try:
                feat_path = self.file_paths[idx]
                
                if os.path.exists(feat_path):
                    feat = np.load(feat_path)  # shape: (1, num_features, time_steps)
                    # Add channel dimension: (num_features, time_steps, 1)
                    feat = feat[..., np.newaxis]  # add channel dim
                    X_batch.append(feat)
                    y_batch.append(self.labels[idx])
                else:
                    print(f"Feature not found: {feat_path}")
                    
            except Exception as e:
                print(f"Error loading {self.file_paths[idx]}: {e}")
                continue
        
        return np.array(X_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)