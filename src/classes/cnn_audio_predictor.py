import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pandas as pd
import sys

sys.path.append("..")

import utils
from utils import extract_mel_spectrogram
from utils import pad_features
from utils import normalize_features


class CNNAudioPredictor:
    """
    A class for predicting audio files using a trained Keras model.
    Uses TensorFlow Dataset API for memory-efficient processing.
    """
    
    def __init__(self, model_path, class_names, target_shape=(128, 128), sample_rate=22050):
        """
        Initialize the AudioPredictor.
        
        Parameters:
        - model_path: Path to the saved Keras model (.h5 or SavedModel format)
        - class_names: List of class names in order (e.g., ['dog_bark', 'cat_meow'])
        - target_shape: Target shape for mel spectrogram (height, width)
        - sample_rate: Audio sample rate for loading
        """
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
        self.target_shape = target_shape
        self.sample_rate = sample_rate
        
        print(f"Model loaded successfully!")
        print(f"Classes: {class_names}")
        print(f"Model input shape: {self.model.input_shape}")
    
    def _preprocess_audio(self, file_path):
        """
        Preprocess a single audio file.
        
        Parameters:
        - file_path: Path to audio file
        
        Returns:
        - Preprocessed mel spectrogram as numpy array
        """
        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Extract mel spectrogram using your custom function
        mel_spec = extract_mel_spectrogram(audio, sr)  # Returns (120, time_frames)

        if mel_spec.shape[0] != self.target_shape[0]:
            if mel_spec.shape[0] > self.target_shape[0]:
                # Crop if too tall
                mel_spec = mel_spec[:self.target_shape[0], :]
            else:
                # Pad if too short
                pad_height = self.target_shape[0] - mel_spec.shape[0]
                mel_spec = np.pad(mel_spec, ((0, pad_height), (0, 0)), mode='constant')
        
        # Pad to fixed length using your custom function
        mel_spec = pad_features(mel_spec, max_len=self.target_shape[1])  # (120, 150)
        
        # Normalize using your custom function
        mel_spec = normalize_features(mel_spec)
        
        # Add channel dimension
        mel_spec = mel_spec[..., np.newaxis]  # (120, 150, 1)
        
        return mel_spec.astype(np.float32)
    
    def create_dataset(self, file_paths, batch_size=32):
        """
        Create a TensorFlow Dataset for efficient data loading.
        
        Parameters:
        - file_paths: List of audio file paths
        - batch_size: Batch size for processing
        
        Returns:
        - tf.data.Dataset object
        """
        def load_and_preprocess(file_path):
            def _load_audio(path):
                path_str = path.numpy().decode('utf-8')
                return self._preprocess_audio(path_str)
            
            features = tf.py_function(_load_audio, [file_path], tf.float32)
            features.set_shape([self.target_shape[0], self.target_shape[1], 1])
            return features
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(
            load_and_preprocess, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def predict(self, file_paths, batch_size=32):
        """
        Predict on a list of audio files.
        
        Parameters:
        - file_paths: List of audio file paths
        - batch_size: Batch size for processing
        
        Returns:
        - List of dictionaries containing prediction results
        """
        print(f"Predicting on {len(file_paths)} files...")
        
        # Create dataset
        dataset = self.create_dataset(file_paths, batch_size)
        
        # Predict
        predictions = self.model.predict(dataset, verbose=1)
        
        # Process results
        results = []
        for i, file_path in enumerate(file_paths):
            pred_idx = np.argmax(predictions[i])
            confidence = predictions[i][pred_idx]
            
            results.append({
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'predicted_class': self.class_names[pred_idx],
                'predicted_index': int(pred_idx),
                'confidence': float(confidence),
                'probabilities': predictions[i].tolist()
            })
        
        return results
    
    def predict_single(self, file_path):
        """
        Predict on a single audio file.
        
        Parameters:
        - file_path: Path to audio file
        
        Returns:
        - Dictionary with prediction results
        """
        results = self.predict([file_path], batch_size=1)
        return results[0]
    
    def predict_folder(self, folder_path, extensions=('.wav', '.mp3', '.flac'), batch_size=32):
        """
        Predict on all audio files in a folder.
        
        Parameters:
        - folder_path: Path to folder containing audio files
        - extensions: Tuple of file extensions to process
        - batch_size: Batch size for processing
        
        Returns:
        - List of dictionaries containing prediction results
        """
        # Get all audio files
        file_paths = []
        for ext in extensions:
            file_paths.extend([str(p) for p in Path(folder_path).rglob(f'*{ext}')])
        
        if not file_paths:
            print(f"No audio files found in {folder_path}")
            return []
        
        return self.predict(file_paths, batch_size)
    
    def get_true_labels_from_folders(self, file_paths):
        """
        Extract true labels from folder structure.
        Assumes files are organized like: dataset/class_name/file.wav
        
        Parameters:
        - file_paths: List of audio file paths
        
        Returns:
        - List of true labels
        """
        true_labels = []
        
        for file_path in file_paths:
            folder_name = os.path.basename(os.path.dirname(file_path))
            if folder_name in self.class_names:
                true_labels.append(folder_name)
            else:
                # Try to find class name in filename
                filename = os.path.basename(file_path).lower()
                found = False
                for class_name in self.class_names:
                    if class_name.lower() in filename:
                        true_labels.append(class_name)
                        found = True
                        break
                if not found:
                    true_labels.append(None)
        
        return true_labels
    
    def get_true_labels_from_dict(self, file_paths, labels_dict):
        """
        Get true labels from a dictionary mapping.
        
        Parameters:
        - file_paths: List of audio file paths
        - labels_dict: Dictionary mapping filename to true label
        
        Returns:
        - List of true labels
        """
        true_labels = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            true_labels.append(labels_dict.get(filename, None))
        return true_labels
    
    def get_true_labels_from_csv(self, file_paths, csv_path, filename_col='filename', label_col='label'):
        """
        Get true labels from a CSV file.
        
        Parameters:
        - file_paths: List of audio file paths
        - csv_path: Path to metadata CSV file
        - filename_col: Name of the column containing filenames
        - label_col: Name of the column containing labels
        
        Returns:
        - List of true labels
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Create dictionary from CSV
        labels_dict = dict(zip(df[filename_col], df[label_col]))
        
        # Get labels for each file
        true_labels = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            label = labels_dict.get(filename, None)
            if label is None:
                print(f"Warning: No label found for {filename}")
            true_labels.append(label)
        
        return true_labels
    
    def evaluate(self, file_paths, true_labels, batch_size=32, average='weighted'):
        """
        Predict and calculate evaluation metrics.
        
        Parameters:
        - file_paths: List of audio file paths
        - true_labels: List of true labels (same order as file_paths)
        - batch_size: Batch size for processing
        - average: Averaging method for metrics ('weighted', 'macro', 'micro')
        
        Returns:
        - Tuple of (results, metrics_dict)
        """
        # Get predictions
        results = self.predict(file_paths, batch_size)
        
        # Extract predicted labels
        y_pred = [r['predicted_class'] for r in results]
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, y_pred, average)
        
        return results, metrics
    
    def calculate_metrics(self, y_true, y_pred, average='weighted'):
        """
        Calculate and display evaluation metrics.
        
        Parameters:
        - y_true: List of true labels
        - y_pred: List of predicted labels
        - average: Averaging method for metrics
        
        Returns:
        - Dictionary containing all metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Print overall metrics
        print("\n" + "="*50)
        print("OVERALL METRICS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("="*50)
        
        # Print detailed classification report
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
        
        # Print confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        print("\nCONFUSION MATRIX:")
        print("Predicted ->")
        print(f"{'True ↓':<15}", end="")
        for name in self.class_names:
            print(f"{name:<15}", end="")
        print()
        for i, name in enumerate(self.class_names):
            print(f"{name:<15}", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[i][j]:<15}", end="")
            print()
        
        return metrics
    
    def save_results(self, results, output_path='predictions.csv'):
        """
        Save prediction results to CSV.
        
        Parameters:
        - results: List of prediction results
        - output_path: Path to save CSV file
        """
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")