import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class SVMAudioPredictor:
    def __init__(self, model_path, scaler_path=None, feature_extractor=None, label_encoder=None):
        """
        Initialize the SVM evaluator
        
        Args:
            model_path: Path to the pickled SVM model
            scaler_path: Path to the pickled StandardScaler (CRITICAL - must use training scaler)
            feature_extractor: Function to extract features from audio files
                              Should take file_path and return feature vector
            label_encoder: LabelEncoder used during training (optional)
                          If None, will create mapping from predictions
        """
        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the scaler (CRITICAL FIX)
        self.scaler = None
        if scaler_path is not None:
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✓ Scaler loaded from {scaler_path}")
                print(f"  Feature dimensions: {len(self.scaler.mean_)}")
            except Exception as e:
                print(f"⚠ Warning: Could not load scaler from {scaler_path}: {e}")
                print("  Predictions may be incorrect without proper scaling!")
        else:
            print("⚠ Warning: No scaler provided!")
            print("  Features will NOT be scaled - this will likely cause incorrect predictions!")
        
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        
        # Print model info
        print(f"✓ Model loaded with {len(self.model.classes_)} classes")
        if self.label_encoder:
            print(f"  Classes: {self.label_encoder.classes_}")
    
    def extract_features_batch(self, file_paths, batch_size=32):
        """
        Extract features from audio files in batches
        
        Args:
            file_paths: List of audio file paths
            batch_size: Number of files to process at once
            
        Returns:
            numpy array of features
        """
        features = []
        feature_dim = None  # Store feature dimension
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(file_paths)-1)//batch_size + 1}")
            
            for file_path in batch:
                try:
                    # Extract features using your feature extractor
                    feature = self.feature_extractor(file_path)
                    features.append(feature)
                    
                    # Store feature dimension from first successful extraction
                    if feature_dim is None:
                        feature_dim = feature.shape
                        print(f"  Feature dimension: {feature_dim}")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    # Add zero vector if we know the dimension, otherwise skip
                    if feature_dim is not None:
                        features.append(np.zeros(feature_dim))
                    else:
                        print(f"Skipping file (feature dimension unknown)")
        
        features_array = np.array(features)
        
        # Verify feature dimensions match scaler expectations
        if self.scaler is not None:
            expected_dim = len(self.scaler.mean_)
            actual_dim = features_array.shape[1] if len(features_array.shape) > 1 else features_array.shape[0]
            
            if expected_dim != actual_dim:
                print(f"\n⚠ WARNING: Feature dimension mismatch!")
                print(f"  Expected: {expected_dim} features (from scaler)")
                print(f"  Got: {actual_dim} features")
                print(f"  This will cause incorrect predictions!")
        
        return features_array
    
    def predict(self, file_paths, batch_size=32):
        """
        Make predictions on audio files
        
        Args:
            file_paths: List of audio file paths
            batch_size: Batch size for processing
            
        Returns:
            predictions: List of predicted labels
            decision_scores: Decision function scores (confidence)
        """
        # Extract features
        print("Extracting features...")
        X = self.extract_features_batch(file_paths, batch_size)
        
        # Scale features (CRITICAL STEP)
        if self.scaler is not None:
            print("Scaling features...")
            X_scaled = self.scaler.transform(X)
            print(f"  Scaled feature range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        else:
            print("⚠ Skipping scaling (no scaler available)")
            X_scaled = X
        
        # Make predictions
        print("Making predictions...")
        predictions_encoded = self.model.predict(X_scaled)
        
        # Get decision scores (useful for debugging)
        decision_scores = self.model.decision_function(X_scaled)
        
        # Decode predictions if label encoder exists
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
        else:
            predictions = predictions_encoded
        
        return predictions, decision_scores
    
    def evaluate(self, file_paths, true_labels, batch_size=32, plot_cm=True):
        """
        Evaluate the SVM model on test data
        
        Args:
            file_paths: List of audio file paths
            true_labels: List of true emotion labels
            batch_size: Batch size for processing
            plot_cm: Whether to plot confusion matrix
            
        Returns:
            predictions: List of predicted labels
            metrics: Dictionary containing all metrics
        """
        # Make predictions
        predictions, decision_scores = self.predict(file_paths, batch_size)
        
        # Check prediction diversity (debugging)
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        print(f"\nPrediction distribution:")
        for pred, count in zip(unique_preds, pred_counts):
            print(f"  {pred}: {count} ({count/len(predictions)*100:.1f}%)")
        
        if len(unique_preds) == 1:
            print("\n⚠ WARNING: Model is predicting only ONE class!")
            print("  This indicates a serious problem with:")
            print("  1. Feature scaling (scaler not loaded/applied)")
            print("  2. Feature extraction (different from training)")
            print("  3. Model training (class imbalance or convergence issues)")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        # Get detailed classification report
        report = classification_report(true_labels, predictions, zero_division=0)
        
        # Print results
        print("\n" + "="*50)
        print("OVERALL METRICS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("="*50)
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(report)
        
        # Plot confusion matrix
        if plot_cm:
            self.plot_confusion_matrix(true_labels, predictions)
        
        # Store metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(true_labels, predictions),
            'decision_scores': decision_scores
        }
        
        return predictions, metrics
    
    def debug_predictions(self, file_paths, true_labels, num_samples=5):
        """
        Debug predictions by showing decision scores for sample files
        
        Args:
            file_paths: List of audio file paths
            true_labels: List of true labels
            num_samples: Number of samples to debug
        """
        print("\n" + "="*60)
        print("DEBUGGING PREDICTIONS")
        print("="*60)
        
        num_samples = min(num_samples, len(file_paths))
        
        for i in range(num_samples):
            file_path = file_paths[i]
            true_label = true_labels[i]
            
            # Extract and scale features
            feature = self.feature_extractor(file_path)
            if self.scaler is not None:
                feature_scaled = self.scaler.transform(feature.reshape(1, -1))
            else:
                feature_scaled = feature.reshape(1, -1)
            
            # Get prediction and scores
            pred_encoded = self.model.predict(feature_scaled)[0]
            scores = self.model.decision_function(feature_scaled)[0]
            
            if self.label_encoder is not None:
                pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
            else:
                pred_label = pred_encoded
            
            print(f"\nSample {i+1}:")
            print(f"  File: {file_path}")
            print(f"  True: {true_label}")
            print(f"  Predicted: {pred_label}")
            print(f"  Decision scores:")
            
            if self.label_encoder is not None:
                for cls, score in zip(self.label_encoder.classes_, scores):
                    marker = "✓" if cls == pred_label else " "
                    print(f"    {marker} {cls:12s}: {score:8.3f}")
            else:
                for idx, score in enumerate(scores):
                    marker = "✓" if idx == pred_encoded else " "
                    print(f"    {marker} Class {idx}: {score:8.3f}")
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """
        Plot confusion matrix
        
        Args:
            true_labels: True labels
            predictions: Predicted labels
        """
        # Get unique labels
        labels = sorted(list(set(true_labels)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()