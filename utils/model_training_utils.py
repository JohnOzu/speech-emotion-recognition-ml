import os
import glob
import matplotlib.pyplot as plt

def cleanup_checkpoints(checkpoints_dir, keep_best=True, keep_latest=5):
    "Clean up old checkpoint files"
    
    # Get all checkpoint files except best_model.keras
    all_checkpoints = glob.glob(os.path.join(checkpoints_dir, 'epoch_*.keras'))
    all_checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    # Keep only latest N
    to_delete = all_checkpoints[keep_latest:]
    
    for filepath in to_delete:
        os.remove(filepath)
        print(f"Deleted: {filepath}")
    
    print(f"\nKept {keep_latest} latest checkpoints")
    if keep_best:
        print("Kept: best_model.keras")

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()