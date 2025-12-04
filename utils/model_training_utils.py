import os
import glob

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