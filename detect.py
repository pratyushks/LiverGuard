from fastai.vision.all import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pathlib

# Define all custom functions used during training
def get_x(fname: Path): 
    return fname

def label_func(x): 
    return 'train_masks' / f'{x.stem}_mask.png'

def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()

def cust_foreground_acc(inp, targ):
    "Includes background in the accuracy computation"
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1)

def generate_mask(image_path, output_path):
    """
    Generate a segmentation mask for the given image and save it to the specified output path.
    
    Args:
        image_path (str): Full path to the input image file.
        output_path (str): Full path to save the generated mask file.
    """
    # Fix for Windows Path compatibility in FastAI
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    try:
        MODEL_FILE = "Models/Cancer_Detection.pkl"
        codes = np.array(["background", "liver", "tumor"])  # Segmentation classes

        # Load the trained model
        try:
            learn = load_learner(MODEL_FILE)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

        # Load the input image
        img = PILImage.create(image_path)
        
        # Predict the mask
        try:
            pred, _, _ = learn.predict(img)
            
            fig, ax = plt.subplots(figsize=(4, 4)) 
            ax.imshow(pred, cmap="nipy_spectral")
            ax.axis("off") 
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0) 
            plt.close(fig)
            print(f"Mask saved at {output_path}")
        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
            raise e
    finally:
        # Restore original Path behavior
        pathlib.PosixPath = temp

