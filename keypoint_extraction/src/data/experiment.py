import argparse
from pathlib import Path
from src.data.keypoint_extractor import Preprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mediapipe as mp

def get_args():
    parser = argparse.ArgumentParser(description="Tic Classification Visualization")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Directory with processed data")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cpu or cuda)")
    parser.add_argument("--window_length", type=int, default=5, help="Window length for smoothing")
    parser.add_argument("--polyorder", type=int, default=2, help="Polynomial order for smoothing")
    parser.add_argument("--seq_length", type=int, default=3094, help="Sequence length for each sample")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Directory to save visualizations")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    return parser.parse_args()

def main():
    args = get_args()

    # Ensure the output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run preprocessing
    preprocessor = Preprocess(args.config, args)
    preprocessor.process_videos(args)

if __name__ == "__main__":
    main()