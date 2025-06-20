import numpy as np
import pandas as pd
from pathlib import Path
import random
from scipy.signal import savgol_filter

# CONFIG:
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
RANDOM_SEED = 42
NAN_THRESHOLD = 0.33 # Fraction of NaN values allowed in the data

# VitPose Kps indices for normalization
VITPOSE_LEFT_SHOULDER = 5
VITPOSE_RIGHT_SHOULDER = 6
VITPOSE_LEFT_HIP = 11
VITPOSE_RIGHT_HIP = 12

# MediaPipe Kps indices for normalization
MEDIAPIPE_NOSE_IDX = 1  
MEDIAPIPE_LEFT_EYE_IDX = 446 
MEDIAPIPE_RIGHT_EYE_IDX = 226

# Savitz-Golay smoothing parameters
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 3

CONFIDENCE_THRESHOLD = 0.70   # Threshold for low confidence keypoints
FRAME_LOW_CONFIDENCE = 0.70  # Fraction of frames with low confidence keypoints to remove them
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


VITPOSE_KEYPOINT_NAMES = [
    # Body keypoints (0-22)
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", 
    "right_small_toe", "right_heel",
    # Face keypoints (23-90)
    "face-0", "face-1", "face-2", "face-3", "face-4", "face-5", "face-6", 
    "face-7", "face-8", "face-9", "face-10", "face-11", "face-12", "face-13",
    "face-14", "face-15", "face-16", "face-17", "face-18", "face-19", "face-20",
    "face-21", "face-22", "face-23", "face-24", "face-25", "face-26", "face-27",
    "face-28", "face-29", "face-30", "face-31", "face-32", "face-33", "face-34",
    "face-35", "face-36", "face-37", "face-38", "face-39", "face-40", "face-41",
    "face-42", "face-43", "face-44", "face-45", "face-46", "face-47", "face-48",
    "face-49", "face-50", "face-51", "face-52", "face-53", "face-54", "face-55",
    "face-56", "face-57", "face-58", "face-59", "face-60", "face-61", "face-62",
    "face-63", "face-64", "face-65", "face-66", "face-67",
    # Hand keypoints (91-132)
    "left_hand_root", "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
    "left_forefinger1", "left_forefinger2", "left_forefinger3", "left_forefinger4",
    "left_middle_finger1", "left_middle_finger2", "left_middle_finger3", "left_middle_finger4",
    "left_ring_finger1", "left_ring_finger2", "left_ring_finger3", "left_ring_finger4",
    "left_pinky_finger1", "left_pinky_finger2", "left_pinky_finger3", "left_pinky_finger4",
    "right_hand_root", "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
    "right_forefinger1", "right_forefinger2", "right_forefinger3", "right_forefinger4",
    "right_middle_finger1", "right_middle_finger2", "right_middle_finger3", "right_middle_finger4",
    "right_ring_finger1", "right_ring_finger2", "right_ring_finger3", "right_ring_finger4",
    "right_pinky_finger1", "right_pinky_finger2", "right_pinky_finger3", "right_pinky_finger4"
]



def get_file_list(base_path):
    """
    Get a list of all .npy files in the specified directories according to the three categories:
    "tourette", "ftlb", and "both". 
    Returns a list of tuples where each tuple contains the file path and its category.
    """

    file_list = []
    tourette_dir = Path(base_path) / "tourette"
    ftlb_dir = Path(base_path) / "ftlb"
    both_dir = Path(base_path) / "both"

    if tourette_dir.exists():
        for f in tourette_dir.glob("*.npy"):
            file_list.append((f, "tourette"))

    if ftlb_dir.exists():
        for f in ftlb_dir.glob("*.npy"):
            file_list.append((f, "ftlb"))

    if both_dir.exists():
        for f in both_dir.glob("*.npy"):
            file_list.append((f, "both"))

    return file_list

def split_by_category(file_list, test_size, min_test = 5):

    n = len(file_list)
    if n < 10:
        n_test = max(1, int(round(n * test_size))) # at least one file in test set
    else:
        n_test = max(min_test, int(round(n * test_size))) # at least 5 files in test set

    n_test = min(n_test, n - 1)  
    random.shuffle(file_list)
    return file_list[n_test:], file_list[:n_test]

def check_nan_fraction(data):
    """
    Check the fraction of NaN values over all frames, keypoints, and two channels (x, y).
    """
    channels_for_nan = data[..., :2] # x and y coordinates, shape (frames, kps, 2)
    total_pts = channels_for_nan.size
    nan_cnt = np.isnan(channels_for_nan).sum()
    return nan_cnt / total_pts

def fill_nans(data):
    """
    Fill NaN values in the data using forward fill, backward fill, and linear interpolation.
    """
    filled = np.copy(data)
    frames, kps, dims = filled.shape

    for kp in range(kps):
        for dim in range(2): # x and y coordinates
            series = pd.Series(filled[:, kp, dim])
            series_filled = (series.fillna(method='ffill')
                            .fillna(method='bfill')
                            .interpolate(method='linear')
                            .fillna(0))
            filled[:, kp, dim] = series_filled.values

    return filled

def process_file(file_path):
    """
    Process a single file by loading the data, checking NaN fraction, filling NaNs,
    and returning the processed data.
    """

    try:
        data = np.load(file_path) # (frames, kps, 3)
    except:
        print(f"Error loading file: {file_path}")
        return None

    frac_nan = check_nan_fraction(data)
    if frac_nan > NAN_THRESHOLD:
        print(f"Skipping file {file_path.name} due to high NaN fraction: {frac_nan:.2f}")
        return None
    
    data_filled = fill_nans(data)

    frac_check = check_nan_fraction(data_filled)

    return data_filled


def normalize_vitpose_per_frame(frame_data):
    """
    Normalize a single frame of VitPose data by centering around the pelvis
    and normalizing by the shoulder distance.
    """

    left_shoulder = frame_data[VITPOSE_LEFT_SHOULDER, :2]
    right_shoulder = frame_data[VITPOSE_RIGHT_SHOULDER, :2]
    left_hip = frame_data[VITPOSE_LEFT_HIP, :2]
    right_hip = frame_data[VITPOSE_RIGHT_HIP, :2]

    shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)
    if shoulder_dist < 1e-6:
        return frame_data
    
    pelvis_center = (left_hip + right_hip) / 2

    frame_data[:, :2] -= pelvis_center  # Centering around pelvis
    frame_data[:, :2] /= shoulder_dist  # Normalizing by shoulder distance
    return frame_data


def normalize_vitpose(data):
    """
    Normalize VitPose data for each frame.
    """

    total_kps = data.shape[1]
    vt_count = total_kps - 468 # we have always 468 for Mediapipe

    vitpose_part = data[:,:vt_count, :]
    mediapipe_part = data[:,vt_count:, :]

    for f in range(vitpose_part.shape[0]):
        vitpose_part[f] = normalize_vitpose_per_frame(vitpose_part[f])
    
    if mediapipe_part is not None:
        return np.concatenate([vitpose_part, mediapipe_part], axis=1)
    else:
        return vitpose_part

def normalize_mediapipe_per_frame(frame_data):
    """
    Normalize a single frame of Mediapipe data by centering around the nose
    and normalizing by the distance between the left and right eyes.
    """

    nose_kp = frame_data[MEDIAPIPE_NOSE_IDX, :2] 
    left_eye = frame_data[MEDIAPIPE_LEFT_EYE_IDX, :2]
    right_eye = frame_data[MEDIAPIPE_RIGHT_EYE_IDX, :2]

    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1e-6:
        return frame_data

    frame_data[:, :2] -= nose_kp  # Centering around nose
    frame_data[:, :2] /= eye_dist  # Normalizing by eye distance
    return frame_data

def normalize_mediapipe(data):
    """
    Normalize Mediapipe data for each frame.
    """

    total_kps = data.shape[1]
    vt_count = total_kps - 468  # we have always 468 for Mediapipe

    vitpose_part = data[:, :vt_count, :]
    mediapipe_part = data[:, vt_count:, :]

    for f in range(mediapipe_part.shape[0]):
        mediapipe_part[f] = normalize_mediapipe_per_frame(mediapipe_part[f])

    return np.concatenate([vitpose_part, mediapipe_part], axis=1)


def apply_savgol_smoothing(data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER):
    """
    Apply Savitz-Golay smoothing to the data.
    """

    frames, kps, dims = data.shape

    if frames < 3:
        return data # Not enough frames to apply smoothing
    
    if window_length > frames:
        wl = frames if frames % 2 == 1 else (frames - 1)
        if wl < 3:
            return data
        window_length = wl

    smoothed_data = np.copy(data)
    for kp in range(kps):
        for dim in range(2):
            series = smoothed_data[:, kp, dim]
            if not np.any(np.isnan(series)):
                smoothed_data[:, kp, dim] = savgol_filter(series, window_length, polyorder, mode='nearest')

    return smoothed_data


def normalize_and_smooth(data):
    """
    Normalize and smooth the data.
    This function applies normalization for both VitPose and Mediapipe data,
    and then applies Savitz-Golay smoothing.
    """

    norm_vitpose = normalize_vitpose(data) # vitpose normalization

    norm_mediapipe = normalize_mediapipe(data) # mediapipe normalization

    # Savitz-Golay smoothing
    smoothed_data = apply_savgol_smoothing(norm_mediapipe, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)

    return smoothed_data


def remove_low_confidence_vitpose(data):
    """
    Remove low confidence keypoints from VitPose data.
    This function checks the confidence scores of VitPose keypoints and removes those
    that have a high fraction of low confidence across frames.
    Returns the modified data and a list of remaining VitPose features.
    """

    frames, kps, dims = data.shape
    vit_kps = 133

    vitpose_data = data[:, :vit_kps, :]  
    mediapipe_data = data[:, vit_kps:, :]  

    if vitpose_data.size == 0:
        return data, None
    
    vitpose_conf = vitpose_data[..., 2] # confidence scores -> (frames , vit_kps)
    low_conf_mask = (vitpose_conf < CONFIDENCE_THRESHOLD)  # Threshold for low confidence
    low_conf_frac = np.mean(low_conf_mask, axis=0) # shape (vit_kps,)

    kps_to_remove = np.where(low_conf_frac > FRAME_LOW_CONFIDENCE)[0] 
    keep_mask = np.ones(vit_kps, dtype=bool)
    keep_mask[kps_to_remove] = False

    kept_indices = np.where(keep_mask)[0]

    vitpose_features = []
    for idx in kept_indices:
        kp_name = VITPOSE_KEYPOINT_NAMES[idx]
        vitpose_features.extend([f"vitpose_{kp_name}_x", f"vitpose_{kp_name}_y"]) # only kept indices -> shape (kept_kps *2,)

    removed_vitpose = vitpose_data[:, keep_mask, :]  # remove low confidence keypoints
    final_data = np.concatenate([removed_vitpose, mediapipe_data], axis=1) 

    return final_data, vitpose_features


    
def preprocess_files(file_list, mode, output_path):
    """
    Preprocess a list of files by loading, normalizing, smoothing, and saving the data.
    The processed data is saved in the specified output path under the given mode (train/test).
    """

    for (file_path, category) in file_list:
        print(f"[{mode.upper()}]  Processing file: {file_path.name}, Category: {category}")

        processed = process_file(file_path)
        if processed is None:
            continue

        data_normalized = normalize_and_smooth(processed)
        final_data, vitpose_features = remove_low_confidence_vitpose(data_normalized)

        out_dir = output_path / mode / category
        out_file = out_dir / f"{file_path.stem}.npz"

        np.savez(out_file, data=final_data, vitpose_features = np.array(vitpose_features, dtype=object))



def main_preprocessing(base_path, output_path):
    """
    Main preprocessing function that orchestrates the entire preprocessing pipeline.
    It retrieves the file list, splits the data into train and test sets, processes the files,
    and saves the processed data in the specified output path.
    """

    base_path = Path(base_path)
    output_path = Path(output_path)

    file_list = get_file_list(base_path) # list of tuples (file_path, category)

    random.seed(RANDOM_SEED)

    tourette_files = [f for f in file_list if f[1] == "tourette"]
    ftlb_files = [f for f in file_list if f[1] == "ftlb"]
    both_files = [f for f in file_list if f[1] == "both"] 

    tourette_train, tourette_test = split_by_category(tourette_files, 0.2)
    ftlb_train, ftlb_test = split_by_category(ftlb_files, 0.2)
    both_train, both_test = split_by_category(both_files, 0.2)

    train_list = tourette_train + ftlb_train + both_train
    test_list = tourette_test + ftlb_test + both_test

    print(f"Total files: {len(file_list)}, Train files: {len(train_list)}, Test files: {len(test_list)}")

    # output directories
    train_t_dir = output_path / "train" / "tourette"
    train_f_dir = output_path / "train" / "ftlb"
    train_b_dir = output_path / "train" / "both"
    test_t_dir = output_path / "test" / "tourette"
    test_f_dir = output_path / "test" / "ftlb"
    test_b_dir = output_path / "test" / "both"

    for d in [train_t_dir, train_f_dir, train_b_dir, test_t_dir, test_f_dir, test_b_dir]:
        d.mkdir(parents=True, exist_ok=True)

    preprocess_files(train_list, "train", output_path) # 2 arrays: 'data' array of shape (frames, kps, 3), 'vitpose_features' array of shape (kept_kps *2,)
    preprocess_files(test_list, "test", output_path)   # 2 arrays: 'data' array of shape (frames, kps, 3), 'vitpose_features' array of shape (kept_kps *2,)


if __name__ == "__main__":
    BASE_PATH = "/user/jafar.bakhshaliyev/u11836/.project/dir.project/bakhshaliyev/DigiStruc/data"    # change this to your base path
    OUTPUT_PATH = "/user/jafar.bakhshaliyev/u11836/.project/dir.project/bakhshaliyev/DigiStruc/dataset" # change this to your output path

    main_preprocessing(BASE_PATH, OUTPUT_PATH)