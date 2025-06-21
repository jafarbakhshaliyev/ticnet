import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis, pearsonr
from scipy.signal import welch
import warnings
from multiprocessing import Pool, cpu_count
import time
import gc

warnings.filterwarnings('ignore')

# CONFIG: 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MEDIAPIPE_PTS = 468
N_WORKERS = 64

WINDOW_SIZES = [5, 10, 15, 20]
OVERLAP_RATIO = 0.5

MEDIAPIPE_REGIONS = {
    'right_eye': [
                246, 161, 160, 159, 158, 157, 173,                     
                33, 7, 163, 144, 145, 153, 154, 155, 133,             
                247, 30, 29, 27, 28, 56, 190,                         
                130, 25, 110, 24, 23, 22, 26, 112, 243,               
                113, 225, 224, 223, 222, 221, 189,                     
                226, 31, 228, 229, 230, 231, 232, 233, 244,           
                143, 111, 117, 118, 119, 120, 121, 128, 245],
    'left_eye': [
                466, 388, 387, 386, 385, 384, 398,                    
                263, 249, 390, 373, 374, 380, 381, 382, 362,          
                467, 260, 259, 257, 258, 286, 414,                    
                359, 255, 339, 254, 253, 252, 256, 341, 463,           
                342, 445, 444, 443, 442, 441, 413,                    
                446, 261, 448, 449, 450, 451, 452, 453, 464,          
                372, 340, 346, 347, 348, 349, 350, 357, 465],
    'right_eyebrow': [156, 70, 63, 105, 66, 107, 55, 193, 35, 124, 46, 53, 52, 65], 
    'left_eyebrow': [383, 300, 293, 334, 296, 336, 285, 417, 265, 353, 276, 283, 282, 295], 
    'mouth_outer': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375], 
    'mouth_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324],
    'nose': [1, 2, 98, 327, 168], 
    'right_cheek': [205], 
    'left_cheek': [425], 
    'face_silhouette': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109] 
}
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class FeatureExtractor:
    def __init__(self, data_path, mediapipe_pts, n_workers=None):
        self.data_path = Path(data_path)
        self.mediapipe_pts = mediapipe_pts
        self.n_workers = n_workers or min(cpu_count(), 8)


        self.window_sizes = WINDOW_SIZES
        self.overlap_ratio = OVERLAP_RATIO

        self._prepare_region_indices()

        print(f"Initialized with {self.n_workers} workers.")



    def _prepare_region_indices(self):
        """
        Prepare indices for each region based on the MEDIAPIPE_REGIONS dictionary.
        """
        self.region_names = list(MEDIAPIPE_REGIONS.keys())
        self.region_indices_flat =[]
        self.region_starts =[]
        self.region_lengths = []

        current_start = 0
        for region in self.region_names:
            indices = MEDIAPIPE_REGIONS[region] # [ indices ]
            self.region_starts.append(current_start)
            self.region_lengths.append(len(indices))
            self.region_indices_flat.extend(indices)
            current_start += len(indices)

        self.region_indices_flat = np.array(self.region_indices_flat, dtype=np.int32)
        self.region_starts = np.array(self.region_starts, dtype=np.int32)
        self.region_lengths = np.array(self.region_lengths, dtype=np.int32)

    def load_data(self, npz_file):
        """
        Load data from a .npz file and return the data and vitpose features.
        """
        try:
            npz_file = np.load(npz_file, allow_pickle=True)
            data = npz_file['data']
            vitpose_features = npz_file.get('vitpose_features', []).tolist()
            data = data.astype(np.float32)

            return data, vitpose_features
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            return None, None

    def create_sliding_windows(self, data, window_size, overlap_ratio):
        """
        Create sliding windows from the data with the specified window size and overlap ratio.
        """
        frames = data.shape[0]
        step_size = max(1, int(window_size * (1 - overlap_ratio)))

        n_windows = max(0, (frames - window_size) // step_size + 1)

        if n_windows == 0:
            return np.array([]).reshape(0, window_size, data.shape[1], data.shape[2])

        window_starts = np.arange(0, n_windows * step_size, step_size) # [0, step_size, 2*step_size, ...] shape (n_windows,)
        window_indices = window_starts[:, None] + np.arange(window_size) # shape (n_windows, window_size)

        return data[window_indices] # shape (n_windows, window_size, kps, 2)

    
    def calculate_movement(self, mediapipe_part, region_indices_flat, region_starts, region_lengths):
        """
        Calculate the movement for each region in the mediapipe_part data.
        mediapipe_part: shape (frames, kps, 2)
        region_indices_flat: flat array of indices for all regions
        region_starts: start indices for each region
        region_lengths: lengths of each region
        Returns a 2D array of movements for each region, shape (n_regions, frames - 1).
        """

        frames, kps, dims = mediapipe_part.shape
        n_regions = len(region_starts)

        all_movements = np.zeros((n_regions, frames - 1))

        for region in range(n_regions):
            start_idx = region_starts[region]
            length = region_lengths[region]

            region_kp_indices = region_indices_flat[start_idx:start_idx + length]
            valid_indices = region_kp_indices[region_kp_indices < kps]

            if len(valid_indices) > 0:

                region_data = mediapipe_part[:, valid_indices, :]  # shape (frames, n_kps, 2)

                diff = region_data[1:] - region_data[:-1]  # shape (frames-1, n_kps, 2)

                movements_per_kp = np.sqrt(np.sum(diff ** 2, axis=-1))  # shape (frames-1, n_kps)

                all_movements[region] = np.mean(movements_per_kp, axis=1)  # shape (frames-1,)


        return all_movements # shape (n_regions, frames - 1)
            

    def extract_statistical_feats(self, movements, region_names, window_size):
        """
        Extract statistical features from the movements of each region.
        movements: shape (n_regions, frames - 1)
        region_names: list of region names
        window_size: size of the sliding window used for feature extraction
        Returns a dictionary of statistical features for each region.
        """

        features = {}
        n_regions = movements.shape[0] # shape (n_regions, frames - 1)

        means = np.mean(movements, axis=1)  # shape (n_regions,)
        stds = np.std(movements, axis=1)  # shape (n_regions,)
        maxs = np.max(movements, axis=1)  # shape (n_regions,)
        mins = np.min(movements, axis=1)  # shape (n_regions,)
        ranges = maxs - mins  # shape (n_regions,)

        burstiness = np.divide(stds, means, out=np.zeros_like(stds), where=means > 1e-8)  # shape (n_regions,)
        burstiness = np.where((stds > 1e-8) & (means > 1e-8), burstiness, 0)  # shape (n_regions,)

        for i, region in enumerate(region_names):
            features[f"{region}_movement_mean_ws{window_size}_mean"] = means[i]
            features[f"{region}_movement_std_ws{window_size}_mean"] = stds[i]
            features[f"{region}_movement_max_ws{window_size}_mean"] = maxs[i]
            features[f"{region}_movement_range_ws{window_size}_mean"] = ranges[i]
            features[f"{region}_movement_burstiness_ws{window_size}_mean"] = burstiness[i]

        return features  # returns a dictionary of statistical features for each region

    
    def extract_freq_feats(self, movements, region_names, window_size, fs=30):
        """
        Extract frequency features from the movements of each region using Welch's method.
        movements: shape (n_regions, frames - 1)
        region_names: list of region names
        window_size: size of the sliding window used for feature extraction
        fs: sampling frequency, default is 30 Hz
        Returns a dictionary of frequency features for each region.
        """

        features = {}

        for i, region in enumerate(region_names):
            region_movements = movements[i] # shape (frames - 1,)
            if len(region_movements) > 8 and np.std(region_movements) > 1e-8:
                try:
                    freqs, psd = welch(region_movements, fs = fs, nperseg=min(len(region_movements), 8))

                    if len(psd) > 1 and not np.any(np.isnan(psd)) and not np.any(np.isinf(psd)):
                        dominant_freq_idx = np.argmax(psd[1:]) + 1  # skip the DC component
                        dominant_freq = freqs[dominant_freq_idx]

                        psd_sum = np.sum(psd)
                        if psd_sum > 0:
                            spectral_centroid = np.sum(freqs * psd) / psd_sum
                        else:
                            spectral_centroid = 0.0

                        cumulative_energy = np.cumsum(psd) /psd_sum if psd_sum > 1e-8 else np.zeros_like(psd)
                        rolloff_idx = np.where(cumulative_energy >= 0.85)[0]
                        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

                        features[f"{region}_dominant_freq_ws{window_size}_mean"] = dominant_freq
                        features[f"{region}_spectral_centroid_ws{window_size}_mean"] = spectral_centroid
                        features[f"{region}_spectral_rolloff_ws{window_size}_mean"] = spectral_rolloff
                    else:
                        features[f"{region}_dominant_freq_ws{window_size}_mean"] = 0.0
                        features[f"{region}_spectral_centroid_ws{window_size}_mean"] = 0.0
                        features[f"{region}_spectral_rolloff_ws{window_size}_mean"] = 0.0

                except Exception:
                    features[f"{region}_dominant_freq_ws{window_size}_mean"] = 0.0
                    features[f"{region}_spectral_centroid_ws{window_size}_mean"] = 0.0
                    features[f"{region}_spectral_rolloff_ws{window_size}_mean"] = 0.0

            else:
                features[f"{region}_dominant_freq_ws{window_size}_mean"] = 0.0
                features[f"{region}_spectral_centroid_ws{window_size}_mean"] = 0.0
                features[f"{region}_spectral_rolloff_ws{window_size}_mean"] = 0.0
            
        return features  # returns a dictionary of frequency features for each region
      


    def extract_cross_region_correlations(self, movements, region_names, window_size):
        """
        Extract cross-region correlation features from the movements of each region.
        movements: shape (n_regions, frames - 1)
        region_names: list of region names
        window_size: size of the sliding window used for feature extraction
        Returns a dictionary of cross-region correlation features for each region pair.
        """

        features = {}
        n_regions = len(region_names)

        region_pairs = [
            (0, 1),  # right_eye, left_eye
            (2, 3),  # right_eyebrow, left_eyebrow
            (0, 2),  # right_eye, right_eyebrow
            (1, 3),  # left_eye, left_eyebrow
            (4, 5),  # mouth_outer, mouth_inner
            (4, 6),  # mouth_outer, nose
            (7, 8),  # right_cheek, left_cheek
            (0, 7),  # right_eye, right_cheek
            (1, 8),  # left_eye, left_cheek
        ]

        correlations = []
        for i, j in region_pairs:
            if i < n_regions and j < n_regions:
                move1, move2 = movements[i], movements[j]
                if (len(move1) > 3 and len(move2) > 3 and np.std(move1) > 1e-8 and np.std(move2) > 1e-8):
                    correlation = np.corrcoef(move1, move2)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
                        region1, region2 = region_names[i], region_names[j]
                        features[f"corr_{region1}_{region2}_ws{window_size}_mean"] = correlation
                    else:
                        region1, region2 = region_names[i], region_names[j]
                        features[f"corr_{region1}_{region2}_ws{window_size}_mean"] = 0.0
                else:
                    region1, region2 = region_names[i], region_names[j]
                    features[f"corr_{region1}_{region2}_ws{window_size}_mean"] = 0.0

        features[f"global_corr_ws{window_size}_mean"] = np.mean(correlations) if correlations else 0.0

        return features

    def extract_body_movements(self, vitpose_part, vitpose_features):
        """
        Extract body movements from the vitpose_part data.
        vitpose_part: shape (frames, n_vitpose_kps, 2)
        vitpose_features: list of vitpose features, each feature is a list of coordinates
        Returns a dictionary of body movements for each part.
        """

        if not vitpose_features or vitpose_part.shape[1] == 0:
            return {}
        
        frames, n_vitpose_kps, dims = vitpose_part.shape

        if frames < 2:
            return {}

        body_movements = {}

        for i, feature_name in enumerate(vitpose_features):
            if i % 2 == 0:  # x coordinates only
                base_name = feature_name.replace('vitpose_', '').replace('_x', '')
                keypoint_idx = i // 2
                
                if keypoint_idx < n_vitpose_kps:  

                    kp_data = vitpose_part[:, keypoint_idx, :]  # shape (frames, 2)
                    diff = kp_data[1:] - kp_data[:-1]           # shape (frames-1, 2)
                    movement = np.sqrt(np.sum(diff**2, axis=-1)) # shape  (frames-1,)
                    
                    body_movements[base_name] = movement
          

        return body_movements  


    def extract_vitpose_feats(self, window, vitpose_cnt, vitpose_features, window_size):
        """ Extract features from the vitpose part of the window.   
        window: shape (window_size, kps, 2) 
        vitpose_cnt: number of vitpose keypoints
        vitpose_features: list of vitpose features, each feature is a list of coordinates
        window_size: size of the sliding window used for feature extraction
        Returns a dictionary of features for the vitpose part of the window.
        """

        features = {}
        if not vitpose_features:
            return features
        
        vitpose_part = window[:, :vitpose_cnt, :]
        if vitpose_part.shape[1] == 0:
            return features
        
        body_movements = self.extract_body_movements(vitpose_part, vitpose_features)

        for part_name, movement_series in body_movements.items():
            if len(movement_series) > 0:
                features[f"body_{part_name}_movement_mean_ws{window_size}_mean"] = np.mean(movement_series)
                features[f"body_{part_name}_movement_std_ws{window_size}_mean"] = np.std(movement_series)
                features[f"body_{part_name}_movement_max_ws{window_size}_mean"] = np.max(movement_series)
                features[f"body_{part_name}_movement_range_ws{window_size}_mean"] = np.max(movement_series) - np.min(movement_series)
                
                # freq features for vitpose
                if len(movement_series) > 8 and np.std(movement_series) > 1e-8:
                    try:
                        freqs, psd = welch(movement_series, fs=30, nperseg=min(len(movement_series), 8))
                        
                        if len(psd) > 1 and not np.any(np.isnan(psd)) and not np.any(np.isinf(psd)) and np.sum(psd) > 1e-8:
                            # dominant frequency
                            dominant_freq_idx = np.argmax(psd[1:]) + 1
                            features[f"body_{part_name}_dominant_freq_ws{window_size}_mean"] = freqs[dominant_freq_idx]
                            
                            # spectral centroid
                            psd_sum = np.sum(psd)
                            features[f"body_{part_name}_spectral_centroid_ws{window_size}_mean"] = np.sum(freqs * psd) / psd_sum
                            
                            # spectral rolloff
                            cumulative_energy = np.cumsum(psd) / psd_sum
                            rolloff_idx = np.where(cumulative_energy >= 0.85)[0]
                            features[f"body_{part_name}_spectral_rolloff_ws{window_size}_mean"] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
                        else:
                            features[f"body_{part_name}_dominant_freq_ws{window_size}_mean"] = 0.0
                            features[f"body_{part_name}_spectral_centroid_ws{window_size}_mean"] = 0.0
                            features[f"body_{part_name}_spectral_rolloff_ws{window_size}_mean"] = 0.0
                    except:
                        features[f"body_{part_name}_dominant_freq_ws{window_size}_mean"] = 0.0
                        features[f"body_{part_name}_spectral_centroid_ws{window_size}_mean"] = 0.0
                        features[f"body_{part_name}_spectral_rolloff_ws{window_size}_mean"] = 0.0
                else:
                    features[f"body_{part_name}_dominant_freq_ws{window_size}_mean"] = 0.0
                    features[f"body_{part_name}_spectral_centroid_ws{window_size}_mean"] = 0.0
                    features[f"body_{part_name}_spectral_rolloff_ws{window_size}_mean"] = 0.0
            else:
                features[f"body_{part_name}_movement_mean_ws{window_size}_mean"] = 0.0
                features[f"body_{part_name}_movement_std_ws{window_size}_mean"] = 0.0
                features[f"body_{part_name}_movement_max_ws{window_size}_mean"] = 0.0
                features[f"body_{part_name}_movement_range_ws{window_size}_mean"] = 0.0
                features[f"body_{part_name}_dominant_freq_ws{window_size}_mean"] = 0.0
                features[f"body_{part_name}_spectral_centroid_ws{window_size}_mean"] = 0.0
                features[f"body_{part_name}_spectral_rolloff_ws{window_size}_mean"] = 0.0

        return features
        
    def extract_window_feats(self, window, vitpose_cnt, window_size, vitpose_features):

        features = {}

        frames, kps, dims = window.shape

        mp_start_idx = vitpose_cnt # mediapipe start index
        mediapipe_part = window[:, mp_start_idx:, :]

        if mediapipe_part.shape[1] < max([max(indices) for indices in MEDIAPIPE_REGIONS.values()]):
            return features

        movements = self.calculate_movement(mediapipe_part, self.region_indices_flat, self.region_starts, self.region_lengths) # shape (n_regions, frames - 1)

        stat_features = self.extract_statistical_feats(movements, self.region_names, window_size) # 
        features.update(stat_features)

        freq_features = self.extract_freq_feats(movements, self.region_names, window_size) 
        features.update(freq_features)

        corr_features = self.extract_cross_region_correlations(movements, self.region_names, window_size)
        features.update(corr_features)

        if vitpose_features and vitpose_cnt > 0:
            body_features = self.extract_vitpose_feats(window, vitpose_cnt, vitpose_features, window_size)
            features.update(body_features)



        return features # returns a dictionary of features for the window


    def extract_multi_scale_features(self, data, vitpose_features):
        """
        Extract multi-scale features from the data using sliding windows.
        data: shape (frames, kps, 2)
        vitpose_features: list of vitpose features, each feature is a list of coordinates
        Returns a dictionary of multi-scale features.
        """
        all_features = {}
        vitpose_cnt = len(vitpose_features) // 2 if vitpose_features else 0 # as it has x and y coordinates

        for window_size in self.window_sizes:

            windows = self.create_sliding_windows(data, window_size, self.overlap_ratio) # shape (n_windows, window_size, kps, 2)


            if len(windows) == 0:
                continue

            window_features_list = []

            batch_size = min(16, len(windows))  
            for batch_start in range(0, len(windows), batch_size):
                batch_end = min(batch_start + batch_size, len(windows))
                batch_windows = windows[batch_start:batch_end]

                for window in batch_windows:
                    window_feats = self.extract_window_feats(window, vitpose_cnt, window_size, vitpose_features) # input shape (window_size, kps, 2), output shape (kps, 2*vitpose_cnt + 1)
                    window_features_list.append(window_feats)
            
            if window_features_list:
                window_df = pd.DataFrame(window_features_list).fillna(0)

                for col in window_df.columns:
                    values = window_df[col].values
                    all_features[f"{col}_mean"] = np.mean(values)
                    all_features[f"{col}_std"] = np.std(values)
                    all_features[f"{col}_max"] = np.max(values)
        
        return all_features  # returns a dictionary of multi-scale features

    def calculate_total_movement(self, data):
        """
        Calculate the total movement across all keypoints in the data.
        data: shape (frames, kps, 2)
        Returns a 1D array of movements for each frame, shape (frames - 1,).
        """
        frames, kps, dims = data.shape

        diff = data[1:] - data[:-1]  # shape (frames-1, kps, 2)

        movements_per_kp = np.sqrt(np.sum(diff ** 2, axis=-1))  # shape (frames-1, kps)

        movements = np.mean(movements_per_kp, axis=1)  # shape (frames-1,)

        return movements # shape (frames - 1,)


    def extract_global_features(self, data, vitpose_features):
        """
        Extract global features from the data.
        data: shape (frames, kps, 2)
        vitpose_features: list of vitpose features, each feature is a list of coordinates
        Returns a dictionary of global features.
        """

        features = {}

        frames, kps, dims = data.shape
        if frames < 3:
            return features

        total_movement = self.calculate_total_movement(data) # shape (frames - 1,)

        if len(total_movement) > 0:
            features["global_movement_mean"] = np.mean(total_movement)
            features["global_movement_std"] = np.std(total_movement)
            features["global_movement_skew"] = skew(total_movement)
            features["global_movement_kurtosis"] = kurtosis(total_movement)

            movement_threshold = np.mean(total_movement) + np.std(total_movement)
            active_frames = np.sum(total_movement > movement_threshold)
            features["movement_activity_ratio"] = active_frames / len(total_movement)
        
        return features  # returns a dictionary of global features
             

    def extract_features_from_file(self, npz_file):
        """
        Extract features from a single .npz file.
        npz_file: Path to the .npz file
        Returns a dictionary of features or None if the file is invalid.
        """

        data, vitpose_features = self.load_data(npz_file)
        if data is None:
            return None

        frames = data.shape[0]
        if frames < 5:
            return None
        
        features = {}

        # multi-scale features
        multi_scale_features = self.extract_multi_scale_features(data, vitpose_features)
        features.update(multi_scale_features)

        # global features
        global_features = self.extract_global_features(data, vitpose_features)
        features.update(global_features)


        gc.collect()

        return features


    def process_batch(self, file_batch):
        """
        Process a batch of files to extract features.
        file_batch: list of tuples (npz_file, label_id, label)
        Returns a list of dictionaries with extracted features.
        """

        results = []
        for npz_file, label_id, label in file_batch:
            feats = self.extract_features_from_file(npz_file)
            if feats is not None:
                feats["filename"] = npz_file.stem
                feats["target"] = label_id
                feats["label"] = label
                results.append(feats)
        return results


    def create_csv(self, split):
        split_path = self.data_path / split
        if not split_path.exists():
            print(f"Split path {split_path} does not exist. Skipping CSV creation.")
            return
        
        all_files = []
        label_mapping = {"tourette": 2, "both": 1, "ftlb": 0}

        for label, label_id in label_mapping.items():
            label_path = split_path / label
            if not label_path.exists():
                print(f"Label path {label_path} does not exist. Skipping.")
                continue
            
            npz_files = list(label_path.glob("*.npz"))
            for npz_file in npz_files:
                all_files.append((npz_file, label_id, label))

        if not all_files:
            print(f"No files found in {split_path}. Skipping CSV creation.")
            return

        print(f"Processing {len(all_files)} files for {split} split...")

        batch_size = max(1, len(all_files) // self.n_workers)
        file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]

        try:
            with Pool(processes=self.n_workers) as pool:
                batch_results = pool.map(self.process_batch, file_batches)
        except Exception as e:
            print(f"Error during multiprocessing: {e}")
            print("Falling back to sequential processing.")
            batch_results = [self.process_batch(batch) for batch in file_batches]

        
        rows = []
        for batch_result in batch_results:
            rows.extend(batch_result) # flattening 


        if not rows:
            print(f"No valid features extracted for {split} split.")
            return

        df = pd.DataFrame(rows).fillna(0)
        csv_out = f"{split}_features.csv"
        df.to_csv(csv_out, index=False)
        print(f"CSV for {split} split created: {csv_out}")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {len(df.columns) - 3}")
        print(f"  Output: {csv_out}")
        print(f"  Class distribution: {df['label'].value_counts().to_dict()}")

        feature_types = {}
        for col in df.columns:
            if col not in ["filename", "target", "label"]:
                if 'ws' in col:
                    if 'body_' in col:
                        feature_types['vitpose_body'] = feature_types.get('vitpose_body', 0) + 1
                    else:
                        feature_types['temporal_windows'] = feature_types.get('temporal_windows', 0) + 1
                    
                elif 'corr_' in col:
                    feature_types['cross_region'] = feature_types.get('cross_region', 0) + 1
                elif 'global_' in col:
                    feature_types['global'] = feature_types.get('global', 0) + 1
                else:
                    feature_types['other'] = feature_types.get('other', 0) + 1

        print(f"Feature types: {feature_types}")

    
    def extract_features(self):

        print("Starting feature extraction...")
        print("=" * 60)
        print(f"Workers: {self.n_workers}")
        print(f"Window sizes: {self.window_sizes}")
        
        start_time = time.time()

        self.create_csv("train")
        self.create_csv("test")

        total_time = time.time() - start_time
        print(f"Total feature extraction time: {total_time:.2f} seconds")
        print("\nFeature extraction completed.")



if __name__ == "__main__":

    DATA_PATH = "/user/jafar.bakhshaliyev/u11836/.project/dir.project/bakhshaliyev/DigiStruc/dataset" # change it to the path of preprocessed data

    extractor = FeatureExtractor(DATA_PATH, MEDIAPIPE_PTS, n_workers= N_WORKERS)
    extractor.extract_features()
    print("Feature extraction completed.")