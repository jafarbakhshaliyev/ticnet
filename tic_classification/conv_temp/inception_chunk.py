import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
from collections import defaultdict
import torch.nn.functional as F
import json
from torch.cuda.amp import GradScaler, autocast
warnings.filterwarnings('ignore')

# CONFIG: 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# directory
os.makedirs('./analysis/temporal_inception/woatt1500', exist_ok=True)

DATA_DIR = "./dataset"
CHUNK_FRAMES = 1500
MAX_KEYPOINTS = 601
BATCH_SIZE = 64
EPOCHS = 15
N_FILTERS = 32
DEPTH = 4
N_RUNS = 5




VITPOSE_KEYPOINT_NAMES = [
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



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# attention mechanism (not used -> expensive)
class TemporalAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.last_attention = None  # Initialize this
        
    def forward(self, x):
        b, c, t = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = (avg_out + max_out).view(b, c, 1)
        
        # Store attention weights for analysis
        self.last_attention = attention.detach().cpu()
        
        return x * attention.expand_as(x)


# temporal inception block
class TemporalInceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=64, kernel_sizes=[3, 9, 19, 39], 
                 bottleneck_size=32, use_bottleneck=True, use_attention=True):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.use_attention = use_attention
        self.bottleneck_size = bottleneck_size
        self.n_filters = n_filters
        
        # bottlenec layer
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, 
                bottleneck_size, 
                kernel_size=1,
                bias=False
            )
        
        # multi-scale temporal convolutions
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    bottleneck_size if use_bottleneck else in_channels,
                    n_filters,
                    kernel_size=k,
                    padding=k//2,
                    bias=False
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        
        # max pooling
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True)
        )
        
        # average pooling
        self.avgpool_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True)
        )
        
        total_filters = n_filters * (len(kernel_sizes) + 2)  
        
        # attention (not used)
        if self.use_attention:
            self.attention = TemporalAttention(total_filters)
        
        self.final_bn = nn.BatchNorm1d(total_filters)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_bottleneck:
            bottleneck = self.bottleneck(x)
        else:
            bottleneck = x
        
        branches = []
        for conv in self.conv_branches:
            branches.append(conv(bottleneck))
        
        branches.append(self.maxpool_branch(x))
        branches.append(self.avgpool_branch(x))
        concat = torch.cat(branches, dim=1)
        
        # attention (not used)
        if self.use_attention:
            concat = self.attention(concat)
        
        output = self.final_bn(concat)
        return self.final_relu(output)

# temporal-convolution architecture
class TemporalInception(nn.Module):
    def __init__(self, num_keypoints, num_classes=3, n_filters=64, depth=8, 
                use_residual=True, use_bottleneck=True, use_attention=True,
                dropout_rate=0.3, use_keypoint_masking=True):
        super().__init__()
        
        self.depth = depth
        self.use_residual = use_residual
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.use_keypoint_masking = use_keypoint_masking 
        
        # input channels: keypoints * 2 - x and y coordinates
        self.input_channels = self.num_keypoints * 2
        
        print(f"Model initialized with {self.input_channels} input channels from {num_keypoints} keypoints")
        
        # keypoint embedding layer
        if use_keypoint_masking:
            self.keypoint_embedding = nn.Parameter(torch.randn(num_keypoints, 32))
            self.mask_token = nn.Parameter(torch.randn(32))
        
        # initial convolution to expand feature space
        self.initial_conv = nn.Sequential(
            nn.Conv1d(self.input_channels, n_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True)
        )
        
        self.inception_blocks = nn.ModuleList()
        current_channels = n_filters
        
        for i in range(depth):
            layer_filters = n_filters  
            if i < depth//2:
                kernel_sizes = [3, 9, 19, 39]
            else:
                kernel_sizes = [3, 9, 19]
            
            self.inception_blocks.append(
                TemporalInceptionBlock(
                    in_channels=current_channels,
                    n_filters=layer_filters,
                    kernel_sizes=kernel_sizes,
                    use_bottleneck=use_bottleneck,
                    use_attention=use_attention
                )
            )
            
            num_conv_branches = len(kernel_sizes)
            num_pooling_branches = 2  # max pool + avg pool
            total_branches = num_conv_branches + num_pooling_branches
            current_channels = layer_filters * total_branches
        

        if use_residual:
            self.residual_blocks = nn.ModuleList()
            res_input_channels = n_filters
            
            for i in range(depth // 2):
                layer_idx = (i + 1) * 2 - 1  # which inception block this connects to
                if layer_idx < depth//2:
                    target_channels = n_filters * 6  # 4 conv + 2 pool
                else:
                    target_channels = n_filters * 5  # 3 conv + 2 pool
                    
                self.residual_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(res_input_channels, target_channels, 1, bias=False),
                        nn.BatchNorm1d(target_channels)
                    )
                )
                res_input_channels = target_channels
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(current_channels * 2, current_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(current_channels // 2, num_classes)
        )
        
        self.final_features = None
        self.inception_blocks[-1].register_forward_hook(self._feature_hook)

    def _feature_hook(self, module, input, output):
        self.final_features = output.detach()

    def apply_keypoint_masking(self, x, keypoint_mask=None):
        """
        Apply masking to invalid keypoints
        x: (batch, keypoints*2, frames)
        keypoint_mask: (batch, keypoints)
        """
        if keypoint_mask is None or not self.use_keypoint_masking:
            return x

        batch_size, channels, frames = x.shape
        device = x.device 

        # reshape to (batch, keypoints, 2, frames)
        x_reshaped = x.view(batch_size, self.num_keypoints, 2, frames)

        #  cover both x and y coordinates
        keypoint_mask = keypoint_mask.float().to(device)  
        mask_expanded = keypoint_mask.unsqueeze(2).unsqueeze(3)  # (batch, keypoints, 1, 1)
        mask_expanded = mask_expanded.expand(-1, -1, 2, frames).float().to(device)  

        # apply mask
        x_masked = x_reshaped * mask_expanded

        # reshape to (batch, keypoints*2, frames)
        return x_masked.view(batch_size, channels, frames)

    def forward(self, x, keypoint_mask=None):
        batch_size, frames, keypoints, coords = x.size()
        
        assert coords == 2, f"Expected 2 coordinates (x, y), got {coords}"
        assert keypoints == self.num_keypoints, f"Expected {self.num_keypoints} keypoints, got {keypoints}"
        
        x = x.permute(0, 2, 3, 1)   # (batch, keypoints, 2, frames)
        x = x.reshape(batch_size, self.input_channels, frames)  # (batch, keypoints*2, frames)
        
        # apply keypoint masking
        x = self.apply_keypoint_masking(x, keypoint_mask)
        
        # initial convolution
        x = self.initial_conv(x)
        
        residual_input = x
        res_idx = 0
        
        # apply inception blocks with residual connections
        for i in range(self.depth):
            x = self.inception_blocks[i](x)
            
            # add residual connection every 2 layers
            if self.use_residual and i % 2 == 1 and res_idx < len(self.residual_blocks):
                residual = self.residual_blocks[res_idx](residual_input)
                if residual.size() == x.size():
                    x = x + residual
                residual_input = x
                res_idx += 1
        
        # global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)
        
        # classification
        output = self.classifier(pooled_features)
        
        return output


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# chunked datasets for three-class keypoint
class ThreeClassKeypointDataset(Dataset):
    def __init__(self, data_path, chunk_frames=1500, is_train=True, 
                 vitpose_target_count=133, mediapipe_count=468):
        super().__init__()
        self.data_path = data_path
        self.chunk_frames = chunk_frames
        self.is_train = is_train
        self.vitpose_target_count = vitpose_target_count
        self.mediapipe_count = mediapipe_count
        self.total_keypoints = vitpose_target_count + mediapipe_count
        
        self.samples = []
        self.file_info = []
        self.class_counts = defaultdict(int)
        
        self.keypoint_stats = {
            'min_vitpose': float('inf'),
            'max_vitpose': 0,
            'vitpose_counts': [],
            'files_with_missing_kps': 0
        }
        
        split = "train" if is_train else "test"
        split_dir = os.path.join(data_path, split)
        
        class_mapping = {'ftlb': 0, 'both': 1, 'tourette': 2}


        
        for class_name, label in class_mapping.items():
            folder = os.path.join(split_dir, class_name)
            if not os.path.exists(folder):
                print(f"Warning: {folder} does not exist")
                continue
                
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".npz"):  
                    fpath = os.path.join(folder, fname)
                    try:
                        npz_data = np.load(fpath, allow_pickle=True)
                        data = npz_data['data']
                        
                        vitpose_features = npz_data.get('vitpose_features', None)
                        
                        frames = data.shape[0]
                        actual_keypoints = data.shape[1]
                        
                        vitpose_kp_count = actual_keypoints - self.mediapipe_count
                        
                        if vitpose_kp_count < 0:
                            print(f"Warning: {fname} has {actual_keypoints} keypoints, less than MediaPipe count ({self.mediapipe_count})")
                            continue
                        
                        if self.keypoint_stats['min_vitpose'] == float('inf'):
                            self.keypoint_stats['min_vitpose'] = vitpose_kp_count
                        self.keypoint_stats['min_vitpose'] = min(self.keypoint_stats['min_vitpose'], vitpose_kp_count)
                        self.keypoint_stats['max_vitpose'] = max(self.keypoint_stats['max_vitpose'], vitpose_kp_count)
                        self.keypoint_stats['vitpose_counts'].append(vitpose_kp_count)
                        
                        if vitpose_kp_count < self.vitpose_target_count:
                            self.keypoint_stats['files_with_missing_kps'] += 1
                        
                        chunk_count = math.ceil(frames / self.chunk_frames)
                        
                        file_id = len(self.file_info)
                        self.file_info.append((fpath, label, class_name, vitpose_kp_count, vitpose_features))
                        self.class_counts[class_name] += 1
                        
                        for chunk_idx in range(chunk_count):
                            self.samples.append((file_id, chunk_idx))
                            
                    except Exception as e:
                        print(f"Error loading {fname}: {e}")
                        continue

        if self.keypoint_stats['min_vitpose'] == float('inf'):
            self.keypoint_stats['min_vitpose'] = 0
            print("Warning: No valid files processed!")
        
        self.print_keypoint_statistics()
        self.print_class_distribution()
        
        print(f"[{split}] Loaded {len(self.file_info)} files, {len(self.samples)} chunks")
        print(f"Class distribution: {dict(self.class_counts)}")

    def print_keypoint_statistics(self):
        """Print comprehensive keypoint statistics"""
        print("\n" + "="*50)
        print("KEYPOINT STATISTICS")
        print("="*50)
        if self.keypoint_stats['min_vitpose'] != float('inf'):
            print(f"VitPose keypoints range: {self.keypoint_stats['min_vitpose']} - {self.keypoint_stats['max_vitpose']}")
            print(f"Target VitPose count: {self.vitpose_target_count}")
            print(f"Files with missing keypoints: {self.keypoint_stats['files_with_missing_kps']}/{len(self.file_info)}")
            
            if self.keypoint_stats['vitpose_counts']:
                vitpose_counts = np.array(self.keypoint_stats['vitpose_counts'])
                print(f"VitPose count distribution:")
                print(f"  Mean: {np.mean(vitpose_counts):.1f}")
                print(f"  Std: {np.std(vitpose_counts):.1f}")
                
                unique_counts, count_freq = np.unique(vitpose_counts, return_counts=True)
                for count, freq in zip(unique_counts, count_freq):
                    print(f"  {count} keypoints: {freq} files ({freq/len(vitpose_counts)*100:.1f}%)")

    def print_class_distribution(self):
        """Print class distribution for focal loss calculation"""
        print("\n" + "="*50)
        print("CLASS DISTRIBUTION")
        print("="*50)
        total_files = sum(self.class_counts.values())
        
        class_order = ['ftlb', 'both', 'tourette'] 
        for i, class_name in enumerate(class_order):
            count = self.class_counts[class_name]
            percentage = (count / total_files * 100) if total_files > 0 else 0
            print(f"Class {i} ({class_name}): {count} files ({percentage:.1f}%)")
        
        if total_files > 0:
            print(f"\nRecommended focal loss alpha values:")
            for i, class_name in enumerate(class_order):
                count = self.class_counts[class_name]
                # Inverse frequency weighting
                alpha = total_files / (len(class_order) * count) if count > 0 else 1.0
                print(f"  Class {i} ({class_name}): {alpha:.3f}")

    def pad_vitpose_keypoints(self, data, actual_vitpose_count, vitpose_features=None):
        """
        pad VitPose keypoints
        """
        if actual_vitpose_count >= self.vitpose_target_count:
            return data[:, :self.vitpose_target_count + self.mediapipe_count, :]
        
        
        vitpose_data = data[:, :actual_vitpose_count, :]  # (frames, actual_vitpose, 2)
        mediapipe_data = data[:, actual_vitpose_count:, :]  # (frames, 468, 2)


        expected_mediapipe_size = self.mediapipe_count
        if mediapipe_data.shape[1] != expected_mediapipe_size:
            print(f"Warning: Expected {expected_mediapipe_size} MediaPipe keypoints, got {mediapipe_data.shape[1]}")
            if mediapipe_data.shape[1] < expected_mediapipe_size:
                mediapipe_pad = expected_mediapipe_size - mediapipe_data.shape[1]
                mediapipe_padding = np.zeros((data.shape[0], mediapipe_pad, 2))
                mediapipe_data = np.concatenate([mediapipe_data, mediapipe_padding], axis=1)
            else:
                mediapipe_data = mediapipe_data[:, :expected_mediapipe_size, :]
        
        pad_count = self.vitpose_target_count - actual_vitpose_count
    
        frames = data.shape[0]
        padding_shape = (frames, pad_count, 2)
        
        # zero padding 
        vitpose_padding = np.zeros(padding_shape)

        reconstructed_data = np.concatenate([
            vitpose_data,      
            vitpose_padding,   
            mediapipe_data    
        ], axis=1)
        
        return reconstructed_data

    def create_keypoint_mask(self, actual_vitpose_count):
        """
        Create a mask indicating which keypoints are real vs padded
        """
        mask = np.ones(self.total_keypoints, dtype=bool)
        if actual_vitpose_count < self.vitpose_target_count:
            mask[actual_vitpose_count:self.vitpose_target_count] = False
        
        return mask

    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        total_samples = sum(self.class_counts.values())
        weights = []
        class_order = ['ftlb', 'both', 'tourette']  
        
        for class_name in class_order:
            if self.class_counts[class_name] > 0:
                weight = total_samples / (3 * self.class_counts[class_name])
                weights.append(weight)
            else:
                weights.append(1.0)
        return torch.FloatTensor(weights)

    def get_focal_loss_alpha(self):
        """Calculate alpha values for focal loss"""
        total_samples = sum(self.class_counts.values())
        alpha_values = []
        class_order = ['ftlb', 'both', 'tourette']  
        
        for class_name in class_order:
            if self.class_counts[class_name] > 0:
                alpha = total_samples / (len(class_order) * self.class_counts[class_name])
                alpha = min(alpha, 10.0) 
                alpha_values.append(alpha)
            else:
                alpha_values.append(1.0)
        return torch.FloatTensor(alpha_values)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_id, chunk_idx = self.samples[idx]
        fpath, label, class_name, actual_vitpose_count, vitpose_features = self.file_info[file_id]
        
        npz_data = np.load(fpath, allow_pickle=True)
        data = npz_data['data']
        
        data = data[..., :2]
        data = np.nan_to_num(data, nan=0.0)
        start_frame = chunk_idx * self.chunk_frames
        end_frame = min((chunk_idx + 1) * self.chunk_frames, data.shape[0])
        chunk = data[start_frame:end_frame]
        
        if len(chunk) < self.chunk_frames:
            pad_frames = self.chunk_frames - len(chunk)
            chunk = np.pad(chunk, ((0, pad_frames), (0, 0), (0, 0)), mode='constant')

        chunk = self.pad_vitpose_keypoints(chunk, actual_vitpose_count, vitpose_features)
        

        keypoint_mask = self.create_keypoint_mask(actual_vitpose_count)
        
        expected_shape = (self.chunk_frames, self.total_keypoints, 2)
        if chunk.shape != expected_shape:
            print(f"Shape mismatch for {fpath}: got {chunk.shape}, expected {expected_shape}")
            if chunk.shape[1] < self.total_keypoints:
                pad_kps = self.total_keypoints - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_kps), (0, 0)), mode='constant')
                keypoint_mask = np.pad(keypoint_mask, (0, pad_kps), mode='constant', constant_values=False)
            elif chunk.shape[1] > self.total_keypoints:
                chunk = chunk[:, :self.total_keypoints, :]
                keypoint_mask = keypoint_mask[:self.total_keypoints]
                
        return (torch.FloatTensor(chunk), # (frames, keypoints, 2)
                torch.LongTensor([label])[0],
                torch.LongTensor([file_id])[0],
                torch.BoolTensor(keypoint_mask))


# data augmentation
class PoseDataAugmentation:
    """
    Data augmentation techniques specifically designed for pose keypoint sequences
    """
    
    def __init__(self, rotation_range=0.1, scale_range=0.1, translation_range=0.05,
                 noise_std=0.01, temporal_mask_prob=0.1, spatial_dropout_prob=0.05):
        self.rotation_range = rotation_range
        self.scale_range = scale_range  
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.temporal_mask_prob = temporal_mask_prob
        self.spatial_dropout_prob = spatial_dropout_prob
    
    def random_rotation(self, pose_seq):
        """Apply random rotation to pose sequence"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                   [sin_angle, cos_angle]])
        
        # Apply rotation to each frame
        rotated_seq = pose_seq.copy()
        for frame_idx in range(pose_seq.shape[0]):
            for kp_idx in range(pose_seq.shape[1]):
                point = pose_seq[frame_idx, kp_idx]
                rotated_seq[frame_idx, kp_idx] = rotation_matrix @ point
        
        return rotated_seq
    
    def random_scale(self, pose_seq):
        """Apply random scaling"""
        scale_factor = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        return pose_seq * scale_factor
    
    def random_translation(self, pose_seq):
        """Apply random translation"""
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)
        translation = np.array([tx, ty])
        
        return pose_seq + translation
    
    def add_noise(self, pose_seq):
        """Add Gaussian noise"""
        noise = np.random.normal(0, self.noise_std, pose_seq.shape)
        return pose_seq + noise
    
    def temporal_masking(self, pose_seq):
        """Randomly mask some temporal frames"""
        masked_seq = pose_seq.copy()
        num_frames = pose_seq.shape[0]
        
        mask_length = int(num_frames * self.temporal_mask_prob)
        if mask_length > 0:
            start_idx = np.random.randint(0, num_frames - mask_length + 1)
            masked_seq[start_idx:start_idx + mask_length] = 0
        
        return masked_seq
    
    def spatial_dropout(self, pose_seq, keypoint_mask=None):
        """Randomly dropout some keypoints"""
        augmented_seq = pose_seq.copy()
        num_keypoints = pose_seq.shape[1]

        if keypoint_mask is not None:
            valid_keypoints = np.where(keypoint_mask)[0]
            num_to_dropout = int(len(valid_keypoints) * self.spatial_dropout_prob)
            
            if num_to_dropout > 0:
                dropout_indices = np.random.choice(valid_keypoints, num_to_dropout, replace=False)
                augmented_seq[:, dropout_indices] = 0
        else:
            num_to_dropout = int(num_keypoints * self.spatial_dropout_prob)
            if num_to_dropout > 0:
                dropout_indices = np.random.choice(num_keypoints, num_to_dropout, replace=False)
                augmented_seq[:, dropout_indices] = 0
        
        return augmented_seq
    
    def apply_augmentation(self, pose_seq, keypoint_mask=None, augment_prob=0.5):
        """Apply random combination of augmentations"""
        if np.random.random() > augment_prob:
            return pose_seq
        
        augmented = pose_seq.copy()
        
        if np.random.random() < 0.3:
            augmented = self.random_rotation(augmented)
        
        if np.random.random() < 0.3:
            augmented = self.random_scale(augmented)
        
        if np.random.random() < 0.3:
            augmented = self.random_translation(augmented)
        
        if np.random.random() < 0.2:
            augmented = self.add_noise(augmented)
        
        if np.random.random() < 0.15:
            augmented = self.temporal_masking(augmented)
        
        if np.random.random() < 0.1:
            augmented = self.spatial_dropout(augmented, keypoint_mask)
        
        return augmented

class AugmentedThreeClassKeypointDataset(ThreeClassKeypointDataset):
    def __init__(self, data_path, chunk_frames=1500, is_train=True,
                 vitpose_target_count=133, mediapipe_count=468,
                 use_augmentation=True, augment_prob=0.5):
        super().__init__(data_path, chunk_frames, is_train, vitpose_target_count, mediapipe_count)
        
        self.use_augmentation = use_augmentation and is_train 
        self.augmenter = PoseDataAugmentation() if self.use_augmentation else None
        self.augment_prob = augment_prob
    
    def __getitem__(self, idx):
        chunk, label, file_id, keypoint_mask = super().__getitem__(idx)

        if self.use_augmentation and self.augmenter:
            if label == 0:  
                augment_prob = self.augment_prob * 1.5
            elif label == 1:  
                augment_prob = self.augment_prob * 1.3
            else:  
                augment_prob = self.augment_prob * 0.8

            chunk_np = chunk.numpy()
            keypoint_mask_np = keypoint_mask.numpy()
            
            augmented_chunk = self.augmenter.apply_augmentation(
                chunk_np, keypoint_mask_np, augment_prob
            )
            
            chunk = torch.FloatTensor(augmented_chunk)
        
        return chunk, label, file_id, keypoint_mask     

# train
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in loader:
        if len(batch) == 4:  
            data, target, _, keypoint_mask = batch
            keypoint_mask = keypoint_mask.to(device)
        else:  
            data, target, _ = batch
            keypoint_mask = None
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data, keypoint_mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(data)
        
        # calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()
        
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy

# evaluate
def evaluate_model(model, loader, device):
    model.eval()
    file_predictions = defaultdict(list)
    file_labels = {}
    file_features = defaultdict(list)
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:  
                data, target, file_id, keypoint_mask = batch
                keypoint_mask = keypoint_mask.to(device)
            else:  
                data, target, file_id = batch
                keypoint_mask = None
            
            data = data.to(device)
            output = model(data, keypoint_mask)
            probs = torch.softmax(output, dim=1)
            
            for i in range(len(data)):
                fid = file_id[i].item()
                file_labels[fid] = target[i].item()
                file_predictions[fid].append(probs[i].cpu().numpy())
                
                # extract features
                if model.final_features is not None:
                    features = torch.mean(model.final_features[i], dim=-1).cpu().numpy()
                    file_features[fid].append(features)
    
    final_preds = []
    final_labels = []
    final_features = []
    
    for fid in file_predictions:
        avg_prob = np.mean(file_predictions[fid], axis=0)
        pred = np.argmax(avg_prob)
        final_preds.append(pred)
        final_labels.append(file_labels[fid])
        
        if fid in file_features and len(file_features[fid]) > 0:
            avg_features = np.mean(file_features[fid], axis=0)
            final_features.append(avg_features)
        else:
            final_features.append(np.zeros(512))  
        
    final_preds = np.array(final_preds)
    final_labels = np.array(final_labels)
    final_features = np.array(final_features)
    
    accuracy = accuracy_score(final_labels, final_preds)
    balanced_accuracy = balanced_accuracy_score(final_labels, final_preds)
    f1 = f1_score(final_labels, final_preds, average='weighted')
    cm = confusion_matrix(final_labels, final_preds)
    
    return accuracy, balanced_accuracy, f1, cm, final_features, file_predictions

# keypoint importance analysis
def analyze_keypoint_importance(model, dataset, device, vitpose_feature_names=None):
    """
    Analyze importance of different keypoint groups using gradient-based attribution
    """
    model.eval()
    
    sample_indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
    
    keypoint_gradients = []
    
    for idx in sample_indices:
        data, label, _, keypoint_mask = dataset[idx]
        data = data.unsqueeze(0).to(device)
        data.requires_grad_(True)
        
        output = model(data, keypoint_mask.unsqueeze(0).to(device) if keypoint_mask is not None else None)
        
        # gradient w.r.t. input for the predicted class
        pred_class = torch.argmax(output, dim=1)
        class_score = output[0, pred_class]
        class_score.backward()
        
        # Calculate gradient magnitude for each keypoint
        grad = data.grad.abs()  # (1, frames, keypoints, 2)
        keypoint_grad = torch.mean(grad, dim=(0, 1, 3)).cpu().numpy()  
        keypoint_gradients.append(keypoint_grad)
        
        data.grad.zero_()
    
    # average gradients across samples
    avg_gradients = np.mean(keypoint_gradients, axis=0)  # (keypoints,)
    
    return avg_gradients

def analyze_body_parts(feature_importance, features, output_dir):
    """
    Analyze importance of different body parts based on keypoint groupings
    """
    try:
        # body part groupings based on VitPose 
        body_part_groups = {
            'facial_features': list(range(0, 5)),  # nose, eyes, ears
            'torso': list(range(5, 13)),           # shoulders, elbows, wrists, hips
            'legs': list(range(13, 17)),           # knees, ankles
            'feet': list(range(17, 23)),           # toes, heels
            'face_details': list(range(23, 68)),   # detailed face keypoints
            'hands': list(range(68, min(133, len(feature_importance))))  # hand keypoints
        }
        
        # importance by body part
        body_part_importance = {}
        for part, indices in body_part_groups.items():
            valid_indices = [i for i in indices if i < len(feature_importance)]
            if valid_indices:
                body_part_importance[part] = np.mean([feature_importance[i] for i in valid_indices])
            else:
                body_part_importance[part] = 0.0
        
        plt.figure(figsize=(12, 8))
        parts = list(body_part_importance.keys())
        values = list(body_part_importance.values())
        
        bars = plt.bar(parts, values, alpha=0.7, color='skyblue')
        plt.title('Feature Importance by Body Part')
        plt.xlabel('Body Part')
        plt.ylabel('Average Importance Score')
        plt.xticks(rotation=45)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/body_part_importance.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return body_part_importance
        
    except Exception as e:
        print(f"Error in analyze_body_parts: {e}")
        return {}

def analyze_feature_interactions(model, sample_data, selected_features, output_dir):
    """
    Analyze interactions between different features using attention weights
    """
    try:
        model.eval()
        with torch.no_grad():
            sample_batch = sample_data[:min(32, len(sample_data))]
            
            _ = model(sample_batch)
            
            attention_weights = []
            for module in model.modules():
                if isinstance(module, TemporalAttention):
                    if hasattr(module, 'last_attention'):
                        attention_weights.append(module.last_attention)
            
            if attention_weights:
                avg_attention = torch.mean(torch.stack(attention_weights), dim=0)
                avg_attention = torch.mean(avg_attention, dim=0).cpu().numpy()
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(avg_attention[:50, :50], cmap='viridis', cbar=True)
                plt.title('Feature Interaction Matrix (Top 50 Features)')
                plt.xlabel('Feature Index')
                plt.ylabel('Feature Index')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/feature_interactions.pdf', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("No attention weights found for interaction analysis")
                
    except Exception as e:
        print(f"Error in analyze_feature_interactions: {e}")

def analyze_misclassifications(predictions, true_labels, features, output_dir):
    """
    Analyze patterns in misclassified samples
    """
    try:
        misclassified = np.where(predictions != true_labels)[0]
        
        if len(misclassified) == 0:
            print("No misclassifications found")
            return
        
        class_names = ['FTLB', 'Both', 'Tourette']
        confusion_pairs = defaultdict(int)
        
        for idx in misclassified:
            true_class = true_labels[idx]
            pred_class = predictions[idx]
            confusion_pairs[(true_class, pred_class)] += 1
        
        plt.figure(figsize=(10, 6))
        pairs = list(confusion_pairs.keys())
        counts = list(confusion_pairs.values())
        
        labels = [f"{class_names[true]} → {class_names[pred]}" for true, pred in pairs]
        plt.bar(labels, counts, alpha=0.7, color='coral')
        plt.title('Misclassification Patterns')
        plt.xlabel('True → Predicted Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/misclassification_patterns.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analyzed {len(misclassified)} misclassifications")
        
    except Exception as e:
        print(f"Error in analyze_misclassifications: {e}")


def safe_plot_wrapper(plot_func):
    def wrapper(*args, **kwargs):
        try:
            return plot_func(*args, **kwargs)
        except Exception as e:
            print(f"Warning: Plot function {plot_func.__name__} failed: {e}")
            return None
    return wrapper


def create_comprehensive_visualizations(feature_importance, sample_data, selected_features, output_dir):
    """
    Create comprehensive visualizations for model analysis
    """
    try:
        top_indices = np.argsort(feature_importance)[-20:]
        
        plt.figure(figsize=(15, 8))
        plt.barh(range(20), feature_importance[top_indices])
        
        if selected_features is not None and len(selected_features) >= 20:
            feature_names = [f"Feature_{i}" for i in top_indices]
        else:
            feature_names = [f"Keypoint_{i}" for i in top_indices]
        
        plt.yticks(range(20), feature_names)
        plt.xlabel('Importance Score')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_features_importance.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(feature_importance, bins=50, alpha=0.7, color='lightblue')
        plt.xlabel('Importance Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Feature Importance Scores')
        plt.axvline(np.mean(feature_importance), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(feature_importance):.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/importance_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created comprehensive visualizations")
        
    except Exception as e:
        print(f"Error in create_comprehensive_visualizations: {e}")

analyze_body_parts = safe_plot_wrapper(analyze_body_parts)
analyze_feature_interactions = safe_plot_wrapper(analyze_feature_interactions)
analyze_misclassifications = safe_plot_wrapper(analyze_misclassifications)
create_comprehensive_visualizations = safe_plot_wrapper(create_comprehensive_visualizations)

def create_feature_importance_report(model, train_dataset, test_dataset, device, 
                                   vitpose_feature_names, run_index, output_dir):
    """
    Create comprehensive feature importance analysis
    """
    try:
        train_importance = analyze_keypoint_importance(model, train_dataset, device, vitpose_feature_names)
        test_importance = analyze_keypoint_importance(model, test_dataset, device, vitpose_feature_names)
        
        sample_data, sample_labels, _, sample_masks = next(iter(DataLoader(test_dataset, batch_size=32, shuffle=False)))
        sample_data = sample_data.to(device)

        body_part_results = analyze_body_parts(train_importance, train_importance, output_dir)
        
        analyze_feature_interactions(model, sample_data, None, output_dir)
        
        model.eval()
        with torch.no_grad():
            sample_outputs = model(sample_data, sample_masks.to(device))
            sample_preds = torch.argmax(sample_outputs, dim=1).cpu().numpy()
        
        analyze_misclassifications(sample_preds, sample_labels.numpy(), train_importance, output_dir)
        create_comprehensive_visualizations(train_importance, sample_data, None, output_dir)
        n_keypoints = len(train_importance)
        if vitpose_feature_names and len(vitpose_feature_names) >= n_keypoints * 2:
            kp_names = [name.replace('vitpose_', '').replace('_x', '') for name in vitpose_feature_names[::2]][:n_keypoints]
        else:
            kp_names = VITPOSE_KEYPOINT_NAMES[:n_keypoints] if n_keypoints <= len(VITPOSE_KEYPOINT_NAMES) else [f"keypoint_{i}" for i in range(n_keypoints)]
        
        body_part_groups = {
            'facial_features': list(range(0, min(5, n_keypoints))),
            'torso': list(range(5, min(13, n_keypoints))),
            'legs': list(range(13, min(17, n_keypoints))),
            'feet': list(range(17, min(23, n_keypoints))),
            'face_details': list(range(23, min(68, n_keypoints))),
            'hands': list(range(68, min(133, n_keypoints)))
        }
        
        body_part_importance = {}
        for part, indices in body_part_groups.items():
            if indices and max(indices) < len(train_importance):
                body_part_importance[part] = {
                    'train': float(np.mean([train_importance[i] for i in indices])),  
                    'test': float(np.mean([test_importance[i] for i in indices]))   
                }
        
        results = {
            'run': int(run_index),  
            'keypoint_importance': {
                'train': [float(x) for x in train_importance.tolist()], 
                'test': [float(x) for x in test_importance.tolist()],    
                'keypoint_names': kp_names
            },
            'body_part_importance': body_part_importance
        }
        
        with open(f'{output_dir}/feature_importance_run_{run_index}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        if run_index == 1:  
            create_summary_visualizations(output_dir)
        
        return results
        
    except Exception as e:
        print(f"Error in create_feature_importance_report: {e}")
        import traceback
        traceback.print_exc()  
        return None


def create_summary_visualizations(output_dir):
    """
    Create summary visualizations from all runs
    """
    try:
        all_results = []
        for run in range(2):
            try:
                with open(f'{output_dir}/feature_importance_run_{run}.json', 'r') as f:
                    all_results.append(json.load(f))
            except:
                continue
        
        if not all_results:
            print("No results found for summary visualizations")
            return
        
    
        all_train_importance = [r['keypoint_importance']['train'] for r in all_results]
        all_test_importance = [r['keypoint_importance']['test'] for r in all_results]
        
        avg_train_importance = np.mean(all_train_importance, axis=0)
        avg_test_importance = np.mean(all_test_importance, axis=0)
        
        keypoint_names = all_results[0]['keypoint_importance']['keypoint_names']
        
        top_indices = np.argsort(avg_train_importance)[-20:]
        
        plt.figure(figsize=(15, 8))
        plt.barh(range(20), avg_train_importance[top_indices])
        plt.yticks(range(20), [keypoint_names[i] if i < len(keypoint_names) else f"kp_{i}" for i in top_indices])
        plt.xlabel('Importance Score')
        plt.title('Top 20 Most Important Keypoints (Training Data)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_keypoints_importance.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        body_parts = ['facial_features', 'torso', 'legs', 'feet', 'face_details', 'hands']
        train_scores = []
        test_scores = []
        
        for part in body_parts:
            part_scores_train = [r['body_part_importance'].get(part, {}).get('train', 0) for r in all_results]
            part_scores_test = [r['body_part_importance'].get(part, {}).get('test', 0) for r in all_results]
            
            train_scores.append(np.mean(part_scores_train))
            test_scores.append(np.mean(part_scores_test))
        
        x = np.arange(len(body_parts))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, train_scores, width, label='Training', alpha=0.8)
        plt.bar(x + width/2, test_scores, width, label='Testing', alpha=0.8)
        
        plt.xlabel('Body Part')
        plt.ylabel('Average Importance Score')
        plt.title('Feature Importance by Body Part')
        plt.xticks(x, body_parts, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/body_part_importance.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in create_summary_visualizations: {e}")


def check_existing_model(run_index, output_dir):
    """Check if a trained model already exists for this run"""
    model_path = f'{output_dir}/best_model_run_{run_index}.pth'
    return os.path.exists(model_path)

def main():
    # params
    data_dir = DATA_DIR
    chunk_frames = CHUNK_FRAMES
    max_keypoints = MAX_KEYPOINTS
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    n_filters = N_FILTERS
    depth = DEPTH
    n_runs = N_RUNS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    results = {
        'train_bal_acc': [],
        'train_f1': [],
        'train_acc': [],
        'test_bal_acc': [],
        'test_f1': [],
        'test_acc': [],
        'test_cm': []
    }
    
    # feature importance 
    all_feature_results = []
    
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"Starting Run {run + 1}/{n_runs}")
        print(f"{'='*50}")
        
        model_exists = check_existing_model(run, './analysis/temporal_inception/woatt1500')

        train_dataset = AugmentedThreeClassKeypointDataset(
            data_dir, chunk_frames=chunk_frames, is_train=True,
            use_augmentation=True, augment_prob=0.5
        )
        test_dataset = ThreeClassKeypointDataset(
            data_dir, chunk_frames=chunk_frames, is_train=False
        )
        print("Dataset as chunks with augmentations creation successful!")

        sample_data, sample_label, sample_file_id, sample_mask = train_dataset[0]
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample mask valid keypoints: {sample_mask.sum().item()}/{len(sample_mask)}")

        # class weights
        class_weights = train_dataset.get_class_weights().to(device)
        print(f"Class weights: {class_weights.cpu()}")
        
        # data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # model without attention
        model = TemporalInception(
            num_keypoints=max_keypoints,
            num_classes=3,
            n_filters=n_filters,
            depth=depth,
            use_residual=True,
            use_bottleneck=True,
            use_attention=False, 
            dropout_rate=0.5
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if model_exists:
            print(f"✓ Found existing model for run {run + 1}, loading saved weights...")
            model.load_state_dict(torch.load(f'./analysis/temporal_inception/woatt1500/best_model_run_{run}.pth'))
            print("Model loaded successfully, skipping training phase.")
            
           
            alpha_values = train_dataset.get_focal_loss_alpha().to(device)
            criterion = FocalLoss(
                alpha=alpha_values,
                gamma=2.0,
                reduction='mean'
            )
            
        else:
            print(f"No existing model found for run {run + 1}, starting training...")
            

            alpha_values = train_dataset.get_focal_loss_alpha().to(device)
            print(f"Focal loss alpha values: {alpha_values.cpu()}")

            criterion = FocalLoss(
                alpha=alpha_values,
                gamma=2.0, 
                reduction='mean'
            )      

            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            
            best_test_bal_acc = 0
            patience = 10
            best_train_loss = float('inf')
            patience_counter = 0
            
            train_losses = []
            train_accs = []
            test_accs = []
            test_bal_accs = []
            
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-" * 30)
                
    
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                
       
                test_acc, test_bal_acc, test_f1, test_cm, test_features, _ = evaluate_model(
                    model, test_loader, device
                )
                
        
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
                print(f"Test Acc: {test_acc:.3f}, Test Bal Acc: {test_bal_acc:.3f}, Test F1: {test_f1:.3f}")
                print(f"Learning Rate: {current_lr:.2e}")
                
  
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_bal_accs.append(test_bal_acc)
                
         
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience_counter = 0
  
                    torch.save(model.state_dict(), f'./analysis/temporal_inception/woatt1500/best_model_run_{run}.pth')
                    print("✓ New best training loss - model saved!")
                else:
                    patience_counter += 1
                    print(f"No improvement in training loss (patience: {patience_counter}/{patience})")
                    
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best training loss: {best_train_loss:.4f}")
                    break
            
         
            model.load_state_dict(torch.load(f'./analysis/temporal_inception/woatt1500/best_model_run_{run}.pth'))
        

        train_acc, train_bal_acc, train_f1, train_cm, _, _ = evaluate_model(
            model, train_loader, device
        )
        test_acc, test_bal_acc, test_f1, test_cm, test_features, test_predictions = evaluate_model(
            model, test_loader, device
        )
        
        print(f"\nFinal Results for Run {run + 1}:")
        print(f"Train - Acc: {train_acc:.3f}, Bal Acc: {train_bal_acc:.3f}, F1: {train_f1:.3f}")
        print(f"Test - Acc: {test_acc:.3f}, Bal Acc: {test_bal_acc:.3f}, F1: {test_f1:.3f}")
        
   
        results['train_bal_acc'].append(train_bal_acc)
        results['train_f1'].append(train_f1)
        results['train_acc'].append(train_acc)
        results['test_bal_acc'].append(test_bal_acc)
        results['test_f1'].append(test_f1)
        results['test_acc'].append(test_acc)
        results['test_cm'].append(test_cm)
        
        try:
            sample_file_path = train_dataset.file_info[0][0]
            npz_data = np.load(sample_file_path, allow_pickle=True)
            vitpose_features = npz_data.get('vitpose_features', None)
            if vitpose_features is not None:
                vitpose_feature_names = vitpose_features.tolist()
            else:
                vitpose_feature_names = None
            
            feature_results = create_feature_importance_report(
                model, train_dataset, test_dataset, device,
                vitpose_feature_names, run, './analysis/temporal_inception/woatt1500'
            )
            if feature_results:
                all_feature_results.append(feature_results)
        except Exception as e:
            print(f"Warning: Feature importance analysis failed: {e}")
        
        if not model_exists and 'train_losses' in locals() and len(train_losses) > 0:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(train_losses)
            plt.title(f'Training Loss - Run {run + 1}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 3, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(test_accs, label='Test Accuracy')
            plt.title(f'Accuracy - Run {run + 1}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot(test_bal_accs)
            plt.title(f'Test Balanced Accuracy - Run {run + 1}')
            plt.xlabel('Epoch')
            plt.ylabel('Balanced Accuracy')
            
            plt.tight_layout()
            plt.savefig(f'./analysis/temporal_inception/woatt1500/training_curves_run_{run}.pdf', dpi=300)
            plt.close()
        else:
            print(f"Skipping training curves plot for run {run + 1} (model was loaded or no training history)")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['FTLB', 'Both', 'Tourette'],
                   yticklabels=['FTLB', 'Both', 'Tourette'])
        plt.title(f'Confusion Matrix - Run {run + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'./analysis/temporal_inception/woatt1500/confusion_matrix_run_{run}.pdf', dpi=300)
        plt.close()
    
    print("\n" + "="*60)
    print(f"FINAL AVERAGED RESULTS ({n_runs} Runs)")
    print("="*60)
    
    print(f"\nTraining Metrics:")
    print(f"Accuracy: {np.mean(results['train_acc']):.3f} ± {np.std(results['train_acc']):.3f}")
    print(f"Balanced Accuracy: {np.mean(results['train_bal_acc']):.3f} ± {np.std(results['train_bal_acc']):.3f}")
    print(f"F1 Score: {np.mean(results['train_f1']):.3f} ± {np.std(results['train_f1']):.3f}")
    
    print(f"\nTest Metrics:")
    print(f"Accuracy: {np.mean(results['test_acc']):.3f} ± {np.std(results['test_acc']):.3f}")
    print(f"Balanced Accuracy: {np.mean(results['test_bal_acc']):.3f} ± {np.std(results['test_bal_acc']):.3f}")
    print(f"F1 Score: {np.mean(results['test_f1']):.3f} ± {np.std(results['test_f1']):.3f}")
    
    avg_cm = np.mean(results['test_cm'], axis=0)
    std_cm = np.std(results['test_cm'], axis=0)
    
    print(f"\nAveraged Confusion Matrix:")
    print(avg_cm)
    print(f"\nConfusion Matrix Standard Deviation:")
    print(std_cm)
    
    save_comprehensive_results(results, avg_cm, std_cm, all_feature_results)
    create_final_visualizations(results, avg_cm, std_cm)
    
    print(f"\nAll results saved to: ./analysis/temporal_inception/woatt1500/")

def save_comprehensive_results(results, avg_cm, std_cm, feature_results):
    """Save comprehensive results to files"""
    
    with open('./analysis/temporal_inception/woatt1500/comprehensive_results.txt', 'w') as f:
        
        f.write("Averaged Results (5 runs):\n")
        f.write("-" * 30 + "\n")
        f.write("Training Metrics:\n")
        f.write(f"Accuracy: {np.mean(results['train_acc']):.3f} ± {np.std(results['train_acc']):.3f}\n")
        f.write(f"Balanced Accuracy: {np.mean(results['train_bal_acc']):.3f} ± {np.std(results['train_bal_acc']):.3f}\n")
        f.write(f"F1 Score: {np.mean(results['train_f1']):.3f} ± {np.std(results['train_f1']):.3f}\n\n")
        
        f.write("Test Metrics:\n")
        f.write(f"Accuracy: {np.mean(results['test_acc']):.3f} ± {np.std(results['test_acc']):.3f}\n")
        f.write(f"Balanced Accuracy: {np.mean(results['test_bal_acc']):.3f} ± {np.std(results['test_bal_acc']):.3f}\n")
        f.write(f"F1 Score: {np.mean(results['test_f1']):.3f} ± {np.std(results['test_f1']):.3f}\n\n")
        
        f.write("Averaged Confusion Matrix:\n")
        f.write("-" * 30 + "\n")
        f.write("       FTLB      Both   Tourette\n")
        for i, row_name in enumerate(['FTLB', 'Both', 'Tourette']):
            f.write(f"{row_name:8}")
            for j in range(3):
                f.write(f"{avg_cm[i, j]:8.2f}")
            f.write("\n")
        
        f.write(f"\nConfusion Matrix Standard Deviation:\n")
        f.write("       FTLB      Both   Tourette\n")
        for i, row_name in enumerate(['FTLB', 'Both', 'Tourette']):
            f.write(f"{row_name:8}")
            for j in range(3):
                f.write(f"{std_cm[i, j]:8.2f}")
            f.write("\n")
        
        f.write("\nPer-run Results:\n")
        f.write("-" * 20 + "\n")
        for run in range(len(results['test_acc'])):
            f.write(f"\nRun {run + 1}:\n")
            f.write(f"Train Acc: {results['train_acc'][run]:.3f}, ")
            f.write(f"Bal Acc: {results['train_bal_acc'][run]:.3f}, ")
            f.write(f"F1: {results['train_f1'][run]:.3f}\n")
            f.write(f"Test Acc: {results['test_acc'][run]:.3f}, ")
            f.write(f"Bal Acc: {results['test_bal_acc'][run]:.3f}, ")
            f.write(f"F1: {results['test_f1'][run]:.3f}\n")
            
            f.write("Confusion Matrix:\n")
            cm = results['test_cm'][run]
            for i in range(3):
                for j in range(3):
                    f.write(f"{cm[i, j]:6.0f}")
                f.write("\n")
    
    results_dict = {
        'train_metrics': {
            'accuracy': {'mean': float(np.mean(results['train_acc'])), 'std': float(np.std(results['train_acc']))},
            'balanced_accuracy': {'mean': float(np.mean(results['train_bal_acc'])), 'std': float(np.std(results['train_bal_acc']))},
            'f1_score': {'mean': float(np.mean(results['train_f1'])), 'std': float(np.std(results['train_f1']))}
        },
        'test_metrics': {
            'accuracy': {'mean': float(np.mean(results['test_acc'])), 'std': float(np.std(results['test_acc']))},
            'balanced_accuracy': {'mean': float(np.mean(results['test_bal_acc'])), 'std': float(np.std(results['test_bal_acc']))},
            'f1_score': {'mean': float(np.mean(results['test_f1'])), 'std': float(np.std(results['test_f1']))}
        },
        'confusion_matrix': {
            'mean': avg_cm.tolist(),
            'std': std_cm.tolist()
        },
        'per_run_results': {
            'train_acc': [float(x) for x in results['train_acc']],
            'train_bal_acc': [float(x) for x in results['train_bal_acc']],
            'train_f1': [float(x) for x in results['train_f1']],
            'test_acc': [float(x) for x in results['test_acc']],
            'test_bal_acc': [float(x) for x in results['test_bal_acc']],
            'test_f1': [float(x) for x in results['test_f1']]
        }
    }
    
    with open('./analysis/temporal_inception/woatt1500/results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

def create_final_visualizations(results, avg_cm, std_cm):
    """Create final summary visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = ['train_acc', 'train_bal_acc', 'train_f1', 'test_acc', 'test_bal_acc', 'test_f1']
    titles = ['Train Accuracy', 'Train Balanced Accuracy', 'Train F1 Score',
              'Test Accuracy', 'Test Balanced Accuracy', 'Test F1 Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = i // 3, i % 3
        data = results[metric]
        
        axes[row, col].boxplot(data)
        axes[row, col].scatter([1] * len(data), data, alpha=0.7, s=50)
        axes[row, col].set_title(title)
        axes[row, col].set_ylabel('Score')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(data)
        axes[row, col].axhline(y=mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.3f}')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('./analysis/temporal_inception/woatt1500/metrics_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(avg_cm, cmap='Blues')
    
    cbar = plt.colorbar(im)
    cbar.set_label('Average Count', rotation=270, labelpad=20)
    

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(['FTLB', 'Both', 'Tourette'])
    ax.set_yticklabels(['FTLB', 'Both', 'Tourette'])
    
    for i in range(3):
        for j in range(3):
            text = f'{avg_cm[i, j]:.1f}\n±{std_cm[i, j]:.1f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_title('Averaged Confusion Matrix (5 runs)\nMean ± Standard Deviation', fontsize=14)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('./analysis/temporal_inception/woatt1500/final_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics_names = ['Accuracy', 'Balanced Accuracy', 'F1 Score']
    train_means = [np.mean(results['train_acc']), np.mean(results['train_bal_acc']), np.mean(results['train_f1'])]
    train_stds = [np.std(results['train_acc']), np.std(results['train_bal_acc']), np.std(results['train_f1'])]
    test_means = [np.mean(results['test_acc']), np.mean(results['test_bal_acc']), np.mean(results['test_f1'])]
    test_stds = [np.std(results['test_acc']), np.std(results['test_bal_acc']), np.std(results['test_f1'])]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_means, width, yerr=train_stds, 
                   label='Training', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, test_means, width, yerr=test_stds, 
                   label='Testing', alpha=0.8, capsize=5)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Summary (5 runs average)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./analysis/temporal_inception/woatt1500/performance_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()