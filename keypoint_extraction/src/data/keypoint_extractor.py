import cv2
import numpy as np
import yaml
from pathlib import Path
import mediapipe as mp
from easy_ViTPose.easy_ViTPose.inference import VitInference
import os
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import wandb
import time
import logging

# Set CUDA environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ["WANDB_API_KEY"] = "2787eefc915d2d0c04877b2b33454e080621d5f5"


# Disable gradients since we're only doing inference
torch.set_grad_enabled(False)

logging.getLogger("ultralytics").setLevel(logging.WARNING)

class Preprocess:
    def __init__(self, config_path, args):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.args = args

        # Clear CUDA cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Download + init pose model
        filename = os.path.join(
            self.config["model"]["model_type"],
            f"{self.config['model']['dataset']}/vitpose-" + self.config["model"]["model_size"] +
            f"-{self.config['model']['dataset']}"
        ) + self.config["model"]["ext"]
        filename_yolo = 'yolov8/yolov8' + self.config['model']['yolo_size'] + self.config['model']['ext_yolo']

        print(f'Downloading model {self.config["model"]["repo_id"]}/{filename}')
        model_path = hf_hub_download(
            repo_id=self.config["model"]["repo_id"],
            filename=filename
        )
        yolo_path = hf_hub_download(
            repo_id=self.config["model"]["repo_id"],
            filename=filename_yolo
        )
      
        # Initialize VitPose on CPU first
        self.pose_model = VitInference(
            model_path, yolo_path,
            self.config["model"]["model_size"],
            dataset=self.config["model"]["dataset"],
            yolo_size=320,
            is_video=False,
            device='cuda'  # Initialize on CPU first
        )

        # Initialize face detection (YOLO or YOLOv8 specialized face model)
        self.face_model = YOLO('yolov10n-face.pt', verbose=False)

        # Person detection model for assistant filtering
        self.person_model = YOLO('yolov8n.pt', verbose=False)

        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_conf_threshold = 0.5  # Adjust this value

    def extract_frames_and_keypoints(self, video_path, output_dir):
        total_frames = 0
        frames_with_keypoints = 0
        frames_without_keypoints = 0
        frames_with_face_keypoints = 0
        frames_with_face_detected_but_no_landmarks = 0
        frames_without_face_detection = 0
        frames_with_missing_keypoints = []

        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        base_name = Path(video_path).stem

        if not video.isOpened():
            print(f"Error opening video file {video_path}")
            return
        
        count = 0
        cnt = 0
        keypoints_data = []

        while True:
            ret, frame = video.read()
            if not ret:
                print("No more frames or unable to read video.")
                break

            # ---------------------------------------IMPORTANT NOTIFICATION-----------------------------------------------------
            # This code can change from using heuristic method such as more than 1 person or face detection, it could be easily replaced
            # by tracking methods for a person then use the keypoints. For our use case, we are using a heuristic method to skip frames
            # with no or more than 1 person detected, and frames without face detection. This is to ensure that we only process frames
            # where we can reliably extract keypoints for a single person.
            # ------------------------------------------------------------------------------------------------------------------

            # Process every Nth frame
            if count % self.config['data']['frame_rate'] == 0:
                total_frames += 1

                # ----------------------------------------------------------
                # Person detection to skip frames with 0 or >1 persons
                # ----------------------------------------------------------
                person_results = self.person_model(frame)
                person_count = 0
                single_person_box = None

                if person_results and person_results[0].boxes:
                    for box in person_results[0].boxes:
                        # Class 0 corresponds to "person" in COCO
                        if int(box.cls[0]) == 0:
                            person_count += 1
                            single_person_box = box.xyxy[0]

                # If no persons or more than 1 person, skip
                if person_count == 0 or person_count > 1:
                    frames_without_keypoints += 1
                    num_keypoints = 133 + 468
                    keypoints_data.append(np.full((num_keypoints, 3), np.nan))
                    frames_with_missing_keypoints.append(total_frames - 1)

                    if self.config['data']['save_frames']:
                        frame_name = f"{base_name}_frame_{cnt:04d}.jpg"
                        output_path = Path(output_dir) / frame_name
                        cv2.imwrite(str(output_path), frame)
                        cnt += 1

                    count += 1
                    continue

                # ----------------------------------------------------------
                # Face detection - If no face => skip VitPose
                # ----------------------------------------------------------
                face_results = self.face_model(frame, conf=self.face_conf_threshold)
                    # Check if we have at least one bounding box from face detection
                if not face_results[0].boxes:
                    # No face detected
                    frames_without_face_detection += 1
                    frames_without_keypoints += 1
                    num_keypoints = 133 + 468
                    keypoints_data.append(np.full((num_keypoints, 3), np.nan))
                    frames_with_missing_keypoints.append(total_frames - 1)

                    if self.config['data']['save_frames']:
                        frame_name = f"{base_name}_frame_{cnt:04d}.jpg"
                        output_path = Path(output_dir) / frame_name
                        cv2.imwrite(str(output_path), frame)
                        cnt += 1
                    # Skip this frame
                    count += 1
                    continue
                # We have at least one face detection
                box = face_results[0].boxes[0]
                x1f, y1f, x2f, y2f = map(int, box.xyxy[0])

                # ----------------------------------------------------------
                #  We have a face => Face Mesh
                # ----------------------------------------------------------
                face_roi = frame[y1f:y2f, x1f:x2f]
                if face_roi is not None and face_roi.size > 0:
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    mp_results = self.face_mesh.process(face_rgb)
                    if mp_results.multi_face_landmarks:
                        frames_with_face_keypoints += 1
                        face_landmarks = mp_results.multi_face_landmarks[0].landmark
                        face_width = x2f - x1f
                        face_height = y2f - y1f
                        mp_face_keypoints = np.array([
                            (lm.x * face_width + x1f, lm.y * face_height + y1f, lm.z)
                            for lm in face_landmarks
                        ])
                    else:
                        frames_with_face_detected_but_no_landmarks += 1
                        mp_face_keypoints = np.full((468, 3), np.nan)
                        num_keypoints = 133 + 468
                        keypoints_data.append(np.full((num_keypoints, 3), np.nan))
                        frames_with_missing_keypoints.append(total_frames - 1)
                        count += 1
                        continue
                else:
                    frames_with_face_detected_but_no_landmarks += 1
                    mp_face_keypoints = np.full((468, 3), np.nan)
                    num_keypoints = 133 + 468
                    keypoints_data.append(np.full((num_keypoints, 3), np.nan))
                    frames_with_missing_keypoints.append(total_frames - 1)
                    count += 1
                    continue

                # ----------------------------------------------------------
                # Pose Estimation with VitPose (only now that face is found)
                # ----------------------------------------------------------
                pose_results = self.pose_model.inference(frame)
                img_pose = self.pose_model.get_image()

                if pose_results:
                    patient_keypoints = pose_results[0]  # The first or only detected pose
                    if patient_keypoints is not None:
                        frames_with_keypoints += 1
                        # Pose has 133 keypoints, e.g.
                        normalized_keypoints = patient_keypoints[:133, :3]
                        # Combine pose + face
                        combined_data = np.concatenate([normalized_keypoints, mp_face_keypoints], axis=0)
                        keypoints_data.append(combined_data)
                    else:
                        frames_without_keypoints += 1
                        num_keypoints = 133 + 468
                        keypoints_data.append(np.full((num_keypoints, 3), np.nan))
                        frames_with_missing_keypoints.append(total_frames - 1)
                else:
                    frames_without_keypoints += 1
                    num_keypoints = 133 + 468
                    keypoints_data.append(np.full((num_keypoints, 3), np.nan))
                    frames_with_missing_keypoints.append(total_frames - 1)

                # ----------------------------------------------------------
                # Save frames if specified
                # ----------------------------------------------------------
                if self.config['data']['save_frames']:
                    frame_name = f"{base_name}_frame_{cnt:04d}.jpg"
                    output_path = Path(output_dir) / frame_name
                    cv2.imwrite(str(output_path), img_pose)
                    cnt += 1

            count += 1

        video.release()
        torch.cuda.empty_cache()  # Add here after video release

        stats = {
            'total_frames': total_frames,
            'frames_with_keypoints': frames_with_keypoints,
            'frames_without_keypoints': frames_without_keypoints,
            'frames_with_face_keypoints': frames_with_face_keypoints,
            'frames_with_face_detected_but_no_landmarks': frames_with_face_detected_but_no_landmarks,
            'frames_without_face_detection': frames_without_face_detection,
            'frames_with_missing_keypoints': frames_with_missing_keypoints,
            'fps': fps
        }
        return keypoints_data, stats

    def process_videos(self, args):

        raw_dir = Path(self.config['data']['raw_dir'])
        processed_dir = Path(self.config['data']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)

        for video_file in raw_dir.glob('*'):
            if video_file.suffix.lower() in ('.mp4', '.avi', '.mov', '.mts', '.wmv', '.mpg'):
                # Check if corresponding .npy file already exists
                npy_path = processed_dir / f"{video_file.stem}_keypoints.npy"
                if npy_path.exists():
                    print(f"Skipping {video_file.name} - already processed ({npy_path} exists)")
                    continue

                print(f"Processing video: {video_file}")
                keypoints_data, stats = self.extract_frames_and_keypoints(video_file, processed_dir)

                if not keypoints_data:
                    print(f"Skipping {video_file.name} - no valid keypoints extracted")
                    continue

                # Convert to numpy array and validate
                keypoints_array = np.array(keypoints_data)
                if keypoints_array.size == 0 or np.all(np.isnan(keypoints_array)):
                    print(f"Skipping {video_file.name} - invalid keypoints data")
                    continue

                 # Save keypoints data
                try:
                    np.save(str(npy_path), keypoints_array)
                except OSError as e:
                    print(f"Error saving keypoints for {video_file}: {e}")
                    continue
            
                # Calculate additional metrics
                percent_frames_with_keypoints = (
                    stats['frames_with_keypoints'] / stats['total_frames'] * 100
                    if stats['total_frames'] > 0 else 0
                )
                torch.cuda.empty_cache()  # Add here after video release
                
                #wandb.log({
                   # "video_name": video_file.name,
                   # "num_frames": len(keypoints_data),
                   # "total_frames": stats['total_frames'],
                   # "frames_with_keypoints": stats['frames_with_keypoints'],
                    #"frames_without_keypoints": stats['frames_without_keypoints'],
                    #"frames_with_face_keypoints": stats['frames_with_face_keypoints'],
                    #"frames_with_face_detected_but_no_landmarks": stats['frames_with_face_detected_but_no_landmarks'],
                   # "frames_without_face_detection": stats['frames_without_face_detection'],
                    #"percent_frames_with_keypoints": percent_frames_with_keypoints,
                   # "frames_with_missing_keypoints": stats['frames_with_missing_keypoints'],
                   # "fps": stats['fps']
                #})

       # wandb.finish()
