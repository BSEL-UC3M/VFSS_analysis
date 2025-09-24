"""
For a new VFSS, read the video, preprocess it, predict all ROIs and save videos with predicted labels and extracted dynamic parameters.
"""

# Importing all necessary libraries
import os
from os.path import join
import glob
import sys
import time
import cv2
import paths_repository
from VFSS_functions import (
    get_directory_paths, 
    transform_VFSS_to_images, 
    jpg_to_train_format, 
    postprocessing_predictions, 
    create_videos, 
    seg_areas_from_video, 
    landmarks_params_from_video
)

################################################################################################################################################

# Constants #
task_name = 'Task010_VFSS' # Model name for inference --> Two options : Task010_VFSS, Task008_VFSS (the best performing models)
frame_size = 800 # Default frame size for preprocessing

# Extract task number from task name
try:
    task_number = task_name.split('_')[0][4:]
except IndexError:
    print("Error: Task name is incorrectly formatted. Exiting...")
    sys.exit()

# Input data names
medical_institutions = ['test'] # List of institutions
VFSS_data = [{'healthy_001': ['t0']}] # For each institution, a new dictionary. In each dictionary --> patient_id: [time_points]

if len(medical_institutions) != len(VFSS_data): 
    print("Error: Medical institutions and VFSS data mismatch. Exiting...")
    sys.exit()

# Automatic configuration of paths and files
raw_VFSS_dir = paths_repository.raw_VFSS_dir
output_dir = paths_repository.output_dir
train_data_dir = paths_repository.train_data_dir
models_dir = paths_repository.models_dir

os.environ['RESULTS_FOLDER'] = models_dir
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"

################################################################################################################################################

def process_vfss():

    """Process VFSS data for each institution, patient, and time point."""

    for institution_index, medical_institution in enumerate(medical_institutions):
        institution_data = VFSS_data[institution_index]

        for patient_id, time_points in institution_data.items():
            for time_point in time_points:
                print(f"\nProcessing: Patient {patient_id}, Time Point {time_point}\n")

                # Define paths
                patient_folder = join(train_data_dir, medical_institution, patient_id, time_point)
                test_dir = {join(patient_folder, 'imagesTs') : join(patient_folder, 'labelsTs')}
                for img_test_dir, label_test_dir in test_dir.items(): pred_labels_dir = f'{label_test_dir}_preds'

                # Step 1: Preprocess VFSS
                preprocess_vfss(medical_institution, patient_id, time_point, img_test_dir)

                # Step 2: Perform inference
                perform_inference(img_test_dir, pred_labels_dir)

                # Step 3: Create labeled videos
                create_labeled_videos(pred_labels_dir, medical_institution, patient_id, time_point)

                # Step 4: Extract parameters
                extract_parameters(pred_labels_dir, medical_institution, patient_id, time_point)

def preprocess_vfss(medical_institution, patient_id, time_point, img_test_dir):
    """Preprocess VFSS data and convert to NifTI format."""
    print("Step 1: Preprocessing VFSS data")
    start_time = time.time()

    VFSS_paths = get_directory_paths(join(raw_VFSS_dir, medical_institution, patient_id, time_point))
    if len(VFSS_paths) > 1:
        print("Warning: Multiple VFSS files found. Using the first file.")
    original_VFSS_path = VFSS_paths[0]

    transform_VFSS_to_images(original_VFSS_path, img_test_dir)
    file_list = sorted(glob.glob(join(img_test_dir, '*.jpg')))

    frame = cv2.imread(file_list[0])
    height, width, _ = frame.shape
    print(f'Original VFSS frame size is {(width, height)}')

    jpg_to_train_format(img_test_dir, frame_size, save_name='VFSS')
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

def perform_inference(img_test_dir, pred_labels_dir):
    """Run inference on test data."""
    print("Step 2: Performing inference")
    start_time = time.time()
    
    os.system(f'nnUNet_predict -i {img_test_dir} -o {pred_labels_dir} -t {task_number} -tr nnUNetTrainerV2 -m 2d --num_threads_preprocessing 1')

    postprocessing_predictions(pred_labels_dir)
    print(f"Inference completed in {time.time() - start_time:.2f} seconds")

def create_labeled_videos(pred_labels_dir, medical_institution, patient_id, time_point):
    """Generate videos with predicted labels."""
    print("Step 3: Creating labeled videos")
    start_time = time.time()

    VFSS_paths = get_directory_paths(join(raw_VFSS_dir, medical_institution, patient_id, time_point))
    if len(VFSS_paths) > 1:
        print("Warning: Multiple VFSS files found. Using the first file.")
    original_VFSS_path = VFSS_paths[0]

    output_folder = join(output_dir, medical_institution, patient_id, time_point)
    os.makedirs(output_folder, exist_ok=True)
    video_name = f'{patient_id}_{time_point}_DL_{task_name}'
    
    create_videos(pred_labels_dir, original_VFSS_path, output_folder, video_name, individual_videos=True)
    print(f"Labeled videos created in {time.time() - start_time:.2f} seconds")

def extract_parameters(pred_labels_dir, medical_institution, patient_id, time_point):
    """Extract segmentation areas and landmarks from videos."""
    print("Step 4: Extracting parameters")
    start_time = time.time()

    output_folder = join(output_dir, medical_institution, patient_id, time_point)
    seg_areas_from_video(output_folder, pred_labels_dir)
    landmarks_params_from_video(pred_labels_dir, output_folder)

    print(f"Parameter extraction completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    process_vfss()
