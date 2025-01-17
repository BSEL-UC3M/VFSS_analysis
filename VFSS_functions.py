# Importing all necessary libraries
import cv2
import os
from os.path import join
import glob
import numpy as np
import pandas as pd
import spicy
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label, regionprops
import shutil
from sympy import Point, Line
import SimpleITK as sitk
import math

from batchgenerators.utilities.file_and_folder_operations import *
            
def get_directory_paths(dirName, only_names = False):
    """
    Get paths of all files in a directory
    """
    list_paths = []
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        for file in filenames:
            if only_names:
                list_paths.append(file)
            else:
                list_paths.append(join(dirName, file))
        list_paths.sort()
    return list_paths
  
def transform_VFSS_to_images(input_VFSS_path, out_dir, return_fps=False):
    """
    Extracts frames from a VFSS and saves them as images in .jpg format.

    Parameters:
        input_VFSS_path (str): Path to the input VFSS file.
        out_dir (str): Directory where extracted images will be saved.
        return_fps (bool): If True, return the frames per second (FPS) of the video.

    Returns:
        float or None: FPS of the video if return_fps is True, otherwise None.
    """
    
    # Initialize video capture
    cam = cv2.VideoCapture(input_VFSS_path)
    if not cam.isOpened(): raise ValueError(f"Error opening video file: {input_VFSS_path}")
    
    # Create output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Frame extraction and saving
    frame_count = 0
    while True:
        ret,frame = cam.read()
        if not ret: break

        img_filename = join(out_dir, f'{frame_count}.jpg')

        # Save frame as an image
        cv2.imwrite(img_filename, frame)
        frame_count += 1

    # Output information
    print(f'Total number of frames: {frame_count}')
    
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(f'FPS: {fps}')

    # Release resources
    cam.release()
    cv2.destroyAllWindows()
    
    if return_fps: return fps

def jpg_to_train_format(dir_imgs, frame_size, save_name):
    """
    Converts JPG images in the specified directory to NIfTI format, resizes/crops them to the given frame size, 
    and saves them with a specific naming convention.
    
    Parameters:
        dir_imgs (str): Directory containing JPG images.
        frame_size (int): Desired frame size for the output images.
        save_name (str): Base name for the saved NIfTI files.
    """

    # Iterate through all JPG files in the directory
    for filename in glob.glob(join(dir_imgs, '*.jpg')):
        img = cv2.imread(filename)
        img = img.transpose((2, 0, 1))  #Change image shape to (Channels, Height, Width)
        img = img[:, None] # Add an extra dimension to match the expected format
        for j, channel in enumerate(img):  # Loop over channels
            if channel.shape[1] > frame_size and channel.shape[2] > frame_size:
                # Crop to the center if image is larger than frame_size
                crop = channel[:,
                                  int((channel.shape[1] - frame_size)/2):-int((channel.shape[1] - frame_size)/2),
                                  int((channel.shape[2] - frame_size)/2):-int((channel.shape[2] - frame_size)/2)]
            else:
                # Pad with zeros if the image is smaller than frame_size
                crop = np.zeros((1, frame_size, frame_size))
                crop[:,
                     int((frame_size - channel.shape[1])/2):-int((frame_size - channel.shape[1])/2),
                     int((frame_size - channel.shape[2])/2):-int((frame_size - channel.shape[2])/2)] = channel
            
            # Convert the array to a SimpleITK Image
            nifti_image = sitk.GetImageFromArray(crop)

            # Save output
            file_number = nnUNet_format(filename.split('/')[-1][:-4])
            sitk.WriteImage(nifti_image, join(dir_imgs, f'{save_name}_{file_number}_{j:04}.nii.gz'))
        
        # Remove the original JPG file after conversion
        os.remove(filename)

def nnUNet_format(num):
    """
    Formats the given number as a string with at least three digits, padding with zeros if necessary.
    
    Parameters:
        num (int or str): The number to format.
        
    Returns:
        str: The formatted number as a three-digit string.
    """

    return f'{int(num):03d}'

def postprocessing_predictions(output_dir, landmark_size = 6, labels_dict=None):
    """
    Post-processes the segmentation predictions by performing the following tasks:
    
    1. **Retrieve Labels Present in all VFSS frames:** 
       Collects unique labels found in the segmented images to identify which structures are present.
    
    2. **Landmark Interpolation:** 
       Interpolates missing or noisy landmarks based on the center of mass (COM) of the landmark in each image. 
       This corrects segmentation errors in landmarks for labels like C2, C4, Tongue, Hyoid, etc.
    
    3. **Crop PPP Label:** 
       Refines the segmentation of the PPP (Posterior Pharyngeal Wall) by cropping it based on the top boundary of 
       the LUMI (Lumen) label and the bottom boundary of the SSO (Suprahyoid Space) labels (both internal and external).
    
    Parameters:
    - output_dir (str): Directory containing the predicted segmentation files (in NIfTI format).
    - landmark_size (int): Size of the region around the landmarks for interpolation. Default is 6.
    - labels_dict (dict): A dictionary mapping label numbers to their corresponding anatomical structure names. 
      Default mapping is provided if none is given.
    """    
    
    if labels_dict is None:
        labels_dict = {1: 'Lumi', 2: 'PPP', 3: 'C2', 4: 'C4', 5: 'Tongue', 6: 'Hyoid', 7: 'sso_int', 8: 'sso_ext', 9: 'stases_val', 10: 'stases_sso'}

    test_labs = sorted(glob.glob(os.path.join(output_dir, '*.nii.gz')))

    # 1. Get labels that appear in VFSS
    labels_nums = []
    for test_file in test_labs: 
        labels_present = list(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(test_file))))
        labels_nums.extend(labels_present)
    labels_nums = list(set(labels_nums))

    # 2. Interpolate landmarks to account for segmentation errors:
    for landmark in [3, 4, 5, 6, 7, 8]: 
        if landmark in labels_present:
            print(f'\nInterpolating {labels_dict[landmark]} points')
            COM_list = [] # Centers of Mass
            files_list = [] # File numbers to interpolate

            for test_file in test_labs:
                file_number = test_file.split('/')[-1].split('.')[0].split('_')[1]
                file_np = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(test_file)))
                file_landmark = np.where(file_np == landmark, 1, 0)

                if len(np.unique(file_landmark)) > 1:
                    files_list.append(int(file_number))
                    COM_list.append(spicy.ndimage.center_of_mass(file_landmark))
                    
            COM_list_x = [COM[0] for COM in COM_list]
            COM_list_y = [COM[1] for COM in COM_list]
            if len(COM_list_x)>2 and len(COM_list_y)>2:
                f = spicy.interpolate.interp1d(files_list, (COM_list_x, COM_list_y))
                COM_x, COM_y = f(range(files_list[0], files_list[-1])).astype(int)

                for pos, file_number in enumerate(range(files_list[0], files_list[-1])):
                    try:
                        num = nnUNet_format(file_number)
                        file_path = [t for t in test_labs if num in t.split('/')[-1]] 
                        file_sitk = sitk.ReadImage(file_path)
                        file_np = sitk.GetArrayFromImage(file_sitk)

                        interpolated_label = np.zeros(file_np.shape)
                        interpolated_label[:,
                                            COM_x[pos] - landmark_size: COM_x[pos] + landmark_size,
                                            COM_y[pos] - landmark_size : COM_y[pos] + landmark_size] = 1
                        interpolated_label = gaussian_filter(interpolated_label, sigma=4)

                        interpolated_file = np.where(interpolated_label>0.4, int(landmark), file_np)
                        interpolated_file = sitk.GetImageFromArray(interpolated_file)
                        interpolated_file.CopyInformation(file_sitk)
                        sitk.WriteImage(interpolated_file, file_path)
                    except: continue

    # 3. Crop PPP label to top of LUMI and bottom of SSO (int + ext)
    PPP_label, lumi_label, sso_int_label, sso_ext_label = 2, 1, 7, 8
    if PPP_label in labels_present:
        print(f'\nCropping {labels_dict[2]} label')
        lumi_max, SSO_min = [], []

        try:
            for test_file in test_labs:
                file_np = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(test_file)))

                # Find LUMI max and SSO min positions
                lumi_y = np.where(file_np == lumi_label)[0]
                sso_int_y = np.where(file_np == sso_int_label)[0]
                sso_ext_y = np.where(file_np == sso_ext_label)[0]

                if len(lumi_y) > 0: lumi_max.append(np.min(lumi_y))
                if len(sso_int_y) > 0: SSO_min.append(np.max(sso_int_y))
                if len(sso_ext_y) > 0: SSO_min.append(np.max(sso_ext_y))

            if lumi_max and SSO_min:
                PPP_max = np.min(np.array(lumi_max))
                PPP_min = np.max(np.array(SSO_min))

                for test_file in test_labs:
                    file_sitk = sitk.ReadImage(test_file)
                    file_np = sitk.GetArrayFromImage(file_sitk)
                    PPP_np = np.where(file_np==PPP_label, PPP_label, 0)
                    file_np = np.where(file_np==PPP_label, 0, file_np)

                    if len(np.unique(PPP_np))>1:
                        SSO_int = 0
                        SSO_ext = 0
                        SSO_int_y = np.where(np.squeeze(file_np==sso_int_label))[0]
                        SSO_ext_y = np.where(np.squeeze(file_np==sso_ext_label))[0]
                        if len(SSO_int_y)>0: SSO_int = np.max(SSO_int_y)
                        if len(SSO_ext_y)>0: SSO_ext = np.max(SSO_ext_y)
                        PPP_bottom = np.max(np.array([SSO_int, SSO_ext]))
                        if PPP_bottom == 0: PPP_bottom = PPP_min

                        PPP_np[:PPP_max, :] = 0
                        PPP_np[PPP_bottom:, :] = 0

                        cropped_file = np.where(PPP_np==PPP_label, PPP_label, file_np)
                        cropped_file = sitk.GetImageFromArray(cropped_file)
                        cropped_file.CopyInformation(file_sitk)
                        sitk.WriteImage(cropped_file, test_file)
        except: 
            print(f'\nNo cropping was needed')


def create_videos(output_dir, original_VFSS_path, outt, save_name, labels_dict=None, colors=None, individual_videos=False):
    """
    This function generates videos from the predicted segmentation masks and corresponding original VFSS frames. 
    It performs the following tasks:
    
    1. **Create a Video with All Labels:**
       - Combines the segmentation masks with the original VFSS frames, color-codes the labels, and creates a video with all the labeled structures.
    
    2. **Create Separate Videos for Each Label:**
       - For each label in the segmentation, creates a separate video highlighting only that specific label.

    Parameters:
    - output_dir (str): Directory containing the predicted segmentation files (in NIfTI format).
    - original_VFSS_path (str): Path to the original VFSS file.
    - outt (str): Output directory where the videos will be saved.
    - save_name (str): Base name for the saved videos.
    - labels_dict (dict): A dictionary mapping label numbers to their corresponding anatomical structure names. 
      If none is provided, a default dictionary is used.
    - colors (dict): A dictionary mapping label numbers to RGB color values. 
      If none is provided, a default color dictionary is used.
    - individual_videos (bool): If True, videos for individual labels will be created, apart from the video with all labels.
    """    
    
    if labels_dict is None:
        labels_dict = {1: 'Lumi', 2: 'PPP', 3: 'C2', 4: 'C4', 5: 'Tongue', 6: 'Hyoid', 7: 'sso_int', 8: 'sso_ext', 9: 'stases_val', 10: 'stases_sso'}
    if colors is None:
        colors = {1:[255,0,0], 2:[0,128,255], 3:[255,0,255], 4:[255,255,0], 5:[255,0,255], 6:[255,255,0], 7:[0,255,255], 8:[0,255,255], 9:[0,0,255], 10:[0,255,0]}

    test_labs = sorted(glob.glob(join(output_dir, '*.nii.gz')))

    # 1. Get labels that appear in VFSS
    labels_nums = []
    for test_file in test_labs: 
        labels_present = list(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(test_file))))
        labels_nums.extend(labels_present)
    labels_nums = list(set(labels_nums))

    # 2. Transform original VFSS file to image frames
    fps = transform_VFSS_to_images(original_VFSS_path, join(output_dir, 'temp'), return_fps=True)
    print('fps: ', fps)
    file_list = sorted(glob.glob(join(output_dir, 'temp', '*.jpg')))

    # 3. Create video with all labels
    print('Creating video with all labels...')
    for frame_file in range(len(file_list)):
        num = nnUNet_format(frame_file)

        # Read image frame
        img_frame = cv2.imread(join(output_dir, 'temp', str(frame_file)+'.jpg'))
        label_file = [l_file for l_file in test_labs if num in l_file.split('/')[-1]] 
        label_np = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(label_file)))
        label_new = np.zeros((img_frame.shape[0], img_frame.shape[1]))
        start_h = max(0, int((img_frame.shape[0]-label_np.shape[0])/2))
        start_w = max(0, int((img_frame.shape[1]-label_np.shape[1])/2))
        end_h = max(0, int((label_np.shape[0]-img_frame.shape[0])/2))
        end_w = max(0, int((label_np.shape[1]-img_frame.shape[1])/2))

        # Dynamically create slices
        # If start_h or start_w is 0, we want to use ":" in that dimension
        slice_new_h = slice(start_h, -start_h if start_h > 0 else None)
        slice_new_w = slice(start_w, -start_w if start_w > 0 else None)

        slice_label_h = slice(end_h, -end_h if end_h > 0 else None)
        slice_label_w = slice(end_w, -end_w if end_w > 0 else None)

        # Apply the slicing
        label_new[slice_new_h, slice_new_w] = label_np[slice_label_h, slice_label_w]
        
        for l in labels_nums:
            if l != 0:
                img_frame[:,:,0] = np.where(label_new==l, colors[l][0], img_frame[:,:,0])
                img_frame[:,:,1] = np.where(label_new==l, colors[l][1], img_frame[:,:,1])
                img_frame[:,:,2] = np.where(label_new==l, colors[l][2], img_frame[:,:,2])
        
        cv2.imwrite(join(output_dir, 'temp', str(frame_file)+'.jpg'), img_frame)

    # Save video with all labels
    frame = cv2.imread(file_list[0])
    height, width, layers = frame.shape
    frame_size = (width,height)  

    out = cv2.VideoWriter(join(outt, f'{save_name}_complete.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)  

    for num, file_frame in enumerate(range(len(file_list))):
        frame = cv2.imread(join(output_dir, 'temp', str(file_frame)+'.jpg'))
        out.write(frame)
    cv2.destroyAllWindows()
    out.release()

    print('Video saved at: ', join(outt, f'{save_name}_complete.avi'))
    shutil.rmtree(join(output_dir, 'temp'))

    # 4. Create video for each label
    if individual_videos:
        print('Creating videos for each label...')
        for l in labels_nums:
            if l!=0:
                transform_VFSS_to_images(original_VFSS_path, join(output_dir, 'temp'))
                for frame_file in range(len(file_list)):
                    num = nnUNet_format(frame_file)

                    # Read image frame
                    img_frame = cv2.imread(join(output_dir, 'temp', str(frame_file)+'.jpg'))
                    label_file = [l_file for l_file in test_labs if num in l_file.split('/')[-1]] 
                    label_np = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(label_file)))
                    label_new = np.zeros((img_frame.shape[0], img_frame.shape[1]))
                    label_new[slice_new_h, slice_new_w] = label_np[slice_label_h, slice_label_w]
                    img_frame[:,:,0] = np.where(label_new==l, colors[l][0], img_frame[:,:,0])
                    img_frame[:,:,1] = np.where(label_new==l, colors[l][1], img_frame[:,:,1])
                    img_frame[:,:,2] = np.where(label_new==l, colors[l][2], img_frame[:,:,2])
                    cv2.imwrite(join(output_dir, 'temp', str(frame_file)+'.jpg'), img_frame)

                # Save video
                frame = cv2.imread(file_list[0])
                height, width, layers = frame.shape
                frame_size = (width,height)  

                out = cv2.VideoWriter(join(outt, f'{save_name}_{labels_dict[l]}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)

                for num, file_frame in enumerate(range(len(file_list))):
                    frame = cv2.imread(join(output_dir, 'temp', str(file_frame)+'.jpg'))
                    out.write(frame)
                cv2.destroyAllWindows()
                out.release()

                shutil.rmtree(join(output_dir, 'temp'))

def seg_areas_from_video(output_dir, labels_dir, frame_size=800, save_name='VFSS'):
    """
    Extracts segmentation areas from labeled video frames and saves them to CSV files.

    Parameters:
        output_dir (str): Directory to save the output CSV files.
        labels_dir (str): Directory containing the segmentation files.
        labels_dict (dict): Dictionary mapping label numbers to their corresponding names.
        labels_nums (list): List of label numbers to process.
        frame_size (int): Size of the frame for the video. Default is 800.
        save_name (str): Prefix for the saved segmentation files. Default is 'VFSS'.
    
    Saves:
        CSV files with the calculated areas for each label.
    """
    labels_dict = {1: 'lumi', 2: 'paroi', 3: 'c2', 4: 'c4', 5: 'langue', 6: 'hyoid', 7: 'sso_int', 8: 'sso_ext', 9: 'stases_val', 10: 'stases_sso'} # Default values
    labels_nums=[1, 9, 10]
    
    label_paths = sorted(glob.glob(join(labels_dir, '*.nii.gz')))
    for label_num in labels_nums:
        print('Processing: ', labels_dict[label_num])
        area_list = []

        for slide, _ in enumerate(label_paths):
            # Get numpy arrays
            numm = nnUNet_format(slide)
            label_path = join(labels_dir, f'{save_name}_{str(numm)}.nii.gz')
            label_sitk = sitk.ReadImage(label_path)
            label_np = sitk.GetArrayFromImage(label_sitk)
            seg = np.zeros(label_np.shape)

            seg = np.where(label_np == label_num, 1, 0)
            
            # Calculate  area
            if len(np.unique(seg))>1:
                labeled_seg = label(seg)
                regions = regionprops(labeled_seg)
                total_area = sum(region.area for region in regions)
                area_list.append(int(total_area))
            else:
                area_list.append(int(0))

        # Save area data to CSV
        df = pd.DataFrame({'Slide': range(1, len(label_paths)+1), 'Area': area_list}) 
        df.to_csv(join(output_dir, labels_dict[label_num].upper()+'.csv'), index=False)
        try: shutil.rmtree(join(labels_dir, 'temp'))
        except: pass

def landmarks_params_from_video(labels_dir, output_dir, paroi=2, c2=3, c4=4, tongue=5, hyoid=6, sso_int=7, sso_ext=8, segments_PCM=3, save_name='VFSS'):
    """
    Computes various anatomical parameters (e.g., tongue movement, hyoid elevation, PCM movement) from labeled video frames and saves them as CSV files.

    Parameters:
        labels_dir (str): Directory containing the segmentation files.
        output_dir (str): Directory to save the output CSV files.
        paroi (int): Label number for the PCM structure. Default is 2.
        c2 (int): Label number for the C2 vertebra. Default is 3.
        c4 (int): Label number for the C4 vertebra. Default is 4.
        tongue (int): Label number for the tongue structure. Default is 5.
        hyoid (int): Label number for the hyoid bone. Default is 6.
        sso_int (int): Label number for the internal SSO. Default is 7.
        sso_ext (int): Label number for the external SSO. Default is 8.
        segments_PCM (int): Number of segments to divide the PCM structure into. Default is 3.
        save_name (str): Prefix for the saved segmentation files. Default is 'VFSS'.
    
    Outputs:
        CSV files with the computed anatomical parameters.
    """

    label_paths = sorted(glob.glob(join(labels_dir, '*.nii.gz')))

    # Parameters
    slides = [s+1 for s, file in enumerate(label_paths)]
    SSO_opening = []
    tongue_movement = []
    hyoid_elevation = []
    hyoid_elevation_X = []
    PCM_movement = np.zeros((segments_PCM, len(slides)))
    dist_vertebrae_list = []

    for file_frame, file in enumerate(label_paths): 
        # Get numpy arrays
        numm = nnUNet_format(file_frame)
        label_path = join(labels_dir, f'{save_name}_{str(numm)}.nii.gz')
        label_sitk = sitk.ReadImage(label_path)
        label_np = sitk.GetArrayFromImage(label_sitk)

        # SSO opening
        try:
            COM_ext = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(sso_ext), 1, 0))))
            COM_int = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(sso_int), 1, 0))))
            dist = np.linalg.norm(abs(COM_ext - COM_int))
            SSO_opening.append(dist)
        except: SSO_opening.append('nan')

        # tongue_movement
        try:
            COM_tongue = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(tongue), 1, 0)))[1])
            COM_C2 = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(c2), 1, 0)))[1])
            dist = np.linalg.norm(abs(COM_tongue - COM_C2))
            tongue_movement.append(dist)
        except: tongue_movement.append('nan')

        # hyoid_elevation
        try:
            COM_hyoid = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(hyoid), 1, 0))))
            COM_C2 = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(c2), 1, 0))))
            COM_C4 = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(c4), 1, 0))))
            lC2C4 = Line(Point(COM_C2), Point(COM_C4))
            pC2C4 = lC2C4.perpendicular_line(Point(COM_C4))
            point_hyoid = Point(COM_hyoid)
            dist = float(pC2C4.distance(point_hyoid))
            hyoid_elevation.append(dist)
            point_X = np.array((COM_hyoid[1]))
            point_C4 = np.array((COM_C4[1]))
            dist_X = np.linalg.norm(point_X - point_C4)
            hyoid_elevation_X.append(dist_X)
        except: 
            hyoid_elevation.append('nan')
            hyoid_elevation_X.append('nan')

        # PCM_movement
        COM_C2 = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(c2), 1, 0))))
        COM_C4 = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(c4), 1, 0))))
        try:
            lC2C4 = Line(Point(COM_C2), Point(COM_C4))
            list_PCM = np.where(np.squeeze(label_np)==int(paroi))
            cut = int((np.max(list_PCM[0]) - np.min(list_PCM[0])) / segments_PCM)
            for pos, seg in enumerate(range(segments_PCM)):
                Y_val = np.min(list_PCM[0]) + int(cut*seg + cut/2)
                X_val = int(np.mean([list_PCM[1][v] for v in np.where(list_PCM[0]==Y_val)[0]]))
                point_PCM = Point(Y_val, X_val)
                dist = float(lC2C4.distance(point_PCM))
                PCM_movement[seg, file_frame] = dist
        except: 
            for pos, seg in enumerate(range(segments_PCM)): PCM_movement[seg, file_frame] = 'nan'

        # Distance vertebrae
        COM_C2 = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(c2), 1, 0))))
        COM_C4 = np.array(spicy.ndimage.center_of_mass(np.squeeze(np.where(label_np==int(c4), 1, 0))))
        dist = np.linalg.norm(COM_C2 - COM_C4)
        if not math.isnan(dist): dist_vertebrae_list.append(dist)
    
    df = pd.DataFrame({'Slide': slides, 'Distances': SSO_opening}) 
    df.to_csv(join(output_dir, 'SSO_opening.csv'))
    df = pd.DataFrame({'Slide': slides, 'Distances': tongue_movement}) 
    df.to_csv(join(output_dir, 'tongue_movement.csv'))
    df = pd.DataFrame({'Slide': slides, 'Distances_elevation': hyoid_elevation, 'Distances_X': hyoid_elevation_X}) 
    df.to_csv(join(output_dir, 'hyoid_elevation.csv'))
    df = pd.DataFrame({'Slide': slides}) 
    for pos, seg in enumerate(range(segments_PCM)):
        df['Segment_'+str(pos+1)] = list(PCM_movement[pos, :])
    df.to_csv(join(output_dir, 'PCM_movement.csv'))

    print('\nAverage C2-C4 distance is: %.3f pixels' %np.mean(np.array(dist_vertebrae_list)))