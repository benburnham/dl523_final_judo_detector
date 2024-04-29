import os
import torch
from mmpose.apis import MMPoseInferencer

# Function to iterate through directories and convert MP4 files to JSON
def convert_dataset(root_dir, new_dataset_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det_model='mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py'
    det_weights='../../rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth'
    pose_detection_model = MMPoseInferencer(pose2d='rtmo',
                                            det_model=det_model,
                                            det_weights=det_weights,
                                            det_cat_ids=[0],    # the category id of 'human' class
                                            device=device)
    
    for category in os.listdir(root_dir):   # All or NO_MIRROR
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            new_category_path = os.path.join(new_dataset_root, category)
            os.makedirs(new_category_path, exist_ok=True)

            for subcategory in os.listdir(category_path):   # train or evaluate
                subcategory_path = os.path.join(category_path, subcategory)
                if os.path.isdir(subcategory_path):
                    new_subcategory_path = os.path.join(new_category_path, subcategory)
                    os.makedirs(new_subcategory_path, exist_ok=True)

                    for technique in os.listdir(subcategory_path):   # Technique
                        technique_path = os.path.join(subcategory_path, technique)
                        if os.path.isdir(technique_path):
                            new_technique_path = os.path.join(new_subcategory_path, technique)
                            os.makedirs(new_technique_path, exist_ok=True)

                            for filename in os.listdir(technique_path):   # Videos
                                if filename.endswith(".mp4"):
                                    mp4_path = os.path.join(technique_path, filename)
                                    detection_generator = pose_detection_model(mp4_path, pred_out_dir=new_technique_path)
                                    detections = [result for result in detection_generator]


# Define the root directory of the original dataset
original_dataset_root = '../../FINAL DATASET/'

# Define the root directory of the new dataset
new_dataset_root = 'Pose_Dataset'

# Call the function to convert the dataset
convert_dataset(original_dataset_root, new_dataset_root)