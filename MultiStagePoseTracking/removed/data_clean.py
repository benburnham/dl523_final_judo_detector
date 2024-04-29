import os
import json

# Function to iterate through directories and convert MP4 files to JSON
def convert_dataset(root_dir):
    total=0
    thresh = 600
    destination='removed'
    for category in os.listdir(root_dir):   # All or NO_MIRROR
        category_path = os.path.join(root_dir, category)
        
        for subcategory in os.listdir(category_path):   # train or evaluate
            subcategory_path = os.path.join(category_path, subcategory)
            
            for technique in os.listdir(subcategory_path):   # Technique
                technique_path = os.path.join(subcategory_path, technique)
                
                for filename in os.listdir(technique_path):   # files
                    if filename.endswith(".json"):
                        json_path = os.path.join(technique_path, filename)
                        with open(json_path, 'r') as json_file:
                            pose_data = json.load(json_file)
                        if len(pose_data)>thresh:
                            destination_file = os.path.join(destination, filename)
                            os.rename(json_path, destination_file)
                            print('Length of', len(pose_data), 'in', json_path, 'Moved to', destination_file)
                            total+=1
    print('Total: ', total)

# Define the root directory of the new dataset
new_dataset_root = 'Pose_Dataset'

# Call the function to convert the dataset
convert_dataset(new_dataset_root)