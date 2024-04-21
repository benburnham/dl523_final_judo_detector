#!/bin/bash

# Directory where the script is located
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Directory to store train and evaluate directories
train_dir="$script_dir/dataset/ALL/train"
eval_dir="$script_dir/dataset/ALL/evaluate"

# Directory to store train and evaluate directories - NO MIRRORS
train_dir_no_mir="$script_dir/dataset/NO_MIRROR/train"
eval_dir_no_mir="$script_dir/dataset/NO_MIRROR/evaluate"

# Iterate through each technique directory
for technique_dir in "$script_dir"/*; do
    # Create train and evaluate directories if not exist
    mkdir -p "$train_dir"
    mkdir -p "$eval_dir"

    # Ensure it's a directory
    if [ -d "$technique_dir" ]; then
        # Get name of technique
        technique_name=$(basename "$technique_dir")
        
        # Define train and evaluate directories for the technique
        train_technique_dir="$train_dir_no_mir/${technique_name}_train"
        eval_technique_dir="$eval_dir_no_mir/${technique_name}_validate"

        # Create train and evaluate directories for the technique
        mkdir -p "$train_technique_dir"
        mkdir -p "$eval_technique_dir"

        # Count total number of mp4 files excluding mirror videos
        total_files=$(find "$technique_dir" -maxdepth 3 -type f -name "*.mp4" ! -iname "*mirror*" | wc -l)

        # Calculate number of files to move to evaluate directory
        eval_count=$(( $total_files / 10 ))

        # Counter for files moved
        moved_count=0

        # Move files to evaluate directory
        find "$technique_dir" -maxdepth 3 -type f -name "*.mp4" ! -iname "*mirror*" | shuf | while read -r file; do
            if [ $moved_count -lt $eval_count ]; then
                cp "$file" "$eval_technique_dir/${technique_name}_validate_$((++moved_count)).mp4"
            else
                cp "$file" "$train_technique_dir/${technique_name}_train_$((++moved_count - $eval_count)).mp4"
            fi
        done

        echo "Moved $eval_count files to $eval_technique_dir"
        echo "Moved $(( $total_files - $eval_count )) files to $train_technique_dir"
        
        # Define train and evaluate directories for the technique
        train_technique_dir="$train_dir/${technique_name}_train"
        eval_technique_dir="$eval_dir/${technique_name}_validate"

        # Create train and evaluate directories for the technique
        mkdir -p "$train_technique_dir"
        mkdir -p "$eval_technique_dir"
        
        # Mirror all videos for given technique
        mirrored="$technique_dir/mirrored"
        mkdir -p "$mirrored"
        for examples_folder in "$technique_dir"/*/*/; do
            examples=$(basename -- "$examples_folder")
            examples_new="${examples}_mirrored"
            mkdir -p "$mirrored/$examples_new"

            for video in "$examples_folder"/*.mp4; do
                filename=$(basename -- "$video")
                mirrored_filename="${filename%.*}_mirrored.mp4"
                
                # Perform horrizontal flip
                ffmpeg -i "$video" -vf "hflip" "$mirrored/$examples_new/$mirrored_filename"
            done
        done

        # Count total number of mp4 files
        total_files=$(find "$technique_dir" -maxdepth 3 -type f -name "*.mp4" | wc -l)

        # Calculate number of files to move to evaluate directory
        eval_count=$(( $total_files / 10 ))

        # Counter for files moved
        moved_count=0

        # Move files to evaluate directory
        find "$technique_dir" -maxdepth 3 -type f -name "*.mp4" | shuf | while read -r file; do
            if [ $moved_count -lt $eval_count ]; then
                mv "$file" "$eval_technique_dir/${technique_name}_validate_$((++moved_count)).mp4"
            else
                mv "$file" "$train_technique_dir/${technique_name}_train_$((++moved_count - $eval_count)).mp4"
            fi
        done

        echo "Moved $eval_count files to $eval_technique_dir"
        echo "Moved $(( $total_files - $eval_count )) files to $train_technique_dir"
        
        rm -r "$technique_dir"
        
        
    fi
done

