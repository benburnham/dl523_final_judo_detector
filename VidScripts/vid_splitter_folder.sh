#!/bin/bash

# Iterate over each directory
for dir in */; do
    # Enter the directory
    cd "$dir"
    
    total_clips=0

    # Iterate over each .mp4 file
    for file in *.mp4; do
        # Check if the file is actually an .mp4 video
        if [[ -f "$file" ]]; then
            
            # Extract the filename without extension
            filename=$(basename -- "$file")
            filename_no_ext="${filename%.*}"

            # Create a directory with the same name as the video
            mkdir "$filename_no_ext"

            # Move the video file into the created directory
            cp "$file" "$filename_no_ext/"

            # Enter the directory
            cd "$filename_no_ext"
            
            # Resize video to 720p using ffmpeg, delete source
            ffmpeg -i "$filename" -vf scale=-2:720 -c:a copy "resized_$filename"
            rm "$file"

            # Execute Scene Splitting
            # Detect scenes
            scene-detect -i "resized_$filename" -t 0.4 -o detections.txt

            # Get times
            scene-time -i detections.txt -o times.txt

            # Split scenes into clips
            scene-cut -i "resized_$filename" -c times.txt 
            
            # Delete resized source file from folder
            rm "resized_$filename"
            
            # Get the number of files in the directory
            num_files=$(ls -1 | wc -l)
            # Subtract 2 from the number of files (excluding "." and "..")
            num_files=$((num_files - 2))
            total_clips=$((total_clips + num_files))

            # Return to the original directory
            cd ..
            
            # Rename the directory
            new_dir_name="${filename_no_ext} - ${num_files} Clips"
            mv "$filename_no_ext" "$new_dir_name"
            
        else
            echo "Skipping non-MP4 file: $file"
        fi

    done

    # Return to the parent directory
    cd ..
    
    # Rename the directory to include the total number of clips
    new_dir_name="${dir%/} - ${total_clips} Clips"
    mv "$dir" "$new_dir_name"
    
    : '
    # Mirror all videos for given technique
    type_new="mirrored_${new_dir_name}"
    mkdir -p "$type_new"

    for examples_folder in "$new_dir_name"/*/; do
        examples=$(basename -- "$examples_folder")
        examples_new="${examples}_mirrored"
        mkdir -p "$type_new/$examples_new"

        for video in "$examples_folder"/*.mp4; do
            filename=$(basename -- "$video")
            mirrored_filename="${filename%.*}_mirrored.mp4"
            
            # perform horrizontal flip
            ffmpeg -i "$video" -vf "hflip" "$type_new/$examples_new/$mirrored_filename"
        
        done
    done
    '
done

echo "Videos processed successfully."
