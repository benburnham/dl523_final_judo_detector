import numpy as np
import cv2
from kalman_filter import KMfilter
from mmpose.apis import MMPoseInferencer
from scipy.optimize import linear_sum_assignment

# Function to calculate Euclidean distance between two keypoints
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to perform object association using Hungarian algorithm
def associate_objects(pred_poses, prev_poses):
    cost_matrix = np.zeros((len(pred_poses), len(prev_poses)))
    for i, pred_pose in enumerate(pred_poses):
        for j, prev_pose in enumerate(prev_poses):
            cost_matrix[i, j] = sum(euclidean_distance(pred_pose[k], prev_pose[k]) for k in range(len(pred_pose)))
    pred_indices, prev_indices = linear_sum_assignment(cost_matrix)
    associations = []
    for pred_idx, prev_idx in zip(pred_indices, prev_indices):
        associations.append((pred_idx, prev_idx))
    return associations

# Function to process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video")
        return

    frame_count = 0
    kalman_filters = []
    prev_poses = None
    inferencer = MMPoseInferencer(
        pose2d='rtmo',
        det_model='mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py',
        det_weights='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth',
        det_cat_ids=[0],  # the category id of 'human' class
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale if needed
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform pose detection on the frame to obtain detected poses
        result = next(inferencer(frame))
        detected_poses = [prediction['keypoints'] for prediction in result['predictions'][0]]

        if prev_poses is not None:
            # Perform object association using Hungarian algorithm
            associations = associate_objects(detected_poses, prev_poses)

            # Update existing Kalman filters
            for pred_idx, prev_idx in associations:
                kalman_filters[prev_idx].x = kalman_filters[prev_idx].xhat(kalman_filters[prev_idx].x)
                kalman_filters[prev_idx].M = kalman_filters[prev_idx].MSEhat(kalman_filters[prev_idx].M)
                kalman_filters[prev_idx].x = kalman_filters[prev_idx].xhat_estimate(kalman_filters[prev_idx].x, kalman_filters[prev_idx].KGC(kalman_filters[prev_idx].M), detected_poses[pred_idx])
                kalman_filters[prev_idx].M = kalman_filters[prev_idx].MSE_estimate(kalman_filters[prev_idx].M, kalman_filters[prev_idx].k)

            # Add new Kalman filters for unassociated keypoints
            unassociated_indices = set(range(len(detected_poses))) - set([assoc[0] for assoc in associations])
            for idx in unassociated_indices:
                new_filter = KMfilter()
                new_filter.x = np.array(detected_poses[idx]).reshape(-1, 1)
                new_filter.M = np.eye(4) * 0.1  # Assuming initial MSE
                kalman_filters.append(new_filter)

        else:
            # Initialize Kalman filters for the first frame
            kalman_filters = [KMfilter() for _ in range(len(detected_poses))]
            for i, pose in enumerate(detected_poses):
                kalman_filters[i].x = np.array(pose).reshape(-1, 1)
                kalman_filters[i].M = np.eye(4) * 0.1  # Assuming initial MSE

        # Store current poses for next frame association
        prev_poses = detected_poses

        frame_count += 1
        if frame_count >= 10:
            print('success')
            break

    cap.release()

# Example usage
video_path = '../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_23.mp4'
process_video(video_path)