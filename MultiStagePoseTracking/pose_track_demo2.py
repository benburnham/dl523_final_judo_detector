import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from mmpose.apis import MMPoseInferencer


# Initialize Kalman Filter parameters
def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0, 0, 0, 0])  # Initial state: [x, y, dx/dt, dy/dt]
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Measurement matrix
    kf.P *= 1000  # Covariance matrix
    kf.R = np.array([[5, 0],
                     [0, 5]])  # Measurement noise
    kf.Q = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # Process noise
    return kf

# Perform Hungarian algorithm for assignment
def hungarian_algorithm(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

# Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Update Kalman filter with detected pose
def update_kalman_filter(kf, detected_pose):
    kf.update(detected_pose)

# Main function
def main():
    cap = cv2.VideoCapture('../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_23.mp4')  # Replace 'input_video.mp4' with your video file
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

    kf_list = []  # List to store Kalman filters for each pose
    pose_id = 0

    inferencer = MMPoseInferencer(
        pose2d='rtmo',
        det_model='mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py',
        det_weights='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth',
        det_cat_ids=[0],  # the category id of 'human' class
    )


    while cap.isOpened():
        ret, frame = cap.read()
        print('frame')
        if not ret:
            break

        # Perform pose detection on the frame to obtain detected poses
        result = next(inferencer(frame))
        detected_poses = [prediction['keypoints'] for prediction in result['predictions'][0]]
        
        if len(detected_poses) > 0:

            if len(kf_list) == 0:
                # Initialize Kalman filters for detected poses
                for pose in detected_poses:
                    kf_list.append(initialize_kalman_filter())
            else:
                # Perform Kalman filter prediction step for each filter
                for kf in kf_list:
                    kf.predict()

                # Calculate cost matrix for Hungarian algorithm
                cost_matrix = np.zeros((len(kf_list), len(detected_poses)))
                for i, kf in enumerate(kf_list):
                    for j, pose in enumerate(detected_poses):
                        # Calculate Euclidean distance between predicted and detected pose keypoints
                        cost_matrix[i, j] = euclidean_distance(kf.x[:2], pose[0])

                # Perform Hungarian algorithm for assignment
                row_ind, col_ind = hungarian_algorithm(cost_matrix)

                # Update Kalman filters with assigned detections
                for i, j in zip(row_ind, col_ind):
                    update_kalman_filter(kf_list[i], detected_poses[j][0])

                # Add new Kalman filters for unassigned detections
                unassigned_detections = set(range(len(detected_poses))) - set(col_ind)
                for j in unassigned_detections:
                    kf_list.append(initialize_kalman_filter())
                    update_kalman_filter(kf_list[-1], detected_poses[j][0])

        # Visualization logic (Drawing detected poses and IDs)
        for i, kf in enumerate(kf_list):
            predicted_pose = kf.x[:2].astype(int)
            cv2.circle(frame, tuple(predicted_pose), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'ID: {i}', tuple(predicted_pose), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)  # Write frame to output video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
