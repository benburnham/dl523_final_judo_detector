import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from mmpose.apis import MMPoseInferencer

def obj_assign(frame_list):
    unique_id = 0
    pose_frame_data = []  # Store pose data for each frame
    test_tracks = []  # Store test tracks
    threshold = 50
    
    # Create Kalman filter instance
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= 0.01
    kf.P *= 1000

    # # All frames
    # print(len(detections))

    # # frame 1
    # print(len(detections[0]))

    # # frame 1 predictions
    # print(len(detections[0]['predictions']))

    # # frame 1 prediction keypoint lists
    # print(len(detections[0]['predictions'][0]))

    # # frame 1 prediction keypoint list 1
    # print(len(detections[0]['predictions'][0][0]))

    # # frame 1 prediction keypoint list 1,
    # keypoints_details = detections[0]['predictions'][0][0]
    # print(keypoints_details['keypoints'])
    # print(keypoints_details['keypoints_scores'])
    # print(keypoints_details['bbox'])
    # print(keypoints_details['bbox_score'])
    
    # Process first frame
    for pose in frame_list[0]['predictions'][0]:
        keypoints = np.array(pose['keypoints'])
        box = pose['bbox'][0]
        box_x = np.abs(box[0] - box[2])
        box_y = np.abs(box[1] - box[3])
        box_size = box_x * box_y
        averages = np.average(keypoints, axis=0)
        x = averages[0]
        y = averages[1]

        # reshape to 2,1
        # ValueError: could not broadcast input array from shape (2,) into shape (2,1)
        # kf.x[:2] = [x, y]
        kf.x[:2] = np.reshape(averages, (2,1))
        kf.x[2:] = np.reshape(np.array([0, 0]), (2,1))

        pose_frame_data.append({
            'id': unique_id,
            'kalman_filter': kf,    # do we need a filter for each point + average?
            'last_dist': 0,
            'last_obs': 0,
            'keypoints': keypoints,
            'box_size': box_size
        })

        test_tracks.append({
            'track_id': unique_id,
            # 'first_frame': frame['frame_id'],
            'averages': np.array([averages]),
            'keypoints': np.array([keypoints]),
            'total_box_size': box_size
        })

        unique_id += 1

    # every frame afterwards
    for frame in frame_list[1:]:
        frame_dict_instance = frame['predictions'][0]
        
        for pose_data in pose_frame_data:
            pose_data['kalman_filter'].predict()

        # Get distance from every object to every known car
        delta = np.zeros((len(frame_dict_instance), len(pose_frame_data)))
        for obj_idx, obj in enumerate(frame_dict_instance):
            keypoints = np.array(obj['keypoints'])
            averages = np.average(keypoints, axis=0)
            x = averages[0]
            y = averages[1]
            for pose_idx, pose_data in enumerate(pose_frame_data):
                # DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future.
                # Ensure you extract a single element from your array before performing this operation.
                delta[obj_idx, pose_idx] = np.sqrt((x - pose_data['kalman_filter'].x[0]) ** 2 + (y - pose_data['kalman_filter'].x[1]) ** 2)

        # Compute Hungarian assignment
        obj_ind, pose_ind = linear_sum_assignment(delta)

        # Assign IDs based on result
        for obj_idx, pose_idx in zip(obj_ind, pose_ind):
            if delta[obj_idx, pose_idx] < threshold:
                pose_data = pose_frame_data[pose_idx]
                keypoints = np.array(frame_dict_instance[obj_idx]['keypoints'])
                box_size = np.abs(frame_dict_instance[obj_idx]['bbox'][0][0] - frame_dict_instance[obj_idx]['bbox'][0][2]) * \
                        np.abs(frame_dict_instance[obj_idx]['bbox'][0][1] - frame_dict_instance[obj_idx]['bbox'][0][3])
                x = np.average(keypoints[:, 0])
                y = np.average(keypoints[:, 1])

                pose_data['kalman_filter'].update([x, y])
                pose_data['last_dist'] = delta[obj_idx, pose_idx]
                # pose_data['last_obs'] = frame['frame_id']
                pose_data['keypoints'] = keypoints
                pose_data['box_size'] = box_size

                for track in test_tracks:
                    if track['track_id'] == pose_data['id']:
                        track['averages'] = np.append(track['averages'], [[x, y]], axis=0)
                        track['keypoints'] = np.append(track['keypoints'], [keypoints], axis=0)
                        track['total_box_size'] += box_size
                        break
            else:
                unique_id += 1
                keypoints = np.array(frame_dict_instance[obj_idx]['keypoints'])
                box_size = np.abs(frame_dict_instance[obj_idx]['bbox'][0][0] - frame_dict_instance[obj_idx]['bbox'][0][2]) * \
                        np.abs(frame_dict_instance[obj_idx]['bbox'][0][1] - frame_dict_instance[obj_idx]['bbox'][0][3])
                averages = np.average(keypoints, axis=0)
                x = averages[0]
                y = averages[1]

                kf = KalmanFilter(dim_x=4, dim_z=2)
                kf.F = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
                kf.H = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]])
                kf.R *= 0.01
                kf.P *= 1000
                kf.x[:2] = np.reshape(averages, (2,1))
                kf.x[2:] = np.reshape(np.array([0, 0]), (2,1))

                pose_frame_data.append({
                    'id': unique_id,
                    'kalman_filter': kf,
                    'last_dist': delta[obj_idx, pose_idx],
                    # 'last_obs': frame['frame_id'],
                    'keypoints': keypoints,
                    'box_size': box_size
                })

                test_tracks.append({
                    'track_id': unique_id,
                    # 'first_frame': frame['frame_id'],
                    'averages': np.array([averages]),
                    'keypoints': np.array([keypoints]),
                    'total_box_size': box_size
                })
                
    return pose_frame_data #, test_tracks



# Main function
def main():
    # cap = cv2.VideoCapture('../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_111.mp4')  # Replace 'input_video.mp4' with your video file
    # output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # out = cv2.VideoWriter('KM_demo_6.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

    inferencer = MMPoseInferencer(
        pose2d='rtmo',
        det_model='mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py',
        det_weights='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth',
        det_cat_ids=[0],  # the category id of 'human' class
    )

    detection_generator = inferencer('Test_Video.mp4', out_dir='poseTracking')
    detections = [result for result in detection_generator]
    # print(detections)

    # All frames
    print(len(detections))

    # frame 1
    print(len(detections[0]))

    # frame 1 predictions
    print(len(detections[0]['predictions']))

    # frame 1 prediction keypoint lists
    print(len(detections[0]['predictions'][0]))

    # frame 1 prediction keypoint list 1
    print(len(detections[0]['predictions'][0][0]))

    # frame 1 prediction keypoint list 1,
    keypoints_details = detections[0]['predictions'][0][0]
    print('keypoints')
    print(keypoints_details['keypoints'])
    print('keypoint_scores')
    print(keypoints_details['keypoint_scores'])
    print('bbox')
    print(keypoints_details['bbox'])
    print('bbox_score')
    print(keypoints_details['bbox_score'])

    print(len(obj_assign(detections)))

if __name__ == "__main__":
    main()
