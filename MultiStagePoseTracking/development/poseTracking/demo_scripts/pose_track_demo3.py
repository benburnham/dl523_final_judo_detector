import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from mmpose.apis import MMPoseInferencer


# Initialize Kalman Filter parameters

class KMfilter():
    def __init__(self):
        self.step = 8./250.
        self.A = np.array([[1, self.step, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, self.step],[ 0, 0 ,0, 1]])
        self.B = 0
        self.H = np.array([[1, 0, 0, 0],[ 0, 0, 1, 0]])
        self.c_w = np.array([0.5]).reshape(1,1)

    # 1. State prediction
    # xhat[t|t − 1] = A xhat[t−1|t−1]
    def xhat(self, xprior):
        newxhat = self.A @ xprior
        # print("newxhat ", newxhat)
        return newxhat

    # 2. MSE Prediction:
    # M[t|t−1] = A M[t−1|t−1]A^T + BCqB^T
    def MSEhat(self, mseprior):
        msepost = self.A @ mseprior @ self.A.T + .05
        # print("msepost ", msepost)
        return msepost

    # 3. Kalman Gain Computation:
    # K[t] = M [t|t − 1]H^T [t] (Cw[t] + H[t] M [t|t − 1]H^T [t])^−1
    # assuming H = [ 1 0 0 0 ; 0 0 1 0 ] (revisit if necessary)
    # assume cW[t] = 0.5
    def KGC(self, mseprior):
        self.k = mseprior @ self.H.T @ (np.linalg.inv(self.c_w + self.H @ mseprior @ self.H.T))
        # print("KGC ", self.k)
        return self.k

    # 4. State Estimation (= Correction):
    # xhat[t|t] = xhat[t|t − 1] + K[t] (z[t] − H[t] xhat[t|t − 1])
    def xhat_estimate(self, xprior, k, measurement):
        # print(xprior, k, measurement)
        xhat_estimate = xprior + k @ ( measurement - self.H @ xprior )
        # print("state estimate ", xhat_estimate)
        return xhat_estimate

    # 5. MSE Estimation:
    # M[t|t] = (1 − K[t]) H[t] M[t|t − 1]
    def MSE_estimate(self, mseprior, k):
        MSE_estimate = (1 - k ) @ self.H @ mseprior
        return MSE_estimate

# DK Adjustments
def obj_assign(frame_dict):
    unique_id = 0
    car_history = []
    threshold = 50
    km = KMfilter()
    car_frame_data = []
    test_tracks = []

    big_identity = np.array([np.identity(4), np.identity(4), np.identity(4), np.identity(4),
                                np.identity(4), np.identity(4), np.identity(4), np.identity(4),
                                np.identity(4), np.identity(4), np.identity(4), np.identity(4),
                                np.identity(4), np.identity(4), np.identity(4), np.identity(4),
                                np.identity(4)])

    # Iterate through all frames
    for frame in frame_dict:
        frame_dict_instance = frame['instances']

        # Give IDs to objects in first frame
        if frame['frame_id'] == 0:
            for pose in frame_dict_instance:
                # Save new ID to JSON array
                # ?

                # change to average pose points x & y to use for calculations
                keypoints = np.array(pose['keypoints'])
                # print("keypoints")
                # print(keypoints)
                box = pose['bbox']
                box = box[0]
                # print("box")
                # print(box)
                box_x = np.absolute(box[0] - box[2])
                box_y = np.absolute(box[1] - box[3])
                box_size = box_x * box_y
                # print(box_size)
                averages = np.average(keypoints, axis=0)
                x = averages[0]
                y = averages[1]
                # print("averages")
                # print(averages)
                # print(x)
                # print(y)

                # Create new car for the ID
                car_history.append([unique_id,
                                   x,
                                   y,
                                   frame['frame_id'],   # 3 Frame
                                   0,       # 4 Last dist
                                   0,       # 5 Last obs
                                   np.array([[x],[0],[y],[0]]),       # 6 state_prior
                                   np.identity(4),  # 7 mse_prior
                                   0,       # 8 kalman gain
                                   keypoints, # 9 pose keypoints pos prior
                                   np.zeros(keypoints.shape), # 10 pose keypoints vel prior
                                   big_identity]) # 11 mse keypoints prior
                test_tracks.append({'track_id': unique_id,
                                    'first_frame': frame['frame_id'],
                                    'averages': np.array([averages]),
                                    'keypoints': np.array([keypoints]),
                                    'last_box_size': box_size,
                                    'total_box_size': box_size})

                unique_id += 1  # step ID counter
            # print(car_history)
            # print()
            # print("test_tracks")
            # print(test_tracks)
            # print()
            # print("setup finished")
            # print()

        # Evaluate from second frame on
        else:
            # Run prediction for all known cars
            for i, car in enumerate(car_history):
                car_history[i][6] = km.xhat(car[6])     # state_prior
                car_history[i][7] = km.MSEhat(car[7])   # mse_prior
                car_history[i][8] = km.KGC(car[7])      # kalman gain

            # Get distance from every object to every known car
            delta = np.zeros((len(frame_dict_instance), len(car_history)))
            for obj in range(len(frame_dict_instance)):
                keypoints = np.array(frame_dict_instance[obj]['keypoints'])
                averages = np.average(keypoints, axis=0)
                x = averages[0]
                y = averages[1]
                # print("averages")
                # print(averages)
                # print(x)
                # print(y)
                for id, car in enumerate(car_history):
                    # compute delta for every object / ID pair
                    delta[obj,id] = np.sqrt((x-car[6][0])**2+(y-car[6][2])**2)
                    #print(delta[obj,id])

            # Create an nxn cost Matrix
            row_delta, col_delta = np.shape(delta)
            if row_delta > col_delta:
                # adding new car
                ones_array = 900*np.ones((row_delta, row_delta - col_delta))
                delta = np.hstack((delta, ones_array))
            elif row_delta < col_delta:
                # adding fake object
                ones_array = 1000*np.ones((col_delta - row_delta, col_delta))
                delta = np.vstack((delta, ones_array))
            #print(delta)

            # Compute Hungarian assigment
            obj_ind, car_ind = scipy.optimize.linear_sum_assignment(delta)
            # print("frame",frame, obj_ind, car_ind)

            # Assign IDs based on result
            for obj in obj_ind:
                # print(delta[obj,car_ind[obj]])

                # within threshold, tagging with id and updating km filter
                if delta[obj,car_ind[obj]] < threshold:
                    # Get ID
                    car = car_history[car_ind[obj]]
                    # print("within threshold, tagging with id and updating km filter")
                    # print("car")
                    # print(car)

                    # Save ID to JSON array
                    # frame_dict[str(frame)][obj]['id'] = car[0]

                    # Get x and y for object (measurement)
                    keypoints = np.array(frame_dict_instance[obj]['keypoints'])
                    box = frame_dict_instance[obj]['bbox']
                    box = box[0]
                    box_x = np.absolute(box[0] - box[2])
                    box_y = np.absolute(box[1] - box[3])
                    box_size = box_x * box_y
                    averages = np.average(keypoints, axis=0)
                    # print(averages)
                    x = averages[0]
                    y = averages[1]

                    # Update kalman filter
                    car[6] = km.xhat_estimate(car[6], car[8], np.array([[x],[y]]))
                    car[7] = km.MSE_estimate(car[7], car[8])
                    car[1] = car[6][0]
                    car[2] = car[6][2]

                    # update kalman pose keypoints
                    last_keypoints = car[9]
                    last_vel = car[10]
                    last_mse = car[11]
                    new_keypoints = np.zeros(last_keypoints.shape)
                    new_vel = np.zeros(last_vel.shape)
                    new_mse = np.zeros(last_mse.shape)
                    for i in range(17):
                        kp_x = last_keypoints[i][0]
                        kp_y = last_keypoints[i][1]
                        vel_x = last_vel[i][0]
                        vel_y = last_vel[i][1]
                        car_state = np.array([kp_x, vel_x, kp_y, vel_y]) # create array of x xvel y yvel
                                    # [9][i][0] [10][i][0] [9][i][1] [10][i][1]
                        car_state = km.xhat_estimate(car_state, car[8], np.array([x, y]))
                        new_mse[i] = km.MSE_estimate(last_mse[i], car[8])
                        new_keypoints[i][0] = car_state[0]
                        new_keypoints[i][1] = car_state[2]
                        new_vel[i][0] = car_state[1]
                        new_vel[i][1] = car_state[3]
                    car[9] = new_keypoints
                    car[10] = new_vel
                    car[11] = new_mse

                    # print("old keypoints")
                    # print(last_keypoints)
                    # print(last_vel)
                    # print(last_mse)
                    # print("new keypoints")
                    # print(new_keypoints)
                    # print(new_vel)
                    # print(new_mse)
                    # print()

                    # Save update to car history array
                    car_history[car_ind[obj]] = car
                    # print()
                    # print("test_tracks[car[0]]['averages']")
                    # print(test_tracks[car[0]]['averages'])
                    # print("np.array([averages])")
                    # print(np.array([averages]))
                    test_tracks[car[0]]['averages'] = np.append(test_tracks[car[0]]['averages'], np.array([averages]), axis=0)
                    test_tracks[car[0]]['keypoints'] = np.append(test_tracks[car[0]]['keypoints'], np.array([keypoints]), axis=0)
                    test_tracks[car[0]]['last_box_size']= box_size
                    test_tracks[car[0]]['total_box_size']= test_tracks[car[0]]['total_box_size'] + box_size
                    # print(test_tracks[car[0]]['averages'])

                # Spare car, update kalman filter
                elif delta[obj,car_ind[obj]] == 1000:
                    # print("Spare car, update kalman filter")
                    # print("car")
                    # print(car)
                    # Get ID
                    car = car_history[car_ind[obj]]

                    # Use prediction as measurement
                    x = car[6][0]
                    y = car[6][2]

                    # Update kalman filter average
                    car[6] = km.xhat_estimate(car[6], car[8], np.array([x,y]))
                    car[7] = km.MSE_estimate(car[7], car[8])
                    car[1] = car[6][0]
                    car[2] = car[6][2]

                    # update kalman pose keypoints
                    last_keypoints = car[9]
                    last_vel = car[10]
                    last_mse = car[11]
                    new_keypoints = np.zeros(last_keypoints.shape)
                    new_vel = np.zeros(last_vel.shape)
                    new_mse = np.zeros(last_mse.shape)
                    for i in range(17):
                        kp_x = last_keypoints[i][0]
                        kp_y = last_keypoints[i][1]
                        vel_x = last_vel[i][0]
                        vel_y = last_vel[i][1]
                        car_state = np.array([kp_x, vel_x, kp_y, vel_y]) # create array of x xvel y yvel
                                    # [9][i][0] [10][i][0] [9][i][1] [10][i][1]
                        car_state = km.xhat_estimate(car_state, car[8], np.array([kp_x,kp_y]))
                        new_mse[i] = km.MSE_estimate(last_mse[i], car[8])
                        new_keypoints[i][0] = car_state[0]
                        new_keypoints[i][1] = car_state[2]
                        new_vel[i][0] = car_state[1]
                        new_vel[i][1] = car_state[3]
                    car[9] = new_keypoints
                    car[10] = new_vel
                    car[11] = new_mse

                    # print("old keypoints")
                    # print(last_keypoints)
                    # print(last_vel)
                    # print(last_mse)
                    # print("new keypoints")
                    # print(new_keypoints)
                    # print(new_vel)
                    # print(new_mse)
                    # print()


                    # Save update to car history array
                    car_history[car_ind[obj]] = car
                    # print()
                    # print("test_tracks[car[0]]['averages']")
                    # print(test_tracks[car[0]]['averages'])
                    # print("np.array([averages])")
                    # print(np.array([averages]))
                    test_tracks[car[0]]['averages'] = np.append(test_tracks[car[0]]['averages'], np.array([averages]), axis=0)
                    test_tracks[car[0]]['keypoints'] = np.append(test_tracks[car[0]]['keypoints'], np.array([new_keypoints]), axis=0)
                    test_tracks[car[0]]['total_box_size']= test_tracks[car[0]]['total_box_size'] + test_tracks[car[0]]['last_box_size'] * .1
                    # print(test_tracks[car[0]]['averages'])

                # Exceeds threshold, create new car
                elif delta[obj,car_ind[obj]] >= threshold:
                    print("Exceeds threshold, create new car")
                    print("car")
                    print(car)
                    # Save new ID to JSON array
                    # frame_dict[str(frame)][obj]['id'] = unique_id

                    # Get objects x and y to initialize new car
                    keypoints = np.array(frame_dict_instance[obj]['keypoints'])
                    averages = np.average(keypoints, axis=0)
                    x = averages[0]
                    y = averages[1]
                    box = pose['bbox']
                    box = box[0]
                    # print("box")
                    # print(box)
                    box_x = np.absolute(box[0] - box[2])
                    box_y = np.absolute(box[1] - box[3])
                    box_size = box_x * box_y
                    # print(box_size)

                    # Create new car for the ID
                    car_history.append([unique_id,
                                        x,
                                        y,
                                        frame['frame_id'],   # 3 Frame
                                        0,       # 4 Last dist
                                        0,       # 5 Last obs
                                        np.array([[x],[0],[y],[0]]),       # 6 state_prior
                                        np.identity(4),  # 7 mse_prior
                                        0,       # 8 kalman gain
                                        keypoints, # 9 pose keypoints pos prior
                                        np.zeros(keypoints.shape), # 10 pose keypoints vel prior
                                        big_identity]) # 11 mse keypoints prior
                    # print("append new track")
                    test_tracks.append({'track_id': unique_id,
                                        'first_frame': frame['frame_id'],
                                        'averages': np.array([averages]),
                                        'keypoints': np.array([keypoints]),
                                        'last_box_size': box_size,
                                        'total_box_size': box_size})

                    # step ID counter
                    unique_id +=1

        # capture car states every frame to track movement
        car_frame_data.append(copy.deepcopy(car_history))

        # DEBUG
        #for car in car_history:
            #print('id:',car[0],'x:',car[1],'y:',car[2])

    # Return JSON array and car data
    return frame_dict, car_frame_data, test_tracks


# Main function
def main():
    cap = cv2.VideoCapture('../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_28.mp4')  # Replace 'input_video.mp4' with your video file
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

    # obj_assign(frame_dict)
    # expects the output for the whole video
    # returns tracks of all pose tracks
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
