import torch
import torch.nn as nn
import numpy as np
import scipy
import copy

from poseTracking.kalman_filter import KMfilter

class KMJudoTechniqueClassifier(torch.nn.Module):
    def __init__(self, hidden_dim, layer_dim, dropout_rate, num_outputs, device):
        super(KMJudoTechniqueClassifier, self).__init__()
        self.num_outputs = num_outputs
        self.device = device

        # LSTM Claasification
        input_dim = 2   # 2 pose sequences, 1 for each combatant
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            layer_dim, 
                            dropout=dropout_rate,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_outputs)

        # Softmax activation for classification function
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, detections):
        # Get pose sequences
        pose1_seq, pose2_seq = self.obj_assign(detections)

        # Prepare LSTM input using pose sequences
        lstm_input = self.prepare_lstm_input(pose1_seq, pose2_seq)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.layer_dim, lstm_input.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, lstm_input.size(0), self.hidden_dim).to(self.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))

        # Dropout and average the sequence
        lstm_out = self.dropout(lstm_out)
        lstm_out = torch.mean(lstm_out, dim=(0, 1), keepdim=True)

        # Linear layer and return predictions
        predictions = self.fc(lstm_out)
        return predictions
    
    def prepare_lstm_input(self, pose1_seq, pose2_seq):

        # Convert lists to tensors
        # pose1_seq = torch.stack(pose1_seq, axis=0)
        # pose2_seq = torch.stack(pose2_seq, axis=0)

        # Prepare data for LSTM classification
        diff = abs(pose1_seq.shape[0]-pose2_seq.shape[0])
        if len(pose1_seq) > len(pose2_seq):
            pose1_seq = pose1_seq[diff:]
        elif len(pose1_seq) < len(pose2_seq):
            pose2_seq = pose2_seq[diff:]

        pose1_seq = torch.from_numpy(pose1_seq).to(self.device)
        pose2_seq = torch.from_numpy(pose2_seq).to(self.device)
        
        pose_seqs = torch.stack([pose1_seq, pose2_seq], dim=1).float()
        lstm_input = pose_seqs.view(pose_seqs.size(0), -1, 2)
        
        return lstm_input

    def class_to_classID(self, technique):
        technique_to_id = {'Osoto Gari': 0, 'Seoi Nage': 1, 'Uchi Mata': 2}
        class_id = technique_to_id.get(technique, -1)
        if class_id != -1:
            one_hot = torch.zeros(self.num_outputs, device=self.device)
            one_hot[class_id] = 1
            return one_hot.unsqueeze(0)
        else:
            print('technique not in list')
            return None
    
    def classID_to_class(self, predictions):
        # class dictionary
        class_mapping_reverse = {0: 'Osoto Gari', 1: 'Seoi Nage', 2: 'Uchi Mata'}

        # Get technique name for given prediction
        predictions = self.softmax(predictions)
        classID = torch.argmax(predictions).item()
        return class_mapping_reverse[classID]

    def top_two_tracks(self, test_tracks):
        first = 0
        second = 0
        first_keypoints = np.zeros(test_tracks[0]['keypoints'].shape)
        second_keypoints = np.zeros(test_tracks[0]['keypoints'].shape)
        # print("test_tracks[i]['total_box_size']")
        for i in range(len(test_tracks)):
            # print(test_tracks[i]['total_box_size'])
            if test_tracks[i]['total_box_size'] > first:
                second = copy.deepcopy(first)
                second_keypoints = copy.deepcopy(first_keypoints)
                first = test_tracks[i]['total_box_size']
                first_keypoints = test_tracks[i]['keypoints']
            elif test_tracks[i]['total_box_size'] > second:
                second = test_tracks[i]['total_box_size']
                second_keypoints = test_tracks[i]['keypoints']
        return first_keypoints, second_keypoints

    def obj_assign(self, frame_dict):
        # returns 2 most in frame tracks
        # keypoint seq 1, keypoint seq 2
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
                                        frame['frame_id'],  # 3 Frame
                                        0,  # 4 Last dist
                                        0,  # 5 Last obs
                                        np.array([[x], [0], [y], [0]]),  # 6 state_prior
                                        np.identity(4),  # 7 mse_prior
                                        0,  # 8 kalman gain
                                        keypoints,  # 9 pose keypoints pos prior
                                        np.zeros(keypoints.shape),
                                        # 10 pose keypoints vel prior
                                        big_identity])  # 11 mse keypoints prior
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
                    car_history[i][6] = km.xhat(car[6])  # state_prior
                    car_history[i][7] = km.MSEhat(car[7])  # mse_prior
                    car_history[i][8] = km.KGC(car[7])  # kalman gain

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
                        delta[obj, id] = np.sqrt((x - car[6][0]) ** 2 + (y - car[6][2]) ** 2)
                        # print(delta[obj,id])

                # Create an nxn cost Matrix
                row_delta, col_delta = np.shape(delta)
                if row_delta > col_delta:
                    # adding new car
                    ones_array = 900 * np.ones((row_delta, row_delta - col_delta))
                    delta = np.hstack((delta, ones_array))
                elif row_delta < col_delta:
                    # adding fake object
                    ones_array = 1000 * np.ones((col_delta - row_delta, col_delta))
                    delta = np.vstack((delta, ones_array))
                # print(delta)

                # Compute Hungarian assigment
                obj_ind, car_ind = scipy.optimize.linear_sum_assignment(delta)
                # print("frame", frame, obj_ind, car_ind)

                # Assign IDs based on result
                for obj in obj_ind:
                    # print(delta[obj,car_ind[obj]])

                    # within threshold, tagging with id and updating km filter
                    if delta[obj, car_ind[obj]] < threshold:
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
                        car[6] = km.xhat_estimate(car[6], car[8], np.array([[x], [y]]))
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
                            car_state = np.array(
                                [kp_x, vel_x, kp_y, vel_y])  # create array of x xvel y yvel
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
                        test_tracks[car[0]]['averages'] = np.append(
                            test_tracks[car[0]]['averages'], np.array([averages]), axis=0)
                        test_tracks[car[0]]['keypoints'] = np.append(
                            test_tracks[car[0]]['keypoints'], np.array([keypoints]), axis=0)
                        test_tracks[car[0]]['last_box_size'] = box_size
                        test_tracks[car[0]]['total_box_size'] = test_tracks[car[0]][
                                                                    'total_box_size'] + box_size
                        # print(test_tracks[car[0]]['averages'])

                    # Spare car, update kalman filter
                    elif delta[obj, car_ind[obj]] == 1000:
                        # print("Spare car, update kalman filter")
                        # print("car")
                        # print(car)
                        # Get ID
                        car = car_history[car_ind[obj]]

                        # Use prediction as measurement
                        x = car[6][0]
                        y = car[6][2]

                        # Update kalman filter average
                        car[6] = km.xhat_estimate(car[6], car[8], np.array([x, y]))
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
                            car_state = np.array(
                                [kp_x, vel_x, kp_y, vel_y])  # create array of x xvel y yvel
                            # [9][i][0] [10][i][0] [9][i][1] [10][i][1]
                            car_state = km.xhat_estimate(car_state, car[8],
                                                         np.array([kp_x, kp_y]))
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
                        averages = np.average(new_keypoints, axis=0)
                        test_tracks[car[0]]['averages'] = np.append(
                            test_tracks[car[0]]['averages'], np.array([averages]), axis=0)
                        test_tracks[car[0]]['keypoints'] = np.append(
                            test_tracks[car[0]]['keypoints'], np.array([new_keypoints]), axis=0)
                        test_tracks[car[0]]['total_box_size'] = test_tracks[car[0]][
                                                                    'total_box_size'] + \
                                                                test_tracks[car[0]][
                                                                    'last_box_size'] * .1
                        # print(test_tracks[car[0]]['averages'])

                    # Exceeds threshold, create new car
                    elif delta[obj, car_ind[obj]] >= threshold:
                        # print("Exceeds threshold, create new car")
                        # print("car")
                        # print(car)
                        # Save new ID to JSON array
                        # frame_dict[str(frame)][obj]['id'] = unique_id

                        # Get objects x and y to initialize new car
                        keypoints = np.array(frame_dict_instance[obj]['keypoints'])
                        averages = np.average(keypoints, axis=0)
                        x = averages[0]
                        y = averages[1]
                        box = frame_dict_instance[obj]['bbox']
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
                                            frame['frame_id'],  # 3 Frame
                                            0,  # 4 Last dist
                                            0,  # 5 Last obs
                                            np.array([[x], [0], [y], [0]]),  # 6 state_prior
                                            np.identity(4),  # 7 mse_prior
                                            0,  # 8 kalman gain
                                            keypoints,  # 9 pose keypoints pos prior
                                            np.zeros(keypoints.shape),
                                            # 10 pose keypoints vel prior
                                            big_identity])  # 11 mse keypoints prior
                        # print("append new track")
                        test_tracks.append({'track_id': unique_id,
                                            'first_frame': frame['frame_id'],
                                            'averages': np.array([averages]),
                                            'keypoints': np.array([keypoints]),
                                            'last_box_size': box_size,
                                            'total_box_size': box_size})

                        # step ID counter
                        unique_id += 1

            # capture car states every frame to track movement
            car_frame_data.append(copy.deepcopy(car_history))

            # DEBUG
            # for car in car_history:
            # print('id:',car[0],'x:',car[1],'y:',car[2])

        # print(test_tracks)
        first_keypoints, second_keypoints = self.top_two_tracks(test_tracks)

        # Return JSON array and car data
        return first_keypoints, second_keypoints
