# https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md
from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model, MMPoseInferencer, visualize
from mmpose.structures import merge_data_samples
from mmpose.visualization import PoseLocalVisualizer

img_path = 'poseTracking/example.jpg'
# img_path = '../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_23.mp4'
config_file = 'mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py'
checkpoint_file ='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth'

model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# please prepare an image with person
batch_results = inference_topdown(model, img_path)
print(batch_results[0].pred_instances['keypoints'])
print(batch_results)

# # merge results as a single data sample
# results = merge_data_samples(batch_results)

# img = imread(img_path, channel_order='rgb')
# pose_local_visualizer = PoseLocalVisualizer()
# pose_local_visualizer.add_datasample('image', img, results, out_file='out_file.jpg')


# build the inferencer with model alias
# inferencer = MMPoseInferencer('rtmo')

# build the inferencer with model config name
# inferencer = MMPoseInferencer('rtmo-l_16xb16-600e_coco-640x640')

# build the inferencer with model config path and checkpoint path/URL
# inferencer = MMPoseInferencer(
#     pose2d='mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py',
#     pose2d_weights='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth'
# )

# result_generator = inferencer(img_path, out_dir='poseTracking')
# # result = next(result_generator)
# results = [result for result in result_generator]

