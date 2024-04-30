# Check Pytorch installation
import torch, torchvision

print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)

# Check MMPose installation
import mmpose

print('mmpose version:', mmpose.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print('cuda version:', get_compiling_cuda_version())
print('compiler information:', get_compiler_version())

# from mmpose.apis import inference_topdown, init_model
# from mmpose.utils import register_all_modules

# register_all_modules()

# config_file = 'mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py'
# checkpoint_file = 'poseTracking/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth'
# model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# # please prepare an image with person
# results = inference_topdown(model, 'poseTracking/example.jpg', vis_out_dir='poseTracking/vis_results.jpg')
# print(results)


# https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md
from mmpose.apis import MMPoseInferencer
# img_path = 'poseTracking/example.jpg'
img_path = '../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_10.mp4'


# build the inferencer with model alias
# inferencer = MMPoseInferencer('rtmo')

# build the inferencer with model config name
# inferencer = MMPoseInferencer('rtmo-l_16xb16-600e_coco-640x640')

# build the inferencer with model config path and checkpoint path/URL
inferencer = MMPoseInferencer(
    pose2d='mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py',
    pose2d_weights='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth'
)

result_generator = inferencer(img_path, out_dir='poseTracking')
# result = next(result_generator)
results = [result for result in result_generator]

