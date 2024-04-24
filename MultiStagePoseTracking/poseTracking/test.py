# # Check Pytorch installation
# import torch, torchvision

# print('torch version:', torch.__version__, torch.cuda.is_available())
# print('torchvision version:', torchvision.__version__)

# # Check MMPose installation
# import mmpose

# print('mmpose version:', mmpose.__version__)

# # Check mmcv installation
# from mmcv.ops import get_compiling_cuda_version, get_compiler_version

# print('cuda version:', get_compiling_cuda_version())
# print('compiler information:', get_compiler_version())

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# please prepare an image with person
results = inference_topdown(model, 'demo.jpg')