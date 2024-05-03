# dl523_final_judo_detector
Judo Throw Detector - EC 523 Final Project


Judo is a modern martial art, combat sport, and Olympic event that originated in Japan in 1882 by Jigoro Kano. The word "Judo" translates to "the gentle way" in Japanese, emphasizing the principle of using an opponent's force against them rather than confronting it with one's own force. Our task is to provide new Judo competitors a tool to evaluate their opponents and see how frequently they use specific techniques. This will enable combatants to focus their game plan and counteract the specific techniques their opponent likes to use. 

Dataset here: https://drive.google.com/drive/folders/1QwVl_AXxCmmvkzQyR24HpUSy3EO5CTU7?usp=sharing

INSTALLATION AND RUNNING:

The MultiStagePoseTracking folder contains both the "Pose Detection" and "Pose Tracking" models. The Pose Tracking files are all denoted with a "km*"

To run this, installation of the numpy and scipy can be done with pip commands, and MMPose moodule can be installed per the instructions on their website. 
https://mmpose.readthedocs.io/en/latest/installation.html

For MMPose, one of the modules did not install correctly on the current pytorch, torchvision, torchaudio version, and downgrading was necessary for us. This can be done with the following command in conda 

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia



After installation, running either "judo_final" for Pose Detection or "km_judo_final" will run the model. 
Note, "judo_final" was last run in a linux environment, and will need 'if __name__ == "__main__":' added to run in a Windows environment

For Pose Detection, Judo_final contains both the model setup and training. 
For Kalman filter, km_judo_final contains the training and km_model contains the model. 

The CNN-LSTM model and training is self contained in a single python file, who's only external dependencies are Pytorch, torchvision and av, which are all able to be installed through pip commands. 

The other two models in their respective folders (GRU and 2_1CNN) can be opened and run in Google Colab

