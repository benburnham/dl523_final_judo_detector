# dl523_final_judo_detector
Judo Throw Detector - EC 523 Final Project


Judo is a modern martial art, combat sport, and Olympic event that originated in Japan in 1882 by Jigoro Kano. The word "Judo" translates to "the gentle way" in Japanese, emphasizing the principle of using an opponent's force against them rather than confronting it with one's own force. Our task is to provide new Judo competitors a tool to evaluate their opponents and see how frequently they use specific techniques. This will enable combatants to focus their game plan and counteract the specific techniques their opponent likes to use.

Dataset here: https://drive.google.com/drive/folders/1QwVl_AXxCmmvkzQyR24HpUSy3EO5CTU7?usp=sharing


The MultiStagePoseTracking folder contains both the "Pose Detection" and "Pose Tracking" models. The Pose Tracking files are all denoted with a "km*"
To run this, installation of the MMPose moodule is required, as per their website. 
https://mmpose.readthedocs.io/en/latest/installation.html

One of the modules did not install correctly on the current pytorch, torchvision, torchaudio version, and downgrading may be necessary for you. This can be done with the following command in conda 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

Both the Numpy and Scipy packages are required to run MultiStagePoseTracking

After installation, running either "judo_final" for Pose Detection or "km_judo_final" will run the model. 
Note, "judo_final" was last run in a linux environment, and will need 'if __name__ == "__main__":' added to run in a Windows environment


Other models in their respective folders can be open and run in Google Colab

