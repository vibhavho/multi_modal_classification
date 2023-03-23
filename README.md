#Understanding Multi-Modal Speaker Recognition via Disentangled Representation Learning
This repository houses the official PyTorch implementation of USC's EE 641 (Deep Learning System) final project.

Abstract
The primary target in supervised machine learning is to model the discriminative probability p(y|x). However, in the process of doing so, data samples often contain "irrelevant" factors that could mislead the model in the estimation. In this work, we address the problem of speaker recognition (loosely speaker classification) from a multi-modal (audio-visual) point of view and present an adversarial invariance approach, and propose a feature extraction pipeline, to achieve the same. Experimental results reveal that the multimodal learning setup achieves a relative improvement of 25.5% in classification accuracy over using x-vectors only and about 0.048% over using the frame-level features only, as the input. We achieve even superior results when the video frames are augmented with Gaussian noise, which makes the adversarial network even more robust to these irrelevant factors, that could have degraded the speaker recognition performance.

Installation
$ git clone https://github.com/sarthaxxxxx/EE641-Project.git
$ cd EE641-Project/
$ pip install requirements.txt
Pre-processing (frame-level extraction from videos)
Configuring ./preprocess/config.yaml
To train the 2DCNN + LSTM based autoencoder, model: convlstm. The resized size of the frames remains the same. However, if you wish to change that, please configure the model architecture as well for reconstruction. You're free to experiment with model: 3dcnn, as well. Set 'data' to your data directory. 'processed' refers to the directory where all the extracted features will be saved, as per the subject id. You're free to configure 'data_aug' as well, if data augmentation to the video frames are needed. The best model will be saved in ./preprocess/models/ckpt/ .

Feature extraction (training) and feature generation (inference) (single GPU)
python3 preprocess/train.py --config (path to the preprocessing config file in your system) --gpu (gpu_id)
python3 preprocess/generate_features.py --config (path to the preprocessing config file in your system) --gpu (gpu_id)
Training and Inference (multimodal / xvec / frame-level input to UAI) (single GPU)
Configuring config.yaml
Set the data directory to ./data/processed/ and model checkpoints accordingly. Change type to 'xvec'/'visual'/'multimodal' depending on the training mode and results you desire. You're free to configure to experiment with different parameters.

python3 train.py --config (path to the main config file in your system) --gpu (gpu_id)
python3 predict.py --config (path to the main config file in your system) --gpu (gpu_id)
Presentation and Poster
Presentation
Report

License
MIT License

Contacts
Vibhav Hosahalli Venkataramaiah - email: vibhavho@usc.edu
Sarthak Kumar Maharana - email: maharana@usc.edu
Gopi Maganti - email: gmaganti@usc.edu
