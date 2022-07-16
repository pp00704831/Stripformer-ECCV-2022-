# Stripformer: Strip Transformer for Fast Image Deblurring (ECCV 2022 Oral)
Pytorch Implementation of "[Stripformer: Strip Transformer for Fast Image Deblurring](https://arxiv.org/abs/2204.04627)"

<img src="./Figure/Intra_Inter.PNG" width = "1000" height = "250" div align=center />

## Installation
The implementation of our BANet is modified from "[DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)"
```
git clone https://github.com/pp00704831/Stripformer.git
cd Stripformer
conda create -n Stripformer python=3.6
source activate Stripformer
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python tqdm pyyaml joblib glog scikit-image tensorboardX albumentations==1.1.0
pip install -U albumentations[imgaug]
```
