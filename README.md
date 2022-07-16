# Stripformer: Strip Transformer for Fast Image Deblurring (ECCV 2022 Oral)
Pytorch Implementation of "[Stripformer: Strip Transformer for Fast Image Deblurring](https://arxiv.org/abs/2204.04627)"

<img src="./Figure/Intra_Inter.PNG" width = "800" height = "200" div align=center />

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


## Training
Download "[GoPro](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" dataset into './datasets' </br>
For example: './datasets/GoPro/train/blur/\*\*/\*.png'

**We train our Stripformer in two stages:** </br>
**1) We pre-train Stripformer for 3000 epochs on patch size 256x256. Please run the following commands.** </br>
```
python pretrained.py
```

**2) After stage 1, we keep training Stripformer for 1000 epochs on patch size 512x512. Please run the following commands.** </br>
```
python train.py
```

## Testing
For reproducing our results on GoPro and HIDE dataset, download the "[Stripformer_gopro.pth](https://drive.google.com/file/d/1pqK-L-A2FpFtJtvv2Ef6_vapjH9_IBzV/view?usp=sharing)"

For reproducing our results on RealBlur dataset, download "[Stripformer_realblur_J.pth](https://drive.google.com/file/d/1n6SRXmv4ZXgLiF5ZcfRdA0HGdvg-tJQk/view?usp=sharing)" and "[Stripformer_realblur_R.pth](https://drive.google.com/file/d/1dtFCNrEK3WFvKHxIOVtichycH89UXh0E/view?usp=sharing)"

* For testing on GoPro test set </br>
Download "[GoPro](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" full dataset or test set into './datasets' </br>
For example: './datasets/GoPro/test/blur/\*\*/\*.png'
```
python predict_GoPro_test_results --weights_path ./Stripformer_gopro.pth 
```
* For testing on HIDE dataset </br>
Download "[HIDE](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" into './datasets' </br>
```
python predict_HIDE_results --weights_path ./Stripformer_gopro.pth 
```
* For testing on RealBlur test sets
Download "[RealBlur_J](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" and "[RealBlur_R](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" into './datasets' </br>
```
python predict_RealBlur_J_test_results --weights_path ./Stripformer_realblur_J.pth 
```
```
python predict_RealBlur_R_test_results --weights_path ./Stripformer_realblur_R.pth 
```

* For testing your own training weight (take GoPro for a example) 
1) Rename the path in line 23 in the predict_GoPro_test_results.py </br>
2) Chage command to --weights_path ./final_Stripformer_gopro.pth when testing

## Evaluation
* For evaluation on GoPro results in MATLAB, download "[Stripformer_GoPro_results](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" into './out'
```
evaluation_GoPro.m
```
* For evaluation on HIDE results in MATLAB, download "[Stripformer_HIDE_results](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" into './out'
```
evaluation_HIDE.m
```
* For evaluation on RealBlur_J results, download "[Stripformer_realblur_J_results](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" into './out'
```
python evaluate_RealBlur_J.py
```
* For evaluation on RealBlur_R results, download "[Stripformer_realblur_R_results](https://drive.google.com/drive/folders/1AlGIJZBsTzH5jdcouHlHIUx_vZgE6EMC?usp=sharing)" into './out'
```
python evaluate_RealBlur_R.py
```
