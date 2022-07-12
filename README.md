# AIRR
Official code of Supervised Attribute Information Removal and Reconstruction for Image Manipulation

## Dependencies
Our code is built on Python3, Pytorch 1.11 and CUDA 11.3. 

## Data
1. Download preprocessed [annotation files](https://drive.google.com/file/d/1Cs30Ny-hn1zi5DmH8bWcQT-5I03gA4QA/view?usp=sharing), including parsing maps of Deepfashion Fine-Grained Attribute and CelebA. Unzip it and put the `data` folder under the current directory.
2. Download and unzip [Deepfashion Synthesis](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html). Put the unzippped `FashionSynthesisBenchmark/` folder under `data/synthesis/`.
3. Download and unzip the original [Deepfashion Fine-Grained Attribute annotations](https://drive.google.com/drive/folders/19J-FY5NY7s91SiHpQQBo2ad3xjIB42iN) and [imgs.zip](https://drive.google.com/drive/folders/0B7EVK8r0v71pekpRNUlMS3Z5cUk?resourcekey=0-GHiFnJuDTvzzGuTj6lE6og). Put these files under `data/attr/`. Run `create_deepfashion_finegrained.py` to resize all images to 224x224.
4. Download and unzip aligned face images from [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?select=img_align_celeba). Put the unzippped `img_align_celeba/` folder under `data/celeba/`.
5. Download and unzip high resolution face images from [CelebA-HQ](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view). Put the unzipped `CelebAMask-HQ` folder under `data/celebahq`.

## Pretrained models
Download and unzip the pretrained [attribute classifier](https://drive.google.com/file/d/1CkdUdBlWewvNz5HkA-iRD-S3K7LR3sBX/view?usp=sharing) and [AIRR models](https://drive.google.com/file/d/1CphcjjNpYwCDhoK2G5s4YibLE2uF_L-u/view?usp=sharing). Put the unzipped folders under the current directory.

## Train
Run `train.py`.

To train on CelebA-HQ, please clone [pSp](https://github.com/eladrich/pixel2style2pixel) repository to the current directory. You also need to download their pretrained image decoder weights for ffhq.

## Test
Run `test.py`. This should generate all test images with the specified attribute under `save_dir`. Please specify `save_dir`, the dataset and the attribute that you would like to manipulate in `test.py`.

## To Do
- [x] code release
- [x] configure dataset
- [ ] test code
- [x] pretrained model release
