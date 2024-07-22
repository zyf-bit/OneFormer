# OneFormer Colab Deepdive
原项目地址https://praeclarumjj3.github.io/oneformer/

# 文件介绍

demo_TJ.py用于跑同济4d数据集中的图片,demo_vod.py用于跑vod数据集中的图片。

本次实验用的是OneFormer项目对于cityscapes数据集的预训练模型。目标与id的对应关系如下https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

根据感兴趣的目标不同，须在demo_TJ或demo_vod中将其中的id更改分类。

### oneformer环境配置

ubuntu-20.04

python3.8.19

cuda11.3
```
1. Clone OneFormer Repo
git clone https://github.com/SHI-Labs/OneFormer-Colab.git

cd OneFormer-Colab/

//之后手动安装requirements.txt中的依赖
```
```
2. Install Dependencies.

git clone 'https://github.com/facebookresearch/detectron2'

pip3 install -r requirements.txt

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip3 install -U opencv-python

pip3 install natten==0.14.6+torch1101cu113 -f https://shi-labs.com/natten/wheels

pip3 install git+https://github.com/cocodataset/panopticapi.git

pip3 install git+https://github.com/mcordts/cityscapesScripts.git

pip3 install ipython-autotime

pip3 install imutils
```
