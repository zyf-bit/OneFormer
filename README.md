# OneFormer Colab Deepdive
原项目地址https://praeclarumjj3.github.io/oneformer/

### oneformer环境配置

ubuntu-20.04

python3.8.19

cuda11.3
```
1. Clone OneFormer Repo
git clone https://github.com/SHI-Labs/OneFormer-Colab.git

cd OneFormer-Colab/
```
```
2. Install Dependencies.

pip3 install -r requirements.txt

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip3 install -U opencv-python

pip3 install natten==0.14.6+torch1101cu113 -f https://shi-labs.com/natten/wheels

pip3 install git+https://github.com/cocodataset/panopticapi.git

pip3 install git+https://github.com/mcordts/cityscapesScripts.git

pip3 install ipython-autotime

pip3 install imutils
```
