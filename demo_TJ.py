import sys, os, distutils.core
dist = distutils.core.run_setup("./detectron2/setup.py")
sys.path.insert(0, os.path.abspath('./detectron2'))





######
#@title 3. Import Libraries and other Utilities
######
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="oneformer")

# Import libraries
import numpy as np
import cv2
import torch
import imutils

# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode


# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)


######
#@title 4. Define helper functions
######
cpu_device = torch.device("cuda")
SWIN_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}

DINAT_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

def setup_cfg(dataset, model_path, use_swin):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
      cfg_path = SWIN_CFG_DICT[dataset]
    else:
      cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg

def setup_modules(dataset, model_path, use_swin):
    cfg = setup_cfg(dataset, model_path, use_swin)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
        from cityscapesscripts.helpers.labels import labels
        stuff_colors = [k.color for k in labels if k.trainId != 255]
        metadata = metadata.set(stuff_colors=stuff_colors)
    
    return predictor, metadata

def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out = visualizer.draw_panoptic_seg_predictions(
    panoptic_seg.to(cpu_device), segments_info, alpha=0.5
)
    return out

def instance_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return out

def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    return out

TASK_INFER = {"panoptic": panoptic_run, 
              "instance": instance_run, 
              "semantic": semantic_run}


use_swin = False #@param {type: 'boolean'}

######
#@title A. Initialize Model
######
# download model checkpoint
# import os
# import subprocess
# if not use_swin:
#   if not os.path.exists("250_16_dinat_l_oneformer_cityscapes_90k.pth"):
#     subprocess.run('wget https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth', shell=True)
#   predictor, metadata = setup_modules("cityscapes", "250_16_dinat_l_oneformer_cityscapes_90k.pth", use_swin)
# else:
#   if not os.path.exists("250_16_swin_l_oneformer_cityscapes_90k.pth"):
#     subprocess.run('wget https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth', shell=True)
#   predictor, metadata = setup_modules("cityscapes", "250_16_swin_l_oneformer_cityscapes_90k.pth", use_swin)




  ######
#@title B. Display Sample Image. You can modify the path and try your own images!
######
######
#@title A. Initialize Model
######
# download model checkpoint
import os
import subprocess
if not use_swin:
  if not os.path.exists("250_16_dinat_l_oneformer_cityscapes_90k.pth"):
    subprocess.run('wget https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth', shell=True)
  predictor, metadata = setup_modules("cityscapes", "250_16_dinat_l_oneformer_cityscapes_90k.pth", use_swin)
else:
  if not os.path.exists("250_16_swin_l_oneformer_cityscapes_90k.pth"):
    subprocess.run('wget https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth', shell=True)
  predictor, metadata = setup_modules("cityscapes", "250_16_swin_l_oneformer_cityscapes_90k.pth", use_swin)
# change path here for another image

#cv2.imshow(out[:, :, ::-1])



import glob
import tqdm
folder_path = "../Data/tj4d_image_2/image_2/"
#print(folder_path)
outputpath="./output_tj4d/"
if not os.path.exists(folder_path):
    print(f"文件夹路径不存在: {folder_path}")
else:
    # 获取所有 JPEG 图像路径
    image_path = glob.glob(os.path.join(folder_path, '*.png'))
    
    # 打印找到的文件路径以调试
    if not image_path:
        print("未找到任何 .jpg 文件")
    else:
        #print(f"找到的文件: {image_path}")
    
        # 按照文件名的数值大小进行排序
        image_path = sorted(image_path, key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        # 打印排序后的结果
        #print("排序后的文件路径:")
        #print(image_path)
###########################################################
import json
# 创建图片编号到结果张量索引的映射
index_map = {}
for i, path in enumerate(image_path):
    filename = os.path.basename(path)
    img_index = int(filename.split('.')[0])
    index_map[img_index] = i

# 保存映射到 JSON 文件
output_path = os.path.join(outputpath, "index_map.json")
with open(output_path, "w") as f:
    json.dump(index_map, f)
##############################################################

batch_size = 1
num_batches = len(image_path) // batch_size + (len(image_path) % batch_size != 0)

for batch_index in range(90,num_batches):
    
    start_index = batch_index * batch_size
    end_index = min((batch_index + 1) * batch_size, len(image_path))
    batch_paths = image_path[start_index:end_index]

    results = []


    for path in tqdm.tqdm(batch_paths, disable=not outputpath):
        print(path)
        img = cv2.imread(path)
        #img = imutils.resize(img, width=512)

        task = "instance" #@param
        out = TASK_INFER[task](img, predictor, metadata).get_image()
        from demo.visualizer import pred_masks1,pred_scores1,pred_classes1
        pred_masks1=torch.tensor(pred_masks1).cpu()
        pred_scores1=torch.tensor(pred_scores1).cpu()
        pred_classes1=torch.tensor(pred_classes1).cpu()
        selected_classes = [11, 12, 13, 14, 15, 16, 17, 18]
        if(min((pred_scores1 > 0.1).shape)==0):
            continue
        else:
            selected_indices = (pred_scores1 > 0.1) & torch.tensor([pred_class in selected_classes for pred_class in pred_classes1])
        print("pred_masks1",pred_masks1)
        print("pred_masks1.shape",pred_masks1.shape)
        print("selected_indices",selected_indices)
        print("selected_indices.shape",selected_indices.shape)
        #selected_indices=selected_indices.item()
        selected_masks = pred_masks1[selected_indices]
        
        selected_pre_classes = pred_classes1[selected_indices]
        person_cal = 0
        car_cal = 0
        cyclist_cal = 0
        truck_cal=0
        bus_cal=0
        train_cal=0
        motorcycle_cal=0
        bicycle_cal=0
        for i in range(len(selected_masks)):
            mask_sel = selected_masks[i]
            class_id_sel = selected_pre_classes[i]
            # 人  1
            if class_id_sel == 11:
                mask_sel = torch.where(mask_sel == 1, mask_sel + person_cal, mask_sel)
                person_cal += 1
            # car bus 2
            if class_id_sel == 13 or class_id_sel == 15:
                mask_sel = torch.where(mask_sel == 1, mask_sel + 100 + car_cal, mask_sel)
                car_cal += 1
            # rider 自行车 摩托车  3
            if class_id_sel == 12 or class_id_sel == 17 or class_id_sel == 18:
                mask_sel = torch.where(mask_sel == 1, mask_sel + 200 + cyclist_cal, mask_sel)
                cyclist_cal += 1  
            # truck  4
            if class_id_sel == 14:
                mask_sel = torch.where(mask_sel == 1, mask_sel + 300 + truck_cal, mask_sel)
                truck_cal += 1
     
            selected_masks[i] = mask_sel
        # print('i',i)
        # print('person_cal',person_cal)
        # print('car_cal',car_cal)
        # print('cyclist_cal',cyclist_cal)
        if len(selected_masks) == 0:
            selected_masks = torch.zeros(960,1280)
        
        mask_result, _ = selected_masks.max(dim=0)
        results=mask_result.reshape(960,1280)
    if(min((pred_scores1 > 0.1).shape)==0):
            continue    
    #results_tensor = torch.stack(results) 
    output_folder = os.path.join(outputpath, 'batch_results')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'mask_{batch_index}.pt')
    print("output_path",output_path)
    print("mask_result.shape",mask_result.shape)
    torch.save(mask_result, output_path)   