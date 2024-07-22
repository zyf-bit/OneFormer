import pickle
import torch
import matplotlib.pyplot as plt
import json
import os

img_index = 0# 假设要查找编号为123的图片对应的张量索引

# 加载映射

with open("./output_vod/index_map.json", "r") as f:
    index_map = json.load(f)
print('index_map',index_map)

# 根据图片编号查找对应的结果张量索引
tensor_index = index_map.get(str(img_index), -1)
print('tensor_index',tensor_index)
if tensor_index != -1:
    load_path = './output_vod/batch_results/mask_{}.pt'.format(tensor_index)
    mask_result = torch.load(load_path)
else:
    # 没找到对应的张量索引
    print(f"图片编号 {img_index} 对应的张量索引不存在")


print('mask_result.shape', mask_result.shape)
print('mask_result\n', mask_result)

plt.imshow(mask_result)

plt.axis('off')
plt.show()   