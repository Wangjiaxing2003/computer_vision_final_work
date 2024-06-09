import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from segment_anything import sam_model_registry, SamPredictor
import clip
from tqdm import tqdm

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

print(device)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "default"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def show_mask_contours(mask, ax, color=(1, 0, 1), thickness=2):
    # 将掩膜格式转换为OpenCV可以处理的形式
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓，而不是填充整个区域
    for cnt in contours:
        cv2.drawContours(image, [cnt], 0, (int(color[0]*255), int(color[1]*255), int(color[2]*255)), thickness)
    ax.imshow(image)


30
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

image_path='2/3/6.bmp'
result_folder = "result"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
input_point = np.array([[200, 120]])
input_label = np.array([1])
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

if len(masks) > 0 and len(scores) > 0:
    mask, score = masks[0], scores[0]  # 获取第一个结果
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask_contours(mask, plt.gca())  # 使用先前提到的函数来只显示轮廓
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask 1, Score: {score:.3f}", fontsize=18)  # 这里我们假设只展示第一个mask，所以是Mask 1
    plt.axis('off')  # 关闭坐标轴
    plt.savefig('result/3_6.jpg', bbox_inches='tight')
    plt.show()
else:
    print("没有检测到 masks 或 scores.")