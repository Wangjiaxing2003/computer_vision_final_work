import cv2
import numpy as np

def estimate_hand_center(image):
    # 转换到 HSV 色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 定义手部肤色在 HSV 空间的范围
    # 注意：这些值仅作为示例，实际值需要根据你的图像进行调整
    lower_skin = np.array([0, 10, 140], dtype=np.uint8)
    upper_skin = np.array([179, 94, 255], dtype=np.uint8)

    # 肤色掩膜
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 对手部区域做膨胀操作，使其更连续
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 图像与掩膜相乘，提出手部
    hand = cv2.bitwise_and(image, image, mask=mask)

    # 查找手部的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 如果有找到轮廓
    if contours:
        # 找到最大轮廓即为手部
        max_contour = max(contours, key=cv2.contourArea)

        # 获取最小外接圆的中心和半径
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        # 返回圆的中心作为手部的估计中心
        return (int(x), int(y))
    else:
        # 如果没有找到轮廓，返回图像中心
        return(image.shape[1] // 2, image.shape[0] // 2)

# 测试函数
image = cv2.imread("2/1/1.bmp")  # 读取你要处理的图像
center = estimate_hand_center(image)
print("Estimated center:", center)
