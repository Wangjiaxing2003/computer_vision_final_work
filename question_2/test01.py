import cv2
import numpy as np
import os

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def apply_sharpening(image, kernel=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])):
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# 设置图像所在文件夹路径
folder_path = "2/3"
# 获取文件夹内的文件名列表
filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".bmp", ".png", ".jpg", ".jpeg"))]

# 读取图像
images = [cv2.imread(filename) for filename in filenames]

# 初始化ORB检测器
orb = cv2.ORB_create()

# 存储对齐后的图像列表
aligned_images = []

# 添加一个新的参数，用于控制特征点匹配的最大数量
max_matches = 30

if images:
    # 将第一张图像作为参考，放入结果列表中
    images[0] = apply_gaussian_blur(images[0])
    images[0] = apply_sharpening(images[0])
    aligned_images.append(images[0])

    # 对参考图像以外的所有图像做特征匹配和对齐
    for i in range(1, len(images)):
        images[i] = apply_gaussian_blur(images[i])
        images[i] = apply_sharpening(images[i])
        # 检测特征点和描述子
        keypoints1, descriptors1 = orb.detectAndCompute(aligned_images[0], None)
        keypoints2, descriptors2 = orb.detectAndCompute(images[i], None)

        # 创建暴力匹配器对象
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 执行匹配
        matches = bf.match(descriptors1, descriptors2)

        # 对匹配的结果进行排序并根据 max_matches 截取
        matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

        # 取出匹配点
        points1 = np.zeros((len(matches), 2), dtype=np.float64)
        points2 = np.zeros_like(points1)

        for j, match in enumerate(matches):
            points1[j, :] = keypoints1[match.queryIdx].pt
            points2[j, :] = keypoints2[match.trainIdx].pt

        # 使用RANSAC方法估算仿射变换矩阵，以增加鲁棒性
        M, mask = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)

        if M is not None and M.shape == (2, 3):
            # 对图像进行变换，进行对齐
            aligned_image = cv2.warpAffine(images[i], M, (images[0].shape[1], images[0].shape[0]))
            aligned_images.append(aligned_image)
        else:
            print(f"无法计算从图像 {i} 到参考图像的仿射变换矩阵。")

# 现在aligned_images包含了所有对齐后的图像
# 保存或展示对齐后的图像
if not os.path.isdir(os.path.join(folder_path, "aligned")):
    os.mkdir(os.path.join(folder_path, "aligned"))

for idx, aligned_image in enumerate(aligned_images):
    aligned_image_path = os.path.join(folder_path, "aligned", f'aligned_image{idx + 1}.jpg')
    cv2.imwrite(aligned_image_path, aligned_image)
    # 如果需要查看图像，取消注释以下行
    # cv2.imshow(f'Aligned Image {idx+1}', aligned_image)

# 如果开启了图像显示，取消注释以下行
# cv2.waitKey(0)
# cv2.destroyAllWindows()
