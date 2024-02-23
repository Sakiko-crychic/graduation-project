import cv2
import numpy as np
import yaml
from pathlib import Path
import sys
import os

# 参数调整
# 图像分辨率设置
resolution = 1
# 定义最小长度
min_length = 30  # 可根据需要调整
# Sobel算子参数
ksize = 5  # Sobel核大小
threshold_value = 2050  # 二值化阈值
# 阴影消除
ShadowThresholding = 225  # 阴影消除二值化阈值

def shadow_detection_hsv(image):
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    shadow_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            h, s, v = dst[i, j]
            # 对像素值进行缩放，确保在加法和减法操作中不会溢出
            h_scaled = h / 255.0
            s_scaled = s / 255.0
            v_scaled = v / 255.0
            denominator = h_scaled + s_scaled + v_scaled
            m = float((s_scaled - v_scaled) / denominator) if denominator != 0 else 0
            shadow_mask[i, j] = 0 if m > 0.2 else 255
    return shadow_mask



def main():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    # 读取标签映射文件
    yaml_file_path = ROOT / "data/mydata_seg.yaml"
    print("Reading label mapping file:", yaml_file_path)
    with open(str(yaml_file_path), 'r') as file:
        label_mapping = yaml.safe_load(file)

    # 读取图像并转换为灰度图像
    image_path = ROOT / "data/seg-images/0096.jpg"
    print("Reading image:", image_path)
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 读取位置信息
    regions_file_path = ROOT / "runs/predict-seg/exp3/labels/0096.txt"
    print("Reading regions file:", regions_file_path)
    with open(str(regions_file_path), 'r') as file:
        regions = file.readlines()

    shadow_mask = shadow_detection_hsv(image)
    image_no_shadow = cv2.inpaint(image, shadow_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(image_no_shadow, cv2.COLOR_BGR2GRAY)
    # 全局二值化
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 高斯滤波
    blurred = cv2.GaussianBlur(thresholded, (7, 7), 0)

    # 膨胀和腐蚀处理
    kernel = np.ones((7, 7), np.uint8)
    dilated_eroded = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    # 打印消除阴影后的图像
    cv2.imshow('Image without Shadow', image_no_shadow)
    cv2.imshow('Image with erode', dilated_eroded)

    # 获取图像的宽度和高度
    image_height, image_width = image_no_shadow.shape[:2]

    # 存储需要应用Sobel算子的区域
    sobel_regions = []

    # 遍历每个 ROI 区域
    for region_info in regions:
        region_info = region_info.strip().split(' ')
        shape_id = int(region_info[0])

        # 只处理 label 为 0 的区域
        if shape_id in [0, 2]:
            # 解析顶点坐标信息
            coordinates = [float(coord) for coord in region_info[1:]]
            coordinates = [(int(coordinates[i] * image_width), int(coordinates[i + 1] * image_height)) for i in
                           range(0, len(coordinates), 2)]
            pts = np.array(coordinates, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # 添加到Sobel算子应用的区域列表中
            sobel_regions.append(pts)

    # 生成一张空白图像用于叠加边缘检测结果
    edges_combined = np.zeros_like(gray)

    # 应用Sobel算子并叠加边缘检测结果
    for pts in sobel_regions:
        # 创建当前ROI区域的蒙版
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # 转换为灰度图像并应用蒙版
        masked_gray = cv2.bitwise_and(dilated_eroded, dilated_eroded, mask=mask)

        # Sobel边缘检测
        sobel_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)

        # 应用阈值处理
        _, sobel_thresholded = cv2.threshold(sobel_combined, threshold_value, 255, cv2.THRESH_BINARY)

        # 将 sobel_thresholded 转换为与 edges_combined 相同的数据类型
        sobel_thresholded = sobel_thresholded.astype(edges_combined.dtype)

        # 将当前ROI区域的边缘检测结果叠加到总图像上
        edges_combined = cv2.add(edges_combined, sobel_thresholded)

    # 查找轮廓
    contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个空白图像，与原始图像具有相同的大小和通道数
    contour_image = np.zeros_like(image)

    # 在轮廓图像上绘制轮廓
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # 显示带有轮廓的图像
    cv2.imshow('Contours', contour_image)
    cv2.waitKey(0)
    # 在原始图像上绘制轮廓并标注每条边的长度
    for contour in contours:
        # 对轮廓进行多边形逼近，减少多边形的顶点数目，以减少凹凸部分造成的短线段
        epsilon = 0.002365 * cv2.arcLength(contour, True)  # 调整epsilon的值，使得多边形逼近更加接近原始轮廓
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 获取逼近后的多边形的所有点
        points = approx.squeeze()

        # 计算每条边的长度并绘制标注
        for i in range(0, len(points)):
            pt1 = points[i]
            pt2 = points[(i + 1) % len(points)]  # 循环取下一个点
            length_pixels = np.linalg.norm(pt2 - pt1)  # 计算两点之间的距离作为边长

            # 仅显示长度大于最小长度的边长
            if length_pixels >= min_length:
                # 工程标注方法：标注线段两端处的长度
                # 计算标注位置
                mid_point = tuple(((pt1.astype(float) + pt2.astype(float)) / 2).astype(int))

                text_offset = (10, -10)  # 文本偏移量，用于调整文本位置
                cv2.putText(image, f'{length_pixels:.2f}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.line(image, tuple(pt1), tuple(pt2), (0, 255, 0), 2)

    # 显示带有直线的图像
    cv2.imshow('Lines Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
