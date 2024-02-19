import cv2
import numpy as np
import yaml
from pathlib import Path
import sys
import os

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

# 读取位置信息
regions_file_path = ROOT / "runs/predict-seg/exp2/labels/0001M.txt"
print("Reading regions file:", regions_file_path)
with open(str(regions_file_path), 'r') as file:
    regions = file.readlines()

# 读取图像并转换为灰度图像
image_path = ROOT / "data/seg-images/0001M.jpg"
print("Reading image:", image_path)
image = cv2.imread(str(image_path))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 获取图像的宽度和高度
image_height, image_width = image.shape[:2]

# 创建空白的蒙版
mask = np.zeros((image_height, image_width), dtype=np.uint8)

# 存储需要进行霍夫变换的区域的蒙版
hough_mask = np.zeros((image_height, image_width), dtype=np.uint8)

# 霍夫变换参数设置
hough_params = {
    0: {'rho': 1, 'threshold': 200, 'minLineLength': 1, 'maxLineGap': 0},
    2: {'rho': 1, 'threshold': 830, 'minLineLength': 10, 'maxLineGap': 2}
}

# 遍历每个 ROI 区域
for region_info in regions:
    region_info = region_info.strip().split(' ')
    shape_id = int(region_info[0])

    # 只处理 label 为 0 和 2 的区域
    if shape_id not in [0, 2]:
        continue

    # 解析顶点坐标信息
    coordinates = [float(coord) for coord in region_info[1:]]
    coordinates = [(int(coordinates[i] * image_width), int(coordinates[i + 1] * image_height)) for i in range(0, len(coordinates), 2)]
    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # 打印顶点坐标信息
    print("Coordinates:", coordinates)

    cv2.fillPoly(mask, [pts], 255)  # 将 ROI 区域设为 255
    cv2.fillPoly(hough_mask, [pts], 255)

    # 如果当前标签对应的参数存在，则应用霍夫变换
    if shape_id in hough_params:
        params = hough_params[shape_id]
        rho = params['rho']
        threshold = params['threshold']
        minLineLength = params['minLineLength']
        maxLineGap = params['maxLineGap']

        # 应用蒙版
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        # 在蒙版中为 255 的区域执行霍夫变换
        lines = cv2.HoughLinesP(masked_gray, rho=rho, theta=np.pi / 180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

        # 将检测到的线绘制到图像上
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if hough_mask[y1, x1] == 255 or hough_mask[y2, x2] == 255:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 重置蒙版和参数
    mask.fill(0)
    hough_mask.fill(0)

# 显示结果图像
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
