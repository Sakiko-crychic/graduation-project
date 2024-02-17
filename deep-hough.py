import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_lines(img, houghLinesP, color=[255, 0, 0], thickness=2):
    for line in houghLinesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def weighted_img(img, initial_img, alpha=0.8, beta=1., λ=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, λ)

# 读取图片
img = cv2.imread("./data/images/urban.jpg")

# 图片灰度化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# 边缘检测
img_edges = cv2.Canny(img_blur, 50, 120)

# 概率霍夫变换
rho = 1.2
theta = np.pi / 360
threshold = 200
min_line_length = 50
max_line_gap = 10
hough_linesP = cv2.HoughLinesP(img_edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

# 初始化 matplotlib subplot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(bottom=0.25)  # 调整布局，提高滑块位置

# 显示图像和滑块
axes[0].imshow(img, cmap="gray")
axes[0].set_title("源图像", fontsize=12)
axes[0].axis("off")

axes[1].imshow(img_edges, cmap="gray")
axes[1].set_title("边缘图", fontsize=12)
axes[1].axis("off")

# 初始化 img_lines 为全局变量
img_lines = np.zeros_like(img)

axes[2].imshow(img_lines)
axes[2].set_title("带有Hough直线的图像（概率霍夫）", fontsize=12)
axes[2].axis("off")

slider_ax_rho = fig.add_axes([0.2, 0.3, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_rho = Slider(slider_ax_rho, 'rho', 0.1, 5.0, valinit=rho)

slider_ax_threshold = fig.add_axes([0.2, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_threshold = Slider(slider_ax_threshold, 'threshold', 0, 500, valinit=threshold)

slider_ax_min_length = fig.add_axes([0.2, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_min_length = Slider(slider_ax_min_length, 'min_length', 0, 200, valinit=min_line_length)

slider_ax_max_gap = fig.add_axes([0.2, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_max_gap = Slider(slider_ax_max_gap, 'max_gap', 0, 50, valinit=max_line_gap)


def update(val):
    global img_lines  # 将 img_lines 声明为全局变量
    rho = slider_rho.val
    threshold = int(slider_threshold.val)  # 将 threshold 转换为整数
    min_line_length = int(slider_min_length.val)  # 将 min_line_length 转换为整数
    max_line_gap = int(slider_max_gap.val)  # 将 max_line_gap 转换为整数

    hough_linesP = cv2.HoughLinesP(img_edges, rho, theta, threshold, minLineLength=min_line_length,
                                   maxLineGap=max_line_gap)

    if hough_linesP is not None:
        img_lines = np.zeros_like(img)
        draw_lines(img_lines, hough_linesP)
        img_lines = weighted_img(img_lines, img)

        axes[2].imshow(img_lines)
        axes[2].set_title("带有Hough直线的图像（概率霍夫）", fontsize=12)
        axes[2].axis("off")



slider_rho.on_changed(update)
slider_threshold.on_changed(update)
slider_min_length.on_changed(update)
slider_max_gap.on_changed(update)

plt.show()
