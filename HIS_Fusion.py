#2024/1/25 HIS加小波变换，计算出各种小波的评价指标以方便比较那种小波融合效果好
import cv2
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error


def canny_edge_weight(image, sigma=1.0):
    # 使用Canny算子提取边缘
    edges = cv2.Canny(image.astype(np.uint8), 50, 150)  # 调整低阈值和高阈值

    # 计算高斯权重
    weights = cv2.GaussianBlur(edges.astype(np.float64), (0, 0), sigma)
    weights /= np.max(weights)  # 归一化到 [0, 1] 范围

    return weights


def evaluate_fusion(original, fused):
    cc = np.corrcoef(original.flatten(), fused.flatten())[0, 1]
    psnr = peak_signal_noise_ratio(original, fused)

    grad_original_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
    grad_original_y = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
    grad_fused_x = cv2.Sobel(fused, cv2.CV_64F, 1, 0, ksize=3)
    grad_fused_y = cv2.Sobel(fused, cv2.CV_64F, 0, 1, ksize=3)
    grad_original = np.sqrt(grad_original_x ** 2 + grad_original_y ** 2)
    grad_fused = np.sqrt(grad_fused_x ** 2 + grad_fused_y ** 2)
    grad_diff = np.abs(grad_original - grad_fused)
    avg_grad = np.mean(grad_diff)

    rmse = np.sqrt(mean_squared_error(original, fused))

    # 计算实际值的平均值
    mean_original = np.mean(original)

    # 计算 RASE
    rase = rmse / mean_original

    return cc, psnr, avg_grad, rmse, rase


# 读取全色和多光谱图像
pan_image = cv2.imread('./data/fusion_images/PAN.png', cv2.IMREAD_GRAYSCALE)
ms_image = cv2.imread('./data/fusion_images/MS.png')

# HIS变换
hsv_image = cv2.cvtColor(ms_image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)

# 尝试不同的小波基
wavelets = ['haar', 'db3', 'bior3.7', 'sym3', 'rbio3.7', 'dmey']  # 可以继续添加其他小波基

result_file_path = './data/fusion_images/result/results.txt'

# 清空文件内容
open(result_file_path, 'w').close()

for wavelet_name in wavelets:
    # 小波分解
    coeffs_pan = pywt.dwt2(pan_image, wavelet_name)
    coeffs_ms = pywt.dwt2(v, wavelet_name)

    # 获取低频和高频分量
    try:
        A_pan, (H_pan, V_pan, D_pan) = coeffs_pan
    except ValueError:
        A_pan, (H_pan, V_pan) = coeffs_pan
        D_pan = None

    try:
        A_ms, (H_ms, V_ms, D_ms) = coeffs_ms
    except ValueError:
        A_ms, (H_ms, V_ms) = coeffs_ms
        D_ms = None

    # 高频分量融合（基于Canny算子的思想）
    weight_pan = canny_edge_weight(pan_image, sigma=1.0)
    weight_ms = canny_edge_weight(v, sigma=1.0)

    # 将权重调整为与高频分量相同的大小
    weight_pan_resized = cv2.resize(weight_pan, (H_pan.shape[1], H_pan.shape[0]))
    weight_ms_resized = cv2.resize(weight_ms, (H_ms.shape[1], H_ms.shape[0]))

    H_fused = weight_pan_resized * H_pan + weight_ms_resized * H_ms
    V_fused = weight_pan_resized * V_pan + weight_ms_resized * V_ms
    D_fused = weight_pan_resized * D_pan + weight_ms_resized * D_ms if D_pan is not None and D_ms is not None else (
        D_pan if D_pan is not None else D_ms)

    # 低频分量融合（取加权平均）
    A_fused = (A_pan + A_ms) / 2

    # 逆小波变换
    if D_fused is not None:
        coeffs_fused = A_fused, (H_fused, V_fused, D_fused)
    else:
        coeffs_fused = A_fused, (H_fused, V_fused)
    fused_image = pywt.idwt2(coeffs_fused, wavelet_name)

    # Resize fused_image to match the original image size
    fused_image = cv2.resize(fused_image.astype(np.uint8), (ms_image.shape[1], ms_image.shape[0]))

    # HIS逆变换
    hsv_fused = cv2.merge([h, s, fused_image])
    fused_rgb = cv2.cvtColor(hsv_fused, cv2.COLOR_HSV2BGR)

    # 保存融合后的图像
    output_path = f'./data/fusion_images/fused_image_{wavelet_name}.png'
    cv2.imwrite(output_path, fused_rgb)

    # 计算客观评价指标并保存到txt文件
    original_gray = cv2.cvtColor(ms_image, cv2.COLOR_BGR2GRAY)
    cc, psnr, avg_grad, rmse, rase = evaluate_fusion(original_gray, fused_image)

    with open(result_file_path, 'a') as result_file:
        result_file.write(f'\nResults for {wavelet_name}:\n')
        result_file.write(f'CC: {cc}\n')
        result_file.write(f'PSNR: {psnr}\n')
        result_file.write(f'Avg Grad: {avg_grad}\n')
        result_file.write(f'RMSE: {rmse}\n')
        result_file.write(f'RASE: {rase}\n')