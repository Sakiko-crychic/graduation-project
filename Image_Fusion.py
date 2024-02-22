#2024/1/26 MOPSO优化
import numpy as np
from pyswarm import pso
from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import img_as_ubyte
import cv2
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.transform import resize

def evaluate(image, params, original_gray):
    # 在这个例子中，假设 params 是图像处理参数
    processed_image = some_image_processing_function(image, params)

    cc, psnr_value, avg_grad, rmse, rase = evaluate_fusion(original_gray, processed_image)

    # 返回一个目标函数值，这里取负值使其变为最大化问题
    return -psnr_value

# 图像处理函数，根据参数处理图像
def some_image_processing_function(image, params):
    # 在这里实现具体的图像处理操作，根据参数修改图像
    # 如果输入图像是彩色的，对每个通道进行相同或类似的处理
    if image.ndim == 3:
        processed_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            processed_image[:, :, channel] = some_processing_function_for_each_channel(image[:, :, channel], params)
    else:
        processed_image = some_processing_function_for_each_channel(image, params)

    # 确保处理后的图像与输入图像具有相同的大小
    processed_image = resize(processed_image, image.shape, mode='reflect', anti_aliasing=True)

    return processed_image

# 通道处理函数，根据参数处理通道
def some_processing_function_for_each_channel(channel, params):
    # 在这里实现具体的通道处理操作，根据参数修改通道
    # 这里只是一个示例，实际上需要替换为具体的通道处理操作
    processed_channel = channel  # 保留通道信息
    return processed_channel

def evaluate_fusion(original, fused):
    # 将原始图像转换为RGB格式，以匹配融合后的图像
    original_rgb = color.gray2rgb(original)

    cc = np.corrcoef(original_rgb.flatten(), fused.flatten())[0, 1]

    psnr_value = psnr(img_as_ubyte(original_rgb), img_as_ubyte(fused))

    # 计算灰度图像的梯度
    grad_original_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
    grad_original_y = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)

    # 分别计算每个通道的融合图像梯度
    grad_fused_x = np.zeros_like(fused)
    grad_fused_y = np.zeros_like(fused)

    for i in range(fused.shape[2]):
        grad_fused_x[:, :, i] = cv2.Sobel(fused[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
        grad_fused_y[:, :, i] = cv2.Sobel(fused[:, :, i], cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    grad_original = np.sqrt(grad_original_x ** 2 + grad_original_y ** 2)
    grad_fused = np.sqrt(np.sum(np.square(grad_fused_x), axis=2) + np.sum(np.square(grad_fused_y), axis=2))

    # 计算梯度差异
    grad_diff = np.abs(grad_original - grad_fused)
    avg_grad = np.mean(grad_diff)

    # 将 original 转换为彩色图像
    original_rgb = color.gray2rgb(original)

    # 确保计算的均方误差具有相同的形状
    rmse = np.sqrt(mean_squared_error(original_rgb, fused))

    # 计算实际值的平均值
    mean_original = np.mean(original_rgb)

    # 计算 RASE
    rase = rmse / mean_original

    return cc, psnr_value, avg_grad, rmse, rase


# MOPSO 优化函数
def mopso_optimization():
    # 读取全色和多光谱图像
    pan_image = cv2.imread('./data/fusion_images/PAN.png', cv2.IMREAD_GRAYSCALE)
    ms_image = cv2.imread('./data/fusion_images/MS.png')
    # 读取输入图像
    input_image_path = "./data/fusion_images/fused_image_bior3.7.png"
    input_image = io.imread(input_image_path)
    original_gray = cv2.cvtColor(ms_image, cv2.COLOR_BGR2GRAY)

    # 将图像转为彩色图（如果不是彩色图的话）
    if input_image.ndim == 2:
        input_image = color.gray2rgb(input_image)

    # 参数范围
    param_ranges = np.array([
        (-1.0, 1.0),
        (-1.0, 1.0),
        # ... 继续列举其他参数的范围
    ])

    # 定义优化问题
    def objective(params):
        # 将参数传递给评价函数
        return evaluate(input_image, params, original_gray)

    # 使用 PSO 算法进行优化
    best_params, _ = pso(objective, lb=param_ranges[:, 0], ub=param_ranges[:, 1])

    # 打印最优参数
    print("Best parameters:", best_params)

    # 使用最优参数处理图像
    processed_image = some_image_processing_function(input_image, best_params)

    # 保存处理后的图像
    output_image_path = "./data/fusion_images/processed_image.png"

    # 确保在保存之前图像处于适当的模式
    processed_image_uint8 = img_as_ubyte(processed_image)

    io.imsave(output_image_path, processed_image_uint8)  # 保持彩色信息
    # 计算评价指标
    cc, psnr_value, avg_grad, rmse, rase = evaluate_fusion(original_gray, processed_image_uint8)

    # 保存结果到文件
    result_path = "D:/myproject/data/fusion_images/result/results_mopso.txt"
    with open(result_path, 'w') as f:
        f.write(f"CC: {cc}\n")
        f.write(f"PSNR: {psnr_value}\n")
        f.write(f"Avg Grad: {avg_grad}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"RASE: {rase}\n")
# 执行 MOPSO 优化
mopso_optimization()

