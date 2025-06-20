import cv2
import numpy as np
import os
from glob import glob

def compensate_illumination(img, blur_kernel=(621, 621), compensate_strength=128):
    # 將影像轉 HSV 以處理亮度（V channel）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    # 模糊 V 通道以估計亮度分布
    blurred = cv2.GaussianBlur(v, blur_kernel, 0)

    # 補償亮度，防止除以零
    corrected_v = (v / (blurred + 1e-6)) * compensate_strength
    corrected_v = np.clip(corrected_v, 0, 255).astype(np.uint8)

    # 合成回 HSV 並轉回 BGR
    hsv[:, :, 2] = corrected_v
    corrected_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return corrected_img

def process_folder(input_folder, output_folder, compensate_strength=128, blur_kernel=(81, 81)):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 取得所有圖片檔案
    image_paths = glob(os.path.join(input_folder, "*.*"))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp','.tif']

    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext not in valid_exts:
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        # 進行亮度補償
        result = compensate_illumination(img, blur_kernel, compensate_strength)

        # 儲存結果
        filename = os.path.basename(path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, result)
        print(f"Saved: {save_path}")

# ======== 使用範例 ========
input_folder = "C:\\Users\\User\\Desktop\\tt"        # 替換成你的輸入資料夾
output_folder = "C:\\Users\\User\\Desktop\\tt\\result"   # 輸出資料夾
compensate_strength = 150           # 可調整補償強度（常見值：100~150）

process_folder(input_folder, output_folder, compensate_strength)
