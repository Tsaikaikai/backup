import cv2
import numpy as np
import os
from pathlib import Path
from glob import glob
from datetime import datetime
import time

def get_timestamp():
    """
    獲取當前時間戳記，精確到毫秒
    返回格式: YYYYMMDD_HHMMSS_mmm
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

def reconstruct(bin_image1,bin_image2):

    # 找到第一張影像中的輪廓
    contours, _ = cv2.findContours(bin_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 創建一個空白影像，用於存放結果
    result = np.zeros_like(bin_image1)

    # 遍歷每個輪廓
    for contour in contours:
        # 創建與原影像大小相同的遮罩
        mask = np.zeros_like(bin_image1)
        # 在遮罩上繪製該輪廓
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # 檢查該區域是否與第二張影像有重疊
        overlap = cv2.bitwise_and(mask, bin_image2)
        if np.any(overlap):  # 如果有重疊
            # 保留整個封閉區域
            result = cv2.bitwise_or(result, mask)

    return result


def process_and_crop_image(input_path, output_dir, threshold, window_size, overlap, down_scale):
    """
    處理圖片並進行切割的主函數
    
    參數:
    input_path: 輸入圖片路徑
    output_dir: 輸出切割後圖片的資料夾路徑
    threshold: 二值化閾值
    window_size: 切割窗口大小
    overlap: 重疊率(0~1之間)
    down_scale: 縮小倍數(建議10)
    """
    # 讀取圖片
    ori_img = cv2.imread(input_path)
    if ori_img is None:
        raise ValueError(f"無法讀取圖片: {input_path}")
    
    #縮小尺度
    height, width = ori_img.shape[:2]
    img = cv2.resize(ori_img,(width//down_scale,height//down_scale),interpolation=cv2.INTER_NEAREST)

    # 轉換為灰度圖
    ori_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(input_path[:-4]+'_binary.tiff',binary)
    
    # 定義形態學操作的核心
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,iterations=2)
    dilated = cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel,iterations=2)
    cv2.imwrite(input_path[:-4]+'_dilated.tiff',dilated)

    #reconstruct
    remask = cv2.imread('D:\\newSGM\\reconstruct_mask.tif')
    remask = cv2.cvtColor(remask, cv2.COLOR_BGR2GRAY)
    remask = cv2.resize(remask,(width//down_scale,height//down_scale),interpolation=cv2.INTER_NEAREST)

    construct_result = reconstruct(dilated,remask)
    cv2.imwrite(input_path[:-4]+'_construct_result.tiff',construct_result)

    # 找到連通區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(construct_result, connectivity=8)
    
    # 找到最大的白色區域（排除背景）
    max_label = 1  # 初始化為1（因為0是背景）
    max_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
    
    # 創建只包含最大區域的遮罩
    largest_component = np.zeros_like(labels, dtype=np.uint8)
    largest_component[labels == max_label] = 255

    #邊緣平整化
    largest_component = cv2.morphologyEx(largest_component,cv2.MORPH_OPEN,kernel,iterations=2)
    largest_component = cv2.morphologyEx(largest_component,cv2.MORPH_CLOSE,kernel,iterations=2)

    #復原尺寸
    height, width = ori_img.shape[:2]
    largest_component = cv2.resize(largest_component,(width,height),interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(input_path[:-4]+'_mask_smoo.jpg',largest_component)

    # 找到邊緣
    edges = cv2.Canny(largest_component, 100, 200)
    
    # 計算滑動窗口的步長（基於重疊率）
    stride = int(window_size * (1 - overlap))
    
    # 獲取圖像尺寸
    height, width = edges.shape
    
    # 切割圖片
    count = 0
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # 提取當前窗口
            window = edges[y:y+window_size, x:x+window_size]
            
            # 計算邊緣像素的位置
            edge_pixels = np.where(window > 0)
            
            if len(edge_pixels[0]) > 0:  # 如果窗口中有邊緣
                # 計算邊緣的中心
                center_y = int(np.mean(edge_pixels[0]))
                center_x = int(np.mean(edge_pixels[1]))
                
                # 計算新的裁剪區域（將邊緣置中）
                start_y = y + center_y - window_size//2
                start_x = x + center_x - window_size//2
                
                # 確保不超出圖像邊界
                start_y = max(0, min(start_y, height - window_size))
                start_x = max(0, min(start_x, width - window_size))
                
                # 提取最終的裁剪區域（使用原始灰度圖）
                crop = ori_gray[start_y:start_y+window_size, start_x:start_x+window_size]
                
                # 使用時間戳記作為檔名
                timestamp = get_timestamp()
                time.sleep(0.001)  # 確保每個時間戳都不同
                
                # 保存裁剪後的灰階圖片
                output_path = os.path.join(output_dir, f'{timestamp}.png')
                cv2.imwrite(output_path, crop)
                count += 1
    
    return count

def process_directory(input_dir, output_dir, threshold=20, window_size=644, overlap=0.5):
    """
    處理整個目錄中的圖片
    
    參數:
    input_dir: 輸入圖片目錄
    output_dir: 輸出圖片目錄
    threshold: 二值化閾值
    window_size: 切割窗口大小
    overlap: 重疊率(0~1之間)
    """
    # 建立輸出目錄
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支援的圖片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    
    # 處理所有圖片
    total_images = 0
    total_crops = 0
    
    for ext in image_extensions:
        image_paths = glob(os.path.join(input_dir, ext))
        total_images += len(image_paths)
        
        for img_path in image_paths:
            try:
                num_crops = process_and_crop_image(
                    input_path=img_path,
                    output_dir=output_dir,
                    threshold=threshold,
                    window_size=window_size,
                    overlap=overlap,
                    down_scale=15
                )
                total_crops += num_crops
                print(f"成功處理 {img_path}，產生了 {num_crops} 張切割圖片")
            except Exception as e:
                print(f"處理 {img_path} 時發生錯誤: {str(e)}")
    
    return total_images, total_crops

# 使用示例
if __name__ == "__main__":
    input_directory = "D:\\newSGM\\Data\\ColorSlice_Image0"  # 替換為你的輸入圖片目錄
    output_directory = "D:\\newSGM\\Data\\ColorSlice_Image0\\crop"  # 替換為你想要的輸出目錄
    
    try:
        start = time.time()
        total_processed, total_crops = process_directory(
            input_dir=input_directory,
            output_dir=output_directory,
            threshold=20,
            window_size=640,
            overlap=0.3
        )
        end = time.time()
        print(f"\n共耗費 {str(end-start)} 秒")
        print(f"處理完成！")
        print(f"總共處理了 {total_processed} 張原始圖片")
        print(f"總共產生了 {total_crops} 張切割後的圖片")
        print(f"所有圖片都已保存到 {output_directory}")
    except Exception as e:
        print(f"處理過程中發生錯誤: {str(e)}")