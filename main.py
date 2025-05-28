import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_golden_image(image, ksize=15):
    """
    使用中值濾波器創建一個均勻乾淨的golden image
    """
    return cv2.medianBlur(image, ksize)

def reduce_noise(image, bilateral_d=9, bilateral_color=75, bilateral_space=75, gaussian_ksize=(5,5)):
    """
    使用多種濾波方法減少圖像中的雜訊
    """
    # 先使用雙邊濾波保留邊緣同時減少雜訊
    denoised = cv2.bilateralFilter(image, bilateral_d, bilateral_color, bilateral_space)
    
    # 輕微的高斯模糊進一步平滑
    denoised = cv2.GaussianBlur(denoised, gaussian_ksize, 0)
    
    return denoised

def detect_scratches(original, golden, threshold=30, use_adaptive=False):
    """
    通過比較原始圖像和golden image來檢測刮傷，
    支援自適應閾值處理
    """
    # 計算差異
    diff = cv2.absdiff(original, golden)
    
    # 轉換為灰度圖像(如果是彩色的)
    if len(diff.shape) == 3:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff
    
    # 應用額外的高斯模糊以減少細微雜訊
    diff_gray = cv2.GaussianBlur(diff_gray, (9, 9), 0)
    
    if use_adaptive:
        # 使用自適應閾值處理，更好地處理不均勻照明和紋理差異
        binary = cv2.adaptiveThreshold(
            diff_gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            threshold
        )
    else:
        # 使用全局閾值
        _, binary = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 使用形態學操作來減少噪聲，先開運算後閉運算
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def texture_removal(image, kernel_size=7):
    """
    使用形態學操作移除紋理
    """
    # 使用形態學開運算去除小物體
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # 使用形態學頂帽操作突出細節
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    # 調整對比度，使刮傷更明顯
    enhanced = cv2.addWeighted(image, 1, tophat, 0.5, 0)
    
    return enhanced

def find_scratches(binary, min_area=50, min_length=20):
    """
    從二值化圖像中找出刮傷輪廓，
    過濾基於面積和長軸比的雜訊
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    significant_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 過濾太小的區域
        if area < min_area:
            continue
            
        # 計算輪廓的最小外接矩形
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        
        # 計算長寬比
        aspect_ratio = max(width, height) / (min(width, height) if min(width, height) > 0 else 1)
        
        # 計算最大尺寸
        max_dimension = max(width, height)
        
        # 過濾非線性的輪廓 (長寬比太小或太大的可能是雜訊)
        # 同時過濾太短的線 (可能是產品顆粒，而非刮傷)
        if aspect_ratio > 2.0 and max_dimension > min_length:
            significant_contours.append(cnt)
    
    return significant_contours

def mark_scratches(image, contours):
    """
    在原始圖像上標記刮傷
    """
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
    
    # 添加邊界框和標籤
    for i, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
        
        # 計算面積並顯示
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)

    
    return result

def load_image(image_path):
    """
    加載圖像，處理路徑問題
    """
    # 讀取圖像 (處理Windows路徑)
    image_path = image_path.replace('\\', '/')
    print(f"嘗試讀取圖像: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        # 嘗試其他編碼方式
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
    
    print(f"成功讀取圖像，尺寸: {image.shape}")
    return image

def process_image(image_path, mask_path=None, ksize=15, threshold=30, use_adaptive=False, min_area=50, min_length=20):
    """
    處理圖像並檢測刮傷，可選擇性使用mask限制檢測區域
    """
    # 讀取原始圖像
    image = load_image(image_path)
    original = image.copy()
    
    # 讀取並處理mask圖像（如果有提供）
    mask = None
    if mask_path and os.path.exists(mask_path):
        print(f"使用mask: {mask_path}")
        mask = load_image(mask_path)
        # 轉換為灰度，並確保是二值化的（白色區域為感興趣區域）
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 確保mask與原圖尺寸一致
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
        print(f"Mask尺寸: {mask.shape}")
        
        # 創建mask的可視化圖像（用於結果顯示）
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_overlay = cv2.addWeighted(image, 0.7, mask_display, 0.3, 0)
    else:
        print("未提供mask或mask不存在，將處理整個圖像")
        # 如果未提供mask，則創建一個全白mask（處理整個圖像）
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        mask_overlay = image.copy()  # 沒有mask時，顯示原始圖像
    
    # 1. 預處理 - 減少雜訊
    denoised = reduce_noise(image)
    
    # 2. 紋理處理 - 移除表面顆粒
    texture_processed = texture_removal(denoised)
    
    # 3. 創建golden image
    golden = create_golden_image(texture_processed, ksize)
    
    # 4. 檢測刮傷 (不使用mask)
    binary_no_mask = detect_scratches(texture_processed, golden, threshold, use_adaptive)
    
    # 存儲無mask時的刮傷輪廓
    scratches_no_mask = find_scratches(binary_no_mask, min_area, min_length)
    marked_no_mask = mark_scratches(original, scratches_no_mask)
    
    # 5. 應用mask：只保留mask中白色區域的檢測結果
    binary_with_mask = cv2.bitwise_and(binary_no_mask, binary_no_mask, mask=mask)
    
    # 6. 找出刮傷輪廓 (使用mask)
    scratches_with_mask = find_scratches(binary_with_mask, min_area, min_length)
    
    # 7. 標記刮傷 (使用mask)
    marked_with_mask = mark_scratches(original, scratches_with_mask)
    
    # 創建比較圖像：左半邊無mask，右半邊有mask
    h, w = original.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = marked_no_mask
    comparison[:, w:] = marked_with_mask
    
    # 添加分隔線
    cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
    
    return {
        'original': original,
        'denoised': denoised,
        'texture_processed': texture_processed,
        'golden': golden,
        'binary_no_mask': binary_no_mask,
        'binary_with_mask': binary_with_mask,
        'marked_no_mask': marked_no_mask,
        'marked_with_mask': marked_with_mask,
        'mask_overlay': mask_overlay,
        'comparison': comparison,
        'mask': mask
    }

def display_results(results):
    """
    顯示處理結果
    """
    # 顯示原始處理結果
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    
    # 原始圖像
    axes1[0, 0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    axes1[0, 0].set_title('original')
    axes1[0, 0].axis('off')
    
    # Mask疊加
    axes1[0, 1].imshow(cv2.cvtColor(results['mask_overlay'], cv2.COLOR_BGR2RGB))
    axes1[0, 1].set_title('mask_overlay')
    axes1[0, 1].axis('off')
    
    # 紋理處理後的圖像
    axes1[0, 2].imshow(cv2.cvtColor(results['texture_processed'], cv2.COLOR_BGR2RGB))
    axes1[0, 2].set_title('texture_processed')
    axes1[0, 2].axis('off')
    
    # Golden image
    axes1[1, 0].imshow(cv2.cvtColor(results['golden'], cv2.COLOR_BGR2RGB))
    axes1[1, 0].set_title('Golden Image')
    axes1[1, 0].axis('off')
    
    # 二值化差異圖像 (有mask)
    axes1[1, 1].imshow(results['texture_processed'], cmap='gray')
    axes1[1, 1].set_title('texture_processed')
    axes1[1, 1].axis('off')
    
    # 標記刮傷的圖像 (有mask)
    axes1[1, 2].imshow(cv2.cvtColor(results['marked_with_mask'], cv2.COLOR_BGR2RGB))
    axes1[1, 2].set_title('marked_with_mask')
    axes1[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 顯示有無mask對比結果
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    
    # 無mask檢測二值化結果
    axes2[0, 0].imshow(results['binary_no_mask'], cmap='gray')
    axes2[0, 0].set_title('binary_no_mask')
    axes2[0, 0].axis('off')
    
    # 有mask檢測二值化結果
    axes2[0, 1].imshow(results['binary_with_mask'], cmap='gray')
    axes2[0, 1].set_title('binary_with_mask')
    axes2[0, 1].axis('off')
    
    # 無mask檢測標記結果
    axes2[1, 0].imshow(cv2.cvtColor(results['marked_no_mask'], cv2.COLOR_BGR2RGB))
    axes2[1, 0].set_title('marked_no_mask')
    axes2[1, 0].axis('off')
    
    # 有mask檢測標記結果
    axes2[1, 1].imshow(cv2.cvtColor(results['marked_with_mask'], cv2.COLOR_BGR2RGB))
    axes2[1, 1].set_title('marked_with_mask')
    axes2[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 直接對比視圖
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(results['comparison'], cv2.COLOR_BGR2RGB))
    plt.title('comparison')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_results(results, output_dir='./'):
    """
    保存處理結果
    """
    cv2.imwrite(f"{output_dir}/original.jpg", results['original'])
    cv2.imwrite(f"{output_dir}/mask_overlay.jpg", results['mask_overlay'])
    cv2.imwrite(f"{output_dir}/denoised.jpg", results['denoised'])
    cv2.imwrite(f"{output_dir}/texture_processed.jpg", results['texture_processed'])
    cv2.imwrite(f"{output_dir}/golden.jpg", results['golden'])
    cv2.imwrite(f"{output_dir}/binary_no_mask.jpg", results['binary_no_mask'])
    cv2.imwrite(f"{output_dir}/binary_with_mask.jpg", results['binary_with_mask'])
    cv2.imwrite(f"{output_dir}/marked_no_mask.jpg", results['marked_no_mask'])
    cv2.imwrite(f"{output_dir}/marked_with_mask.jpg", results['marked_with_mask'])
    cv2.imwrite(f"{output_dir}/comparison.jpg", results['comparison'])
    print(f"結果已保存至 {output_dir} 目錄")

def compare_mask(image_path, mask_path, output_dir=None, ksize=15, threshold=30, use_adaptive=False, min_area=50, min_length=20, show_results=True):
    """
    比較有無mask對檢測結果的影響
    """
    print("開始處理圖像...")
    
    # 檢查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到圖像文件: {image_path}")
        
    has_mask = mask_path and os.path.exists(mask_path)
    
    if not has_mask:
        print(f"警告: 找不到mask文件: {mask_path}")
        print("將只顯示無mask處理結果")
    
    # 處理圖像
    results = process_image(
        image_path, 
        mask_path if has_mask else None,
        ksize, 
        threshold, 
        use_adaptive,
        min_area,
        min_length
    )
    
    # 顯示結果
    if show_results:
        display_results(results)
    
    # 保存結果
    if output_dir:
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        save_results(results, output_dir)
        print(f"已保存比較結果到: {output_dir}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    # 預設使用路徑
    default_path = r"C:\Users\User\Desktop\compare_AOI\test1.png"
    default_mask_path = r"C:\Users\User\Desktop\compare_AOI\Mask1.png"
    
    parser = argparse.ArgumentParser(description='檢測圖像中的刮傷並比較有無mask的差異')
    parser.add_argument('--image_path', type=str, default=default_path, help='要處理的圖像路徑')
    parser.add_argument('--mask_path', type=str, default=default_mask_path, help='遮罩圖像路徑，白色區域為感興趣區域')
    parser.add_argument('--ksize', type=int, default=35, help='中值濾波器核大小 (必須是奇數)')
    parser.add_argument('--threshold', type=int, default=7, help='差異檢測閾值 (0-255)')
    parser.add_argument('--adaptive', action='store_true', help='使用自適應閾值而非全局閾值')
    parser.add_argument('--min_area', type=int, default=30, help='最小輪廓面積')
    parser.add_argument('--min_length', type=int, default=15, help='最小刮傷長度')
    parser.add_argument('--save', type=bool, default=True, help='保存結果圖像')
    parser.add_argument('--output_dir', type=str, default=os.path.dirname(default_path), help='輸出目錄')
    parser.add_argument('--no-show', action='store_true', help='不顯示結果，僅保存')
    parser.add_argument('--compare', action='store_true', help='執行有無mask的結果比較')
    
    args = parser.parse_args()
    
    print(f"處理圖像: {args.image_path}")
    print(f"使用mask: {args.mask_path}")
    print(f"輸出目錄: {args.output_dir}")
    
    # 確保ksize是奇數
    if args.ksize % 2 == 0:
        args.ksize += 1
    
    try:
        # 執行帶比較的處理
        if args.compare:
            compare_mask(
                args.image_path,
                args.mask_path,
                args.output_dir if args.save else None,
                args.ksize,
                args.threshold,
                args.adaptive,
                args.min_area,
                args.min_length,
                not args.no_show
            )
        else:
            # 執行常規處理
            results = process_image(
                args.image_path, 
                args.mask_path,
                args.ksize, 
                args.threshold, 
                args.adaptive,
                args.min_area,
                args.min_length
            )
            
            if not args.no_show:
                display_results(results)
            
            if args.save:
                # 確保輸出目錄存在
                os.makedirs(args.output_dir, exist_ok=True)
                save_results(results, args.output_dir)
                
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()