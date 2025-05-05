import numpy as np
import cv2
from scipy import ndimage, fft
from matplotlib import pyplot as plt
import os

class SSOModel:
    """
    Spatial Standard Observer (SSO) 模型實現
    用於計算面板中 Mura 缺陷的 JND 值
    """
    
    def __init__(self, viewing_distance_cm=50, screen_ppi=96, screen_width_cm=34.5):
        """
        初始化 SSO 模型
        
        參數:
        viewing_distance_cm: 觀察距離，單位為厘米
        screen_ppi: 螢幕像素密度，每英寸像素數
        screen_width_cm: 螢幕寬度，單位為厘米
        """
        self.viewing_distance_cm = viewing_distance_cm
        self.screen_ppi = screen_ppi
        self.screen_width_cm = screen_width_cm
        
        # 計算每度視角的像素數
        self.pixels_per_degree = self._calculate_pixels_per_degree()
        
        # SSO 模型參數（這些參數可以根據視覺實驗進行調整）
        self.psf_sigma = 0.5  # PSF 高斯函數標準差 (視角度)
        self.csf_peak = 4.0   # CSF 峰值頻率 (cycles/degree)
        self.csf_bandwidth = 1.5  # CSF 帶寬
        self.gain = 270.0     # 增益係數
        self.exponent = 2.0   # Minkowski 池化指數
    
    def _calculate_pixels_per_degree(self):
        """計算每度視角包含的像素數"""
        # 每英寸厘米數
        cm_per_inch = 2.54
        
        # 計算每度視角的像素數
        # 視角(度) = 2 * arctan(對象尺寸/(2*距離))
        screen_width_inch = self.screen_width_cm / cm_per_inch
        screen_width_pixels = screen_width_inch * self.screen_ppi
        screen_width_degrees = 2 * np.arctan(self.screen_width_cm / (2 * self.viewing_distance_cm)) * 180 / np.pi
        
        return screen_width_pixels / screen_width_degrees
    
    def apply_psf(self, image):
        """
        應用點擴散函數 (PSF)
        模擬眼球光學系統的模糊效應
        """
        # 計算高斯濾波器的標準差（像素單位）
        sigma_pixels = self.psf_sigma * self.pixels_per_degree
        
        # 應用高斯濾波
        blurred_image = ndimage.gaussian_filter(image, sigma=sigma_pixels)
        
        return blurred_image
    
    def apply_csf(self, image):
        """
        應用對比敏感度函數 (CSF)
        模擬視覺系統對不同空間頻率的敏感度
        """
        # 轉換到頻域
        image_fft = fft.fft2(image)
        image_fft_shifted = fft.fftshift(image_fft)
        
        # 創建頻率網格
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # 創建二維網格，表示從圖像中心開始的距離
        y, x = np.mgrid[-crow:rows-crow, -ccol:cols-ccol]
        
        # 轉換為循環/度 (cycles/degree)
        freq_y = y / (rows * (1/self.pixels_per_degree))
        freq_x = x / (cols * (1/self.pixels_per_degree))
        freq = np.sqrt(freq_x**2 + freq_y**2)
        
        # 計算 CSF（使用對數高斯模型）
        log_freq = np.log(freq + 1e-6)  # 避免 log(0)
        log_peak = np.log(self.csf_peak)
        csf = self.gain * np.exp(-(log_freq - log_peak)**2 / (2 * self.csf_bandwidth**2))
        
        # 處理零頻率點
        csf[crow, ccol] = 0
        
        # 應用 CSF 濾波器
        filtered_fft = image_fft_shifted * csf
        filtered_image = np.abs(fft.ifft2(fft.ifftshift(filtered_fft)))
        
        return filtered_image
    
    def calculate_local_contrast(self, image):
        """
        計算局部對比度並應用非線性處理
        """
        # 計算局部均值
        mean_image = ndimage.gaussian_filter(image, sigma=2*self.pixels_per_degree)
        
        # 避免除以零
        mean_image = np.maximum(mean_image, 1e-6)
        
        # 計算局部對比度 (Weber 對比度)
        contrast = (image - mean_image) / mean_image
        
        # 非線性處理 (簡化版本)
        k = 0.3  # 半飽和常數
        response = contrast / (np.abs(contrast) + k)
        
        return response
    
    def spatial_pooling(self, response_diff):
        """
        執行空間池化操作
        使用 Minkowski 距離計算總體視覺差異
        """
        # 計算 Minkowski 範數
        pooled_response = np.power(
            np.mean(np.power(np.abs(response_diff), self.exponent)), 
            1/self.exponent
        )
        
        return pooled_response
    
    def response_to_jnd(self, pooled_response):
        """
        將池化後的視覺響應轉換為 JND 值
        """
        # 這裡使用簡化的線性映射，實際應用中可能需要基於實驗數據進行校準
        jnd_value = 3.52 * pooled_response
        
        return jnd_value
    
    def process_image(self, image):
        """
        處理單一圖像通過 SSO 模型的所有步驟
        """
        # 1. 應用 PSF
        psf_image = self.apply_psf(image)
        
        # 2. 應用 CSF
        csf_image = self.apply_csf(psf_image)
        
        # 3. 計算局部對比度和視覺響應
        response = self.calculate_local_contrast(csf_image)

        result = {'psf': psf_image, 'csf': csf_image, 'response':response}
        return result
    
    def calculate_jnd(self, test_image, reference_image):
        """
        計算測試圖像與參考圖像之間的 JND 值
        """
        # 確保輸入是浮點數
        test_image = test_image.astype(float)
        reference_image = reference_image.astype(float)
        
        # 將圖像轉換為亮度值 (假設輸入為 8 位灰度圖)
        # 對於實際應用，這裡應該加入正確的亮度校準
        test_luminance = test_image / 255.0 * 100.0  # 假設最大亮度為 100 cd/m^2
        ref_luminance = reference_image / 255.0 * 100.0
        
        # 處理測試和參考圖像
        test_results = self.process_image(test_luminance)
        ref_results = self.process_image(ref_luminance)
        
        # 計算響應差異
        response_diff = test_results['response'] - ref_results['response']
        
        # 空間池化
        pooled_diff = self.spatial_pooling(response_diff)
        
        # 轉換為 JND
        jnd_value = self.response_to_jnd(pooled_diff)
        
        # 創建 JND 地圖 (用於可視化)
        jnd_map = np.abs(response_diff)
        
        # 保存中間處理結果以便分析
        plt.figure(figsize=(15, 10))
        
        # PSF 處理結果
        plt.subplot(2, 3, 1)
        plt.imshow(test_results['psf'], cmap='gray')
        plt.title('Test PSF')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(ref_results['psf'], cmap='gray')
        plt.title('Reference PSF')
        plt.axis('off')
        
        # CSF 處理結果
        plt.subplot(2, 3, 2)
        plt.imshow(test_results['csf'], cmap='gray')
        plt.title('Test CSF')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(ref_results['csf'], cmap='gray')
        plt.title('Reference CSF')
        plt.axis('off')
        
        # 視覺響應
        plt.subplot(2, 3, 3)
        plt.imshow(test_results['response'], cmap='jet')
        plt.title('Test Response')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(ref_results['response'], cmap='jet')
        plt.title('Reference Response')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sso_intermediate_results.png')
        plt.close()
        
        return jnd_value, jnd_map
    
    def visualize_results(self, test_image, reference_image, jnd_map):
        """
        可視化 SSO 分析結果
        """
        plt.figure(figsize=(15, 10))
        
        # 原始圖像
        plt.subplot(2, 2, 1)
        plt.imshow(test_image, cmap='gray')
        plt.title('Test Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(reference_image, cmap='gray')
        plt.title('Reference Image')
        plt.axis('off')
        
        # JND 地圖
        plt.subplot(2, 2, 3)
        plt.imshow(jnd_map, cmap='jet')
        plt.title('JND Map')
        plt.colorbar()
        plt.axis('off')
        
        # 差異圖
        plt.subplot(2, 2, 4)
        diff = test_image.astype(float) - reference_image.astype(float)
        plt.imshow(diff, cmap='coolwarm')
        plt.title('Raw Difference')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sso_visualization.png')
        plt.show()
        
        return jnd_map


def align_images(test_image, reference_image, mark_x=1136, mark_y=292, mark_size=40):
    """
    使用定位點對齊測試圖像和參考圖像
    
    參數:
    test_image: 待測試圖像
    reference_image: 參考圖像
    mark_x, mark_y: 定位點左上角座標
    mark_size: 定位點尺寸
    
    返回:
    aligned_test: 對齊後的測試圖像
    """
    # 提取參考定位點
    mark_region = reference_image[mark_y:mark_y+mark_size, mark_x:mark_x+mark_size]
    
    # 使用模板匹配在測試圖像中找到定位點
    result = cv2.matchTemplate(test_image, mark_region, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    test_mark_x, test_mark_y = max_loc
    
    # 計算偏移量
    offset_x = test_mark_x - mark_x
    offset_y = test_mark_y - mark_y
    
    print(f"檢測到的偏移量: X={offset_x}, Y={offset_y}")
    
    # 執行對齊（平移變換）
    M = np.float32([[1, 0, -offset_x], [0, 1, -offset_y]])
    aligned_test = cv2.warpAffine(test_image, M, (test_image.shape[1], test_image.shape[0]))
    
    return aligned_test

def crop_roi(image, mark_x=1136, mark_y=292, mark_size=40, shift_x=1099, shift_y=181, crop_x=1343, crop_y=840):
    """
    根據定位點裁剪感興趣區域
    
    參數:
    image: 輸入圖像
    mark_x, mark_y: 定位點座標
    mark_size: 定位點尺寸
    roi_size: 感興趣區域尺寸
    
    返回:
    roi: 裁剪後的感興趣區域
    """
    
    # 計算 ROI 的左上角座標
    roi_x = max(0, mark_x - shift_x)
    roi_y = max(0, mark_y - shift_y)
    
    # 確保 ROI 不會超出圖像邊界
    roi_x = min(roi_x, image.shape[1] - crop_x)
    roi_y = min(roi_y, image.shape[0] - crop_y)
    
    # 裁剪 ROI
    roi = image[roi_y:roi_y+crop_y, roi_x:roi_x+crop_x]
    
    return roi, (roi_x, roi_y)

def visualize_mark_and_roi(image, mark_x=1136, mark_y=292, mark_size=40, shift_x=1099, shift_y=181, crop_x=1343, crop_y=840):
    """
    可視化定位點和感興趣區域
    """
    # 複製圖像以避免修改原始圖像
    vis_image = image.copy()
    if len(vis_image.shape) == 2:  # 灰度圖轉為彩色圖
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # 標記定位點 (紅色矩形)
    cv2.rectangle(vis_image, (mark_x, mark_y), (mark_x+mark_size, mark_y+mark_size), (0, 0, 255), 2)
    
    # 計算 ROI 的左上角座標
    roi_x = max(0, mark_x - shift_x)
    roi_y = max(0, mark_y - shift_y)
    
    # 標記 ROI (綠色矩形)
    cv2.rectangle(vis_image, (roi_x, roi_y), (roi_x+crop_x, roi_y+crop_y), (0, 255, 0), 2)
    
    return vis_image

def main():
    # 載入測試和參考圖像
    # 這裡您需要替換為實際的圖像路徑
    test_image_path = "C:\\Users\\User\\Desktop\\mura\\9-JND2.4-02.tif"
    reference_image_path = "C:\\Users\\User\\Desktop\\mura\\9-OK.tif"
    
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    
    # 檢查圖像是否成功載入
    if test_image is None or reference_image is None:
        print("無法載入圖像，請檢查路徑")
        return
    
    # 確保兩張圖像尺寸相同
    if test_image.shape != reference_image.shape:
        print("圖像尺寸不匹配，將調整大小")
        test_image = cv2.resize(test_image, (reference_image.shape[1], reference_image.shape[0]))
    
    # 定位點參數
    mark_x, mark_y = 1136, 292
    mark_size = 40
    shift_x=1060
    shift_y=160
    crop_x=1280
    crop_y=780
    
    # 可視化定位點和 ROI
    vis_ref = visualize_mark_and_roi(reference_image, mark_x, mark_y, mark_size, shift_x, shift_y, crop_x, crop_y)
    
    # 根據參考圖像中的定位點對齊測試圖像
    aligned_test = align_images(test_image, reference_image, mark_x, mark_y, mark_size)
    
    # 可視化對齊結果
    vis_aligned = visualize_mark_and_roi(aligned_test, mark_x, mark_y, mark_size, shift_x, shift_y, crop_x, crop_y)
    
    # 裁剪感興趣區域
    ref_roi, (roi_x, roi_y) = crop_roi(reference_image, mark_x, mark_y, mark_size, shift_x, shift_y, crop_x, crop_y)
    test_roi, _ = crop_roi(aligned_test, mark_x, mark_y, mark_size, shift_x, shift_y, crop_x, crop_y)
    
    # 顯示對齊和裁剪結果
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(vis_ref)
    plt.title('Reference Image with Mark and ROI')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(vis_aligned)
    plt.title('Aligned Test Image with Mark and ROI')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(ref_roi, cmap='gray')
    plt.title('Reference ROI')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(test_roi, cmap='gray')
    plt.title('Test ROI')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('C:/Users/User/Desktop/mura/alignment_results.png')
    plt.show()
    
    # 創建 SSO 模型實例
    # 這裡的參數需要根據實際觀察條件進行設置
    sso = SSOModel(viewing_distance_cm=50, screen_ppi=96, screen_width_cm=34.5)
    
    # 計算 ROI 的 JND
    jnd_value, jnd_map = sso.calculate_jnd(test_roi, ref_roi)
    
    print(f"計算的 Mura JND 值: {jnd_value:.4f}")
    
    # 保存和可視化結果
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(test_roi, cmap='gray')
    plt.title('Test ROI')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(ref_roi, cmap='gray')
    plt.title('Reference ROI')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(jnd_map, cmap='jet')
    plt.title(f'JND Map (Overall JND = {jnd_value:.4f})')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    diff = test_roi.astype(float) - ref_roi.astype(float)
    plt.imshow(diff, cmap='coolwarm')
    plt.title('Raw Difference')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('C:/Users/User/Desktop/mura/sso_results.png')
    plt.show()
    
    # 將結果保存到文件
    result_file = 'C:/Users/User/Desktop/mura/jnd_result.txt'
    with open(result_file, 'a+') as f:
        f.write(f"圖像: {os.path.basename(test_image_path)}\n")
        f.write(f"參考: {os.path.basename(reference_image_path)}\n")
        f.write(f"定位點座標: ({mark_x}, {mark_y})\n")
        f.write(f"ROI 座標: ({roi_x}, {roi_y}) 大小: {crop_x}x{crop_y}\n")
        f.write(f"Mura JND 值: {jnd_value:.4f}\n")
    
    print(f"結果已保存到 {result_file}")



if __name__ == "__main__":
    main()