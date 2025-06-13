import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class OptimizedRainbowDetector:
    def __init__(self):
        # 預定義HSV範圍
        self.color_ranges = {
            'red': [(np.array([0,250,190]), np.array([8,255,255])),
                   (np.array([148,250,190]), np.array([179,255,255]))],
            'green': [(np.array([53,250,190]), np.array([93,255,255]))],
            'blue': [(np.array([99,250,190]), np.array([139,255,255]))]
        }
        self.min_contour_size = 80
        self.min_separation = 100
        self.data = None  # 需要設置你的data
    
    def create_color_masks_vectorized(self, img_hsv):
        """向量化創建所有顏色遮罩"""
        masks = {}
        
        # 紅色遮罩（兩個範圍）
        mask_red1 = cv2.inRange(img_hsv, self.color_ranges['red'][0][0], self.color_ranges['red'][0][1])
        mask_red2 = cv2.inRange(img_hsv, self.color_ranges['red'][1][0], self.color_ranges['red'][1][1])
        masks['red'] = cv2.bitwise_or(mask_red1, mask_red2)
        
        # 綠色和藍色遮罩
        masks['green'] = cv2.inRange(img_hsv, self.color_ranges['green'][0][0], self.color_ranges['green'][0][1])
        masks['blue'] = cv2.inRange(img_hsv, self.color_ranges['blue'][0][0], self.color_ranges['blue'][0][1])
        
        return masks
    
    def get_color_x_coordinates(self, mask, min_panels_ratio=0.5, existing_panels=0):
        """獲取顏色區域的x座標"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        x_coordinates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > self.min_contour_size or h > self.min_contour_size:
                x_coordinates.append(x)
        
        # 檢查是否有足夠的面板
        if len(x_coordinates) > existing_panels * min_panels_ratio:
            return np.mean(x_coordinates) if x_coordinates else None
        return None
    
    def check_rainbow_optimized(self, img_hsv, existing_panels=0):
        """優化的彩虹檢測"""
        # 一次性創建所有顏色遮罩
        masks = self.create_color_masks_vectorized(img_hsv)
        
        # 並行計算各顏色的平均x座標
        color_positions = {}
        for color in ['red', 'green', 'blue']:
            x_avg = self.get_color_x_coordinates(masks[color], existing_panels=existing_panels)
            if x_avg is None:
                return False
            color_positions[color] = x_avg
        
        # 檢查彩虹順序：綠 < 紅 < 藍，且跨度 > 100
        green_x = color_positions['green']
        red_x = color_positions['red']
        blue_x = color_positions['blue']
        
        return (green_x < red_x < blue_x) and (blue_x - green_x) > self.min_separation
    
    def process_single_roi(self, args):
        """處理單個ROI的函數（用於多進程）"""
        image, roi_pox, height, width = args
        
        # 建立遮罩
        mask = np.zeros((height, width), dtype=np.uint8)
        roi_points = np.array(roi_pox, dtype=np.int32)
        cv2.fillPoly(mask, [roi_points], 255)
        
        # 應用遮罩並轉換為HSV
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        img_hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        
        # 檢測彩虹
        return "Rainbow" if self.check_rainbow_optimized(img_hsv, 0) else "Null"
    
    def analysis_rainbow_pattern_optimized(self, image, use_multiprocessing=True, num_workers=None):
        """優化的彩虹模式分析"""
        height, width = image.shape[:2]
        
        if not use_multiprocessing or len(self.data["obj_list"]) < 4:
            # 小數據量使用單線程
            results = []
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 預先轉換HSV
            
            for obj in self.data["obj_list"]:
                mask = np.zeros((height, width), dtype=np.uint8)
                roi_pox = obj["roi_pox"]
                roi_points = np.array(roi_pox, dtype=np.int32)
                cv2.fillPoly(mask, [roi_points], 255)
                
                # 應用遮罩到HSV圖像
                masked_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
                
                if self.check_rainbow_optimized(masked_hsv, 0):
                    results.append("Rainbow")
                else:
                    results.append("Null")
            
            return results
        
        else:
            # 大數據量使用多進程
            if num_workers is None:
                num_workers = min(mp.cpu_count(), len(self.data["obj_list"]))
            
            # 準備參數
            args_list = []
            for obj in self.data["obj_list"]:
                args_list.append((image, obj["roi_pox"], height, width))
            
            # 使用進程池
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(self.process_single_roi, args_list))
            
            return results
    
    def analysis_rainbow_pattern_batch_optimized(self, image, batch_size=10):
        """批次處理優化版本"""
        height, width = image.shape[:2]
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        results = []
        obj_list = self.data["obj_list"]
        
        # 批次處理
        for i in range(0, len(obj_list), batch_size):
            batch = obj_list[i:i+batch_size]
            batch_results = []
            
            # 為這個批次創建所有遮罩
            batch_masks = []
            for obj in batch:
                mask = np.zeros((height, width), dtype=np.uint8)
                roi_points = np.array(obj["roi_pox"], dtype=np.int32)
                cv2.fillPoly(mask, [roi_points], 255)
                batch_masks.append(mask)
            
            # 批次處理HSV轉換和檢測
            for mask in batch_masks:
                masked_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
                
                if self.check_rainbow_optimized(masked_hsv, 0):
                    batch_results.append("Rainbow")
                else:
                    batch_results.append("Null")
            
            results.extend(batch_results)
        
        return results

# 使用範例
def example_usage():
    detector = OptimizedRainbowDetector()
    # detector.data = your_data  # 設置你的資料
    
    # 方法1：多進程版本（適合大量ROI）
    # results = detector.analysis_rainbow_pattern_optimized(image, use_multiprocessing=True)
    
    # 方法2：批次處理版本（適合中等數量ROI）
    # results = detector.analysis_rainbow_pattern_batch_optimized(image, batch_size=20)
    
    # 方法3：單線程優化版本（適合少量ROI）
    # results = detector.analysis_rainbow_pattern_optimized(image, use_multiprocessing=False)
    
    pass
