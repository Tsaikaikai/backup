import os
import json
import cv2
import numpy as np
import sys

class Aging_detect():
    def __init__(self, image_path, config_file):
        self.config_file = config_file
        self.image_path = image_path


    def load_config(self):
        with open(self.config_file, 'r') as f:
            return json.load(f)
        
    def check_average_value(self,img, roi_points, gray_spec):    

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, roi_points, 255)

        # Get the bounding rectangle of the ROI
        x, y, w, h = cv2.boundingRect(roi_points)

        # Calculate the center of the ROI
        center_x, center_y = x + w // 2, y + h // 2

        # Calculate the size of the 50x50 region
        size = min(100, w, h)
        half_size = size // 2

        # Define the region of interest
        roi_img = img[max(0, center_y - half_size):min(img.shape[0], center_y + half_size),
                    max(0, center_x - half_size):min(img.shape[1], center_x + half_size)]

        # Convert the ROI to HSV
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # Calculate the mean V value
        v_mean = np.mean(hsv_roi[:,:,2])
        
        # 判斷是否為黑色或灰色
        # Check if average gray is within the specified range
        if gray_spec[0] <= v_mean <= gray_spec[1]:
            return True
        else:
            return False  # Return False if no pixels in ROI

    def detect_rainbow(self,roi):
        # Convert ROI to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for red (two ranges), green, blue, white
        lower_colors = [
            (160, 40, 100),  # Red (lower range 2)
            (40, 40, 100),   # Green
            (100, 40, 100),  # Blue
        ]
        upper_colors = [
            (180, 255, 255),  # Red (upper range 2)
            (80, 255, 255),   # Green
            (140, 255, 255),  # Blue
        ] 
        
        # Check if colors are present
        color_presence = []
        for lower, upper in zip(lower_colors, upper_colors):
            mask = cv2.inRange(hsv, lower, upper)
            if np.any(mask):
                color_presence.append(True)
            else:
                color_presence.append(False)
        
        # Define the expected order of colors (red, green, blue, white)
        expected_order = [0, 1, 2]
        
        # Check if all required colors are present
        if not all(color_presence):
            return False
        
        # Check if colors are in the correct order
        color_positions = []
        for i, present in enumerate(color_presence):
            if present:
                # Calculate the average position of each color
                y, x = np.where(cv2.inRange(hsv, lower_colors[i], upper_colors[i]))
                if len(y) > 0:
                    color_positions.append((i, np.mean(y)))
        
        # Sort colors by their vertical position
        color_positions.sort(key=lambda x: x[1])
        
        # Check if the order matches the expected order
        actual_order = [pos[0] for pos in color_positions]
        return actual_order == expected_order

    def check_hsv_and_size(self,roi, hsv_lower, hsv_upper, area_spec, width_spec, height_spec):
        # Convert ROI to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask for the specified HSV range
        mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area, width, and height of the contour
        area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Check if area, width, and height are within specified ranges
        if (area_spec[0] <= area <= area_spec[1] and
            width_spec[0] <= w <= width_spec[1] and
            height_spec[0] <= h <= height_spec[1]):
            return True
        
        return False

    def process_image(self,image_path, config, result_folder, image_no):
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取圖像: {image_path}")
            return None, None

        img = cv2.blur(img, (5, 5))
        results = []

        for i, obj in enumerate(config['obj_list'], 1):
            roi_points = np.array(obj['roi_pox'], np.int32)
            roi_points = roi_points.reshape((-1, 1, 2))
            
            if not self.check_average_value(img, roi_points, config['obj_gray_spec']):
                result = "Black"
            else:
                mask = np.zeros(img.shape[:2], np.uint8)
                cv2.fillPoly(mask, [roi_points], 255)
                roi = cv2.bitwise_and(img, img, mask=mask)
                
                if self.detect_rainbow(roi):
                    result = "Rainbow"
                else:
                    l255_config = next((item for item in config['detect_list'] if item["type_name"] == "L255"), None)
                    if l255_config:
                        if self.check_hsv_and_size(roi, l255_config['HSV_LOWER'], l255_config['HSV_UPPER'],
                                            obj['area_spec'], obj['width_spec'], obj['height_spec']):
                            result = "White"
                        else:
                            result = "NG"
                            self.save_ng_roi(image_path, obj['roi_pox'], i, image_no, result_folder)
                    else:
                        result = "NG"
                        self.save_ng_roi(image_path, obj['roi_pox'], i, image_no, result_folder)
            
            results.append(result)
        
        return results, img.shape[:2]
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None, None

        img = cv2.blur(img, (5, 5))
        results = []

        for obj in config['obj_list']:
            roi_points = np.array(obj['roi_pox'], np.int32)
            roi_points = roi_points.reshape((-1, 1, 2))
            
            if not check_average_value(img, roi_points, config['obj_gray_spec']):
                result = "Black"
            else:
                mask = np.zeros(img.shape[:2], np.uint8)
                cv2.fillPoly(mask, [roi_points], 255)
                roi = cv2.bitwise_and(img, img, mask=mask)
                
                if detect_rainbow(roi):
                    result = "Rainbow"
                else:
                    l255_config = next((item for item in config['detect_list'] if item["type_name"] == "L255"), None)
                    if l255_config:
                        if check_hsv_and_size(roi, l255_config['HSV_LOWER'], l255_config['HSV_UPPER'],
                                            obj['area_spec'], obj['width_spec'], obj['height_spec']):
                            result = "White"
                        else:
                            result = "NG"
                    else:
                        result = "NG"
            
            results.append(result)
        
        return results, img.shape[:2]

    def determine_final_result(self,all_results):
        num_rois = len(all_results[list(all_results.keys())[0]])
        final_results = ["OK"] * num_rois

        for roi_index in range(num_rois):
            ng_count = 0
            for image_results in all_results.values():
                if image_results[roi_index] == "NG":
                    ng_count += 1
                else:
                    ng_count = 0
                
                if ng_count == 2:
                    final_results[roi_index] = "NG"
                    break

        return final_results

    def create_result_image(self,config, final_results, image_size, result_folder):
        result_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        for obj, result in zip(config['obj_list'], final_results):
            roi_points = np.array(obj['roi_pox'], np.int32)
            roi_points = roi_points.reshape((-1, 1, 2))
            
            color = (0, 255, 0) if result == "OK" else (0, 0, 255)
            cv2.polylines(result_img, [roi_points], True, color, 2)

        result_path = os.path.join(result_folder, 'result.jpg')
        cv2.imwrite(result_path, result_img)

    def create_result_json(self,config, final_results, all_results, result_folder):
        result_json = []
        for i, (obj, result) in enumerate(zip(config['obj_list'], final_results), 1):
            ng_images = [j+1 for j, img_results in enumerate(all_results.values()) if img_results[i-1] == "NG"]
            obj_result = {
                "obj_no": str(i),
                "roi_pox": obj['roi_pox'],
                "area_spec": obj['area_spec'],
                "width_spec": obj['width_spec'],
                "height_spec": obj['height_spec'],
                "judge": result,
                "ok_sample": "",
                "judge_detail": [
                    {
                        "type": "NG",
                        "cnt": len(ng_images),
                        "detail": ",".join(map(str, ng_images))
                    }
                ]
            }
            result_json.append(obj_result)
        
        with open(os.path.join(result_folder, 'result.json'), 'w') as f:
            json.dump(result_json, f, indent=4)
        result_json = []
        for i, (obj, result) in enumerate(zip(config['obj_list'], final_results), 1):
            obj_result = {
                "obj_no": str(i),
                "roi_pox": obj['roi_pox'],
                "area_spec": obj['area_spec'],
                "width_spec": obj['width_spec'],
                "height_spec": obj['height_spec'],
                "judge": result,
                "ok_sample": "",
                "judge_detail": [
                    {
                        "type": "NG",
                        "cnt": sum(1 for img_results in all_results.values() if img_results[i-1] == "NG"),
                        "detail": ",".join(str(j+1) for j, img_results in enumerate(all_results.values()) if img_results[i-1] == "NG")
                    }
                ]
            }
            result_json.append(obj_result)
        
        with open(os.path.join(result_folder, 'result.json'), 'w') as f:
            json.dump(result_json, f, indent=4)

    def save_ng_roi(self,image_path, roi_points, obj_no, image_no, result_folder):
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取圖像: {image_path}")
            return

        # 創建遮罩
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        roi_points = np.array(roi_points, np.int32)
        cv2.fillPoly(mask, [roi_points], 255)

        # 應用遮罩
        roi = cv2.bitwise_and(img, img, mask=mask)

        # 獲取 ROI 的邊界框
        x, y, w, h = cv2.boundingRect(roi_points)
        roi_cropped = roi[y:y+h, x:x+w]

        # 儲存切割後的 ROI
        filename = f"obj_{obj_no}_NG_{image_no:03d}.jpeg"
        output_path = os.path.join(result_folder, filename)
        cv2.imwrite(output_path, roi_cropped)

    def main(self):
        config = self.load_config()
        
        if not os.path.exists('output'):
            os.makedirs('output')
        
        result_folder = os.path.join(self.image_path, 'result')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
        all_results = {}
        image_size = None
        
        for image_no, filename in enumerate(sorted(os.listdir(self.image_path)), 1):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.image_path, filename)
                results, img_size = self.process_image(image_path, config, result_folder, image_no)
                if results is not None:
                    all_results[filename] = results
                    if image_size is None:
                        image_size = img_size
                    print(f"已處理: {filename}, 結果: {results}")
        
        final_results = self.determine_final_result(all_results)
        self.create_result_image(config, final_results, image_size, result_folder)
        
        # 創建並保存結果 JSON
        self.create_result_json(config, final_results, all_results, result_folder)
        
        print(f"最終結果: {final_results}")


if __name__ == '__main__':
    
    image_path = sys.argv[1]
    config_file = sys.argv[2]

    obj = Aging_detect(image_path, config_file)
    obj.main()
'''

if __name__ == "__main__":
    
    image_path = './dataset/Exposition/capB'
    config_file = '20241018_S17_Exposition2.json'

    obj = Aging_detect(image_path, config_file)
    obj.main()
'''
