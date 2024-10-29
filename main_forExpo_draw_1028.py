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
        hsize = min(100, h)
        wsize = min(150, w)
        half_hsize = hsize // 2
        half_wsize = wsize // 2

        # Define the region of interest
        roi_img = img[max(0, center_y - half_hsize):min(img.shape[0], center_y + half_hsize),
                    max(0, center_x - half_wsize):min(img.shape[1], center_x + half_wsize)]

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

    def process_image(self, image_path, config, result_folder, image_no):
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取圖像: {image_path}")
            return None, None
        
        # 創建結果圖的副本
        result_img = img.copy()
        img = cv2.blur(img, (5, 5))
        results = []

        # 定義結果顏色對應字典 (BGR格式)
        result_colors = {
            "Black": (200, 200, 200),
            "Rainbow": (255, 0, 255),  # 紫色
            "White": (10, 10, 10),
            "NG": (0, 0, 255)  # 紅色
        }

        # 定義文字參數
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 2
        last_result = "NULL"

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
                            #self.save_ng_roi(image_path, obj['roi_pox'], i, "Pattern_NG", image_no-1, result_folder)
                    else:
                        result = "NG"
                        #self.save_ng_roi(image_path, obj['roi_pox'], i, "Pattern_NG", image_no-1, result_folder)
            
            results.append(result)
            
            # 在結果圖上繪製ROI區域和結果
            color = result_colors.get(result, (0, 0, 0))
            cv2.polylines(result_img, [roi_points], True, (255, 255, 255), 2)
            
            # 計算ROI中心點來放置文字
            M = cv2.moments(roi_points)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = roi_points[0][0]
            
            # 在ROI中心放置結果文字
            text_size = cv2.getTextSize(result, font, font_scale, thickness)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            cv2.putText(result_img, result, (text_x, text_y), font, font_scale, color, thickness)
        
        # 儲存結果圖
        result_filename = f"result_{os.path.basename(image_path)}"
        result_path = os.path.join(result_folder, result_filename)
        cv2.imwrite(result_path, result_img)
        
        return results, img.shape[:2]

    def determine_final_result(self, all_results):
        num_rois = len(all_results[list(all_results.keys())[0]])
        final_results = ["OK"] * num_rois

        for roi_index in range(num_rois):
            ng_count = 0
            black_count = 0
            for image_results in all_results.values():
                if image_results[roi_index] == "NG":
                    ng_count += 1
                    black_count = 0
                elif image_results[roi_index] == "Black":
                    black_count += 1
                    ng_count = 0
                else:
                    ng_count = 0
                    black_count = 0
                
                if ng_count == 2:
                    final_results[roi_index] = "NG"
                    break
                elif black_count >= black_continue:
                    final_results[roi_index] = "NG"
                    break

        return final_results

    def create_result_image_o(self,config, final_results, all_results, image_size, result_folder):

        result_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        for obj, result in zip(config['obj_list'], final_results):

            roi_points = np.array(obj['roi_pox'], np.int32)
            roi_points = roi_points.reshape((-1, 1, 2))
            
            if result == "OK":
                color = (0, 255, 0) 
                cv2.polylines(result_img, [roi_points], True, color, 5)
                cv2.putText(result_img, obj['obj_no'] + '. OK', roi_points[0][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 255), 2)


            else :
                color = (0, 0, 255)
                cv2.polylines(result_img, [roi_points], True, color, 5)
                cv2.putText(result_img, obj['obj_no'] + '. NG', roi_points[0][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 255), 2)

        result_path = os.path.join(result_folder, 'result.jpg')
        cv2.imwrite(result_path, result_img)

    def create_result_image(self, config, final_results, all_results, image_size, result_folder):
        result_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        
        # 遍歷每個 obj 和其對應的結果
        for i, (obj, result) in enumerate(zip(config['obj_list'], final_results)):
            roi_points = np.array(obj['roi_pox'], np.int32)
            roi_points = roi_points.reshape((-1, 1, 2))
            
            # 取得該 ROI 所有圖片的結果
            roi_results = [img_results[i] for img_results in all_results.values()]
            image_files = list(all_results.keys())
            
            if result == "OK":
                # 尋找最長連續的 White 序列
                max_white_seq = []
                current_white_seq = []
                
                for j, res in enumerate(roi_results):
                    if res == "White":
                        current_white_seq.append(j)
                    else:
                        if len(current_white_seq) > len(max_white_seq):
                            max_white_seq = current_white_seq.copy()
                        current_white_seq = []
                
                if len(current_white_seq) > len(max_white_seq):
                    max_white_seq = current_white_seq.copy()
                    
                # 如果有 White 結果
                if max_white_seq:
                    # 選取最長連續序列的中間位置
                    mid_idx = max_white_seq[len(max_white_seq)//2]
                    selected_image = image_files[mid_idx]
                else:
                    # 尋找最長連續的 Black 序列
                    max_black_seq = []
                    current_black_seq = []
                    
                    for j, res in enumerate(roi_results):
                        if res == "Black":
                            current_black_seq.append(j)
                        else:
                            if len(current_black_seq) > len(max_black_seq):
                                max_black_seq = current_black_seq.copy()
                            current_black_seq = []
                    
                    if len(current_black_seq) > len(max_black_seq):
                        max_black_seq = current_black_seq.copy()
                    
                    # 選取最長連續序列的中間位置
                    mid_idx = max_black_seq[len(max_black_seq)//2]
                    selected_image = image_files[mid_idx]
            else:  # NG case
                # 尋找最長連續的 NG 序列
                max_ng_seq = []
                current_ng_seq = []
                
                for j, res in enumerate(roi_results):
                    if res == "NG":
                        current_ng_seq.append(j)
                    else:
                        if len(current_ng_seq) > len(max_ng_seq):
                            max_ng_seq = current_ng_seq.copy()
                        current_ng_seq = []
                
                if len(current_ng_seq) > len(max_ng_seq):
                    max_ng_seq = current_ng_seq.copy()
                
                # 如果找到 NG 序列
                if max_ng_seq:
                    # 選取最長連續序列的中間位置
                    mid_idx = max_ng_seq[len(max_ng_seq)//2]
                    selected_image = image_files[mid_idx]
                else:
                    # 如果找不到 NG 序列，尋找最長連續的 Black 序列
                    max_black_seq = []
                    current_black_seq = []
                    
                    for j, res in enumerate(roi_results):
                        if res == "Black":
                            current_black_seq.append(j)
                        else:
                            if len(current_black_seq) > len(max_black_seq):
                                max_black_seq = current_black_seq.copy()
                            current_black_seq = []
                    
                    if len(current_black_seq) > len(max_black_seq):
                        max_black_seq = current_black_seq.copy()
                    
                    # 選取最長連續序列的中間位置
                    mid_idx = max_black_seq[len(max_black_seq)//2]
                    selected_image = image_files[mid_idx]
            
            # 讀取選中的圖片
            img_path = os.path.join(self.image_path, selected_image)
            img = cv2.imread(img_path)
            
            # 創建遮罩並取得 ROI
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_points], 255)
            roi = cv2.bitwise_and(img, img, mask=mask)
            
            # 將 ROI 貼到結果圖上
            result_img = cv2.add(result_img, roi)
            
            # 繪製邊界和結果文字
            if result == "OK":
                color = (0, 255, 0)  # Green for OK
            else:
                color = (0, 0, 255)  # Red for NG
                
            putposition = roi_points[3][0] + [0,-30]
            if putposition[1]<0 : putposition[1]=0

            cv2.polylines(result_img, [roi_points], True, color, 5)
            cv2.putText(result_img, 'No.' + obj['obj_no'] + f'  {result}', 
                    putposition, 
                    cv2.FONT_HERSHEY_DUPLEX, 2, 
                    color, 2)
        
        # 保存結果圖片
        result_path = os.path.join(result_folder, 'result.jpg')
        cv2.imwrite(result_path, result_img)

    def create_result_json(self, config, final_results, all_results, result_folder):
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
                "judge_detail": []
            }

            # NG判断
            ng_count = sum(1 for img_results in all_results.values() if img_results[i-1] == "NG")
            ng_sequence = []
            last_result = 'NULL'
            if ng_count > 0:
                for j, img_results in enumerate(all_results.values()) :
                    if last_result == "NG" and img_results[i-1] == "NG" :
                        ng_sequence.append(j)
                    last_result = img_results[i-1]

                obj_result["judge_detail"].append({
                    "type": "Pattern_NG",
                    "cnt": len(ng_sequence),
                    "detail": ",".join(map(str, ng_sequence))
                })

                for ind in ng_sequence:
                    filename = f"/IMG_{ind:03d}.jpeg"
                    self.save_ng_roi(image_path+filename,obj_result['roi_pox'],obj_result['obj_no'],'Pattern_NG',ind,image_path+'/result')

            # Black_NG判断
            black_sequences = []
            current_sequence = []
            for j, img_results in enumerate(all_results.values()):
                if img_results[i-1] == "Black":
                    current_sequence.append(j+1)
                else:
                    if len(current_sequence) >= black_continue:
                        black_sequences.extend(current_sequence)
                    current_sequence = []
            
            if len(current_sequence) >= black_continue:
                black_sequences.extend(current_sequence)

            if black_sequences:
                black_sequences = [x - 1 for x in black_sequences]
                obj_result["judge_detail"].append({
                    "type": "Black_NG",
                    "cnt": len(black_sequences),
                    "detail": ",".join(map(str, black_sequences))
                })

                for ind in black_sequences:
                    filename = f"/IMG_{ind:03d}.jpeg"
                    self.save_ng_roi(image_path+filename,obj_result['roi_pox'],obj_result['obj_no'],'Black_NG',ind,image_path+'/result')

            result_json.append(obj_result)
        
        with open(os.path.join(result_folder, 'result.json'), 'w') as f:
            json.dump(result_json, f, indent=4)

        return result_json

    def save_ng_roi(self,image_path, roi_points, obj_no, ng_name, image_no, result_folder):
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
        filename = f"obj_{obj_no}_{ng_name}_{image_no:03d}.jpeg"
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
        
        # 創建並保存結果 JSON
        result_json = self.create_result_json(config, final_results, all_results, result_folder)

        #寫入最後結果圖
        self.create_result_image(config, final_results, all_results, image_size, result_folder)

        print(f"最終結果: {final_results}")


if __name__ == '__main__':
    
    image_path = sys.argv[1]
    config_file = sys.argv[2]
    black_continue = 15
    obj = Aging_detect(image_path, config_file)
    obj.main()
'''

if __name__ == "__main__":
    
    image_path = './dataset/Exposition/1022/IMG_035_A'
    config_file = '20241018_S17_ExpositionA.json'
    black_continue = 15
    obj = Aging_detect(image_path, config_file)
    obj.main()
'''
