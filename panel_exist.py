import cv2
import numpy as np
import json

# 假設影像路徑和 JSON 數據
image_path = "your_image_path.jpg"  # 請替換為實際影像路徑
json_data = {
    "obj_gray_spec": [12, 255],
    "obj_occur_ratio": 0.3,
    "detect_roi": [[15, 1701], [1062, 1779], [1056, 60], [9, 174]],
    "obj_list": [
        {"obj_no": "1", "roi_pox": [[45, 411], [1044, 354], [1053, 120], [33, 168]], "area_spec": [60000, 200000], "width_spec": [850, 1080], "height_spec": [100, 300]},
        {"obj_no": "2", "roi_pox": [[33, 624], [1011, 585], [1011, 414], [39, 465]], "area_spec": [60000, 200000], "width_spec": [850, 1080], "height_spec": [100, 300]},
        {"obj_no": "3", "roi_pox": [[39, 885], [1020, 879], [1023, 663], [42, 690]], "area_spec": [60000, 200000], "width_spec": [750, 1080], "height_spec": [100, 300]},
        {"obj_no": "4", "roi_pox": [[30, 1113], [1017, 1110], [1029, 912], [33, 924]], "area_spec": [60000, 200000], "width_spec": [800, 1080], "height_spec": [80, 300]},
        {"obj_no": "5", "roi_pox": [[48, 1332], [999, 1362], [1017, 1158], [57, 1125]], "area_spec": [60000, 200000], "width_spec": [700, 1080], "height_spec": [100, 300]},
        {"obj_no": "6", "roi_pox": [[69, 1530], [951, 1569], [999, 1380], [81, 1344]], "area_spec": [60000, 200000], "width_spec": [700, 1080], "height_spec": [100, 300]},
        {"obj_no": "7", "roi_pox": [[99, 1710], [942, 1773], [963, 1587], [117, 1542]], "area_spec": [80000, 200000], "width_spec": [700, 1080], "height_spec": [100, 300]}
    ]
}

# 假設的 panel_exist 列表
panel_exist = [1, 1, 1, 1, 1, 1, 0]

# 讀取影像
image = cv2.imread(image_path)
if image is None:
    raise ValueError("無法讀取影像，請檢查路徑是否正確")

# 轉為灰度圖
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 從 JSON 中提取灰度範圍
gray_min, gray_max = json_data["obj_gray_spec"]

# 定義檢測亮度的閾值（這裡假設平均灰度值大於某值為亮）
brightness_threshold = 100  # 可根據實際情況調整

# 檢查每個面板的亮度
bright_panels = 0
existing_panels = sum(panel_exist)  # 計算存在的面板數量

for i, obj in enumerate(json_data["obj_list"]):
    if panel_exist[i] == 0:  # 如果面板不存在，跳過
        continue
    
    # 提取 ROI 座標
    roi_points = np.array(obj["roi_pox"], dtype=np.int32)
    
    # 創建掩膜
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    
    # 提取 ROI 區域
    roi_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    
    # 計算 ROI 區域的平均灰度值
    mean_gray = cv2.mean(roi_gray, mask=mask)[0]
    
    # 判斷是否為亮的面板
    if mean_gray > brightness_threshold:
        bright_panels += 1
        print(f"面板 {obj['obj_no']} 是亮的，平均灰度值: {mean_gray:.2f}")
    else:
        print(f"面板 {obj['obj_no']} 未亮，平均灰度值: {mean_gray:.2f}")

# 判斷是否有点灯
if existing_panels > 0:
    if bright_panels > existing_panels / 2:
        print(f"亮的面板數量: {bright_panels}，存在的面板數量: {existing_panels}，判斷為：有点灯")
    else:
        print(f"亮的面板數量: {bright_panels}，存在的面板數量: {existing_panels}，判斷為：無点灯")
else:
    print("無存在的面板，無法判斷是否有点灯")
