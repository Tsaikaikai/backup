import cv2
import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# 設定圖片資料夾路徑與輸出影片名稱
image_folder = r'D:\Datasets\AIF\250803_fly\3'
output_video = 'output.mp4'
fps = 30  # 每秒幀數

# 取得所有圖片檔案並排序
images = sorted(
    [img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png','bmp'))],
    key=natural_sort_key
)

# 讀取第一張圖片以取得原始尺寸
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
original_height, original_width = first_frame.shape[:2]

# 計算縮小後的尺寸
new_width = original_width
new_height = original_height

# 建立影片寫入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

# 將每張縮小後的圖片寫入影片
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    #resized_frame = cv2.resize(frame, (new_width, new_height))
    video.write(frame) 

video.release()
print("影片輸出完成！")
