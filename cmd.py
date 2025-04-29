import os
import shutil
from pathlib import Path

# 設定來源和目標資料夾
source_folder = r"C:\Users\User\Desktop\Test"
target_folder = r"C:\Users\User\Desktop\Test_Output"  # 您可以根據需要修改目標資料夾路徑

# 確保目標資料夾存在
os.makedirs(target_folder, exist_ok=True)

# 設定參數
img_start = 1
img_end = 5
copies_per_img = 8  # 每張原始影像複製8張
total_copies = 5    # 整體複製5次

print(f"開始處理 {img_end} 張原始影像，每張複製 {copies_per_img} 次，然後整體複製 {total_copies} 次...")

# 階段1：生成初始的40張影像（每張原圖複製8張）
initial_images = []  # 用於儲存初始生成的40張影像路徑

# 處理每一張原始圖片
for i in range(img_start, img_end + 1):
    # 計算原始圖片名稱 (IMG_001 到 IMG_005)
    orig_num = f"{i:03d}"
    orig_img = f"IMG_{orig_num}.jpeg"
    orig_path = os.path.join(source_folder, orig_img)
    
    # 檢查原始檔案是否存在
    if not os.path.exists(orig_path):
        print(f"警告：找不到原始檔案 {orig_path}")
        continue
    
    # 計算目標圖片的開始編號
    start_idx = (i - 1) * copies_per_img
    
    # 複製並重命名每一份複本
    for j in range(copies_per_img):
        new_idx = start_idx + j
        new_num = f"{new_idx:03d}"
        new_img = f"IMG_{new_num}.jpeg"
        temp_path = os.path.join(target_folder, new_img)
        
        print(f"複製 {orig_img} 到 {new_img}")
        shutil.copy2(orig_path, temp_path)
        initial_images.append((temp_path, new_img))

print(f"\n階段1完成：生成初始40張影像")

# 階段2：將初始的40張影像複製5次（但第一次已在階段1完成，所以這裡再複製4次）
for copy_idx in range(1, total_copies):
    base_idx = copy_idx * (img_end - img_start + 1) * copies_per_img
    
    for idx, (src_path, src_name) in enumerate(initial_images):
        new_idx = base_idx + idx
        # 確保不超過199
        if new_idx <= 199:
            new_num = f"{new_idx:03d}"
            new_img = f"IMG_{new_num}.jpeg"
            new_path = os.path.join(target_folder, new_img)
            
            print(f"複製集合 {copy_idx+1}：將 {src_name} 複製為 {new_img}")
            shutil.copy2(src_path, new_path)

# 計算實際生成的影像數量
actual_images = 0
for file in os.listdir(target_folder):
    if file.startswith("IMG_") and file.endswith(".jpeg"):
        actual_images += 1

print(f"\n完成！總共生成了 {actual_images} 張影像。")
print(f"已保存在 {target_folder} 資料夾中。")