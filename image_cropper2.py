import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any
from shapely.geometry import Polygon, box

class ImageCropperWithAnnotations:
    def __init__(self, ng_folder: str, label_folder: str, output_folder: str, 
                 crop_size: int = 644, overlap_ratio: float = 0.1, visualize: bool = False):
        """
        初始化影像切割器
        
        Args:
            ng_folder: NG影像資料夾路徑
            label_folder: 標註json檔案資料夾路徑
            output_folder: 輸出資料夾路徑
            crop_size: 切割影像大小
            overlap_ratio: 重疊比例 (0.1 = 10%)
            visualize: 是否產生可視化圖片
        """
        self.ng_folder = Path(ng_folder)
        self.label_folder = Path(label_folder)
        self.output_folder = Path(output_folder)
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.visualize = visualize
        
        # 創建輸出資料夾
        self.output_ng_folder = self.output_folder / "NG"
        self.output_label_folder = self.output_folder / "label"
        self.output_ng_folder.mkdir(parents=True, exist_ok=True)
        self.output_label_folder.mkdir(parents=True, exist_ok=True)
        
        # 如果需要可視化，創建可視化資料夾
        if self.visualize:
            self.output_vis_folder = self.output_folder / "visualization"
            self.output_vis_original_folder = self.output_vis_folder / "original"
            self.output_vis_cropped_folder = self.output_vis_folder / "cropped"
            self.output_vis_original_folder.mkdir(parents=True, exist_ok=True)
            self.output_vis_cropped_folder.mkdir(parents=True, exist_ok=True)
    
    def load_annotation(self, json_path: str) -> Dict[str, Any]:
        """載入標註檔案"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """產生不同顏色用於標註可視化"""
        colors = []
        for i in range(num_colors):
            # 使用HSV色彩空間產生均勻分布的顏色
            hue = int(180 * i / num_colors)
            color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))
        return colors
    
    def draw_annotations(self, image: np.ndarray, annotation: Dict[str, Any], 
                        crop_info: Dict = None) -> np.ndarray:
        """
        在影像上繪製標註
        
        Args:
            image: 輸入影像
            annotation: 標註資料
            crop_info: 切割資訊 {'x': crop_x, 'y': crop_y} (可選，用於原圖標註)
        
        Returns:
            繪製標註後的影像
        """
        vis_image = image.copy()
        shapes = annotation.get("shapes", [])
        
        if not shapes:
            return vis_image
        
        # 為不同標籤產生不同顏色
        labels = list(set(shape["label"] for shape in shapes))
        colors = self.generate_colors(len(labels))
        label_colors = dict(zip(labels, colors))
        
        for shape in shapes:
            if shape["shape_type"] == "polygon":
                points = np.array(shape["points"], dtype=np.int32)
                label = shape["label"]
                color = label_colors[label]
                
                # 繪製多邊形
                cv2.fillPoly(vis_image, [points], color, cv2.LINE_AA)
                cv2.polylines(vis_image, [points], True, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 添加標籤文字
                if len(points) > 0:
                    text_pos = tuple(map(int, points[0]))
                    cv2.putText(vis_image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                              0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 如果是原圖，繪製切割框
        if crop_info is not None:
            crop_x, crop_y = crop_info['x'], crop_info['y']
            cv2.rectangle(vis_image, 
                         (crop_x, crop_y), 
                         (crop_x + self.crop_size, crop_y + self.crop_size),
                         (0, 255, 0), 3)  # 綠色切割框
            
            # 添加切割區域標籤
            cv2.putText(vis_image, f"Crop ({crop_x},{crop_y})", 
                       (crop_x, crop_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        return vis_image
    
    def visualize_original_with_crops(self, image: np.ndarray, annotation: Dict[str, Any], 
                                    crop_positions: List[Tuple[int, int]], 
                                    output_path: str):
        """可視化原始影像和所有切割位置"""
        vis_image = image.copy()
        
        # 繪製原始標註
        vis_image = self.draw_annotations(vis_image, annotation)
        
        # 繪製所有切割框
        for i, (crop_x, crop_y) in enumerate(crop_positions):
            # 不同的切割框用不同顏色
            color = ((i * 50) % 255, (i * 80) % 255, (i * 120) % 255)
            cv2.rectangle(vis_image, 
                         (crop_x, crop_y), 
                         (crop_x + self.crop_size, crop_y + self.crop_size),
                         color, 2)
            
            # 添加切割區域編號
            cv2.putText(vis_image, f"{i}", 
                       (crop_x + 5, crop_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2, cv2.LINE_AA)
        
        # 添加圖例
        legend_height = 100
        legend = np.zeros((legend_height, vis_image.shape[1], 3), dtype=np.uint8)
        cv2.putText(legend, f"Original Image - Total Crops: {len(crop_positions)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(legend, f"Crop Size: {self.crop_size}x{self.crop_size}, Overlap: {self.overlap_ratio*100:.1f}%", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 合併圖例和影像
        final_image = np.vstack([legend, vis_image])
        
        cv2.imwrite(output_path, final_image)
        return final_image
    
    def save_annotation(self, annotation: Dict[str, Any], output_path: str):
        """儲存標註檔案"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
    
    def point_in_crop(self, point: List[float], crop_x: int, crop_y: int) -> bool:
        """檢查點是否在切割範圍內"""
        x, y = point
        return (crop_x <= x < crop_x + self.crop_size and 
                crop_y <= y < crop_y + self.crop_size)
    
    def clip_polygon_to_crop(self, points: List[List[float]], 
                           crop_x: int, crop_y: int) -> List[List[List[float]]]:
        """
        使用shapely裁剪多邊形到切割區域，支持凹多邊形
        
        Args:
            points: 原始多邊形頂點座標
            crop_x: 切割區域左上角x座標
            crop_y: 切割區域左上角y座標
            
        Returns:
            裁剪後的多邊形頂點列表(相對座標)的列表
        """
        # 定義裁剪窗口(矩形)
        crop_rect = box(
            crop_x, 
            crop_y, 
            crop_x + self.crop_size, 
            crop_y + self.crop_size
        )
        
        # 創建原始多邊形
        original_polygon = Polygon(points)
        
        # 檢查多邊形是否有效
        if not original_polygon.is_valid:
            # 嘗試修復無效多邊形
            original_polygon = original_polygon.buffer(0)
            if not original_polygon.is_valid:
                return []
        
        # 執行裁剪
        clipped_polygon = original_polygon.intersection(crop_rect)
        
        # 處理裁剪結果
        result_polygons = []
        
        if clipped_polygon.is_empty:
            return []
        
        # 處理多種幾何類型
        if clipped_polygon.geom_type == 'Polygon':
            # 單個多邊形
            coords = list(clipped_polygon.exterior.coords)
            if len(coords) >= 3:
                # 轉換為相對座標
                relative_coords = [[x - crop_x, y - crop_y] for x, y in coords]
                result_polygons.append(relative_coords)
        elif clipped_polygon.geom_type == 'MultiPolygon':
            # 多個多邊形
            for polygon in clipped_polygon.geoms:
                coords = list(polygon.exterior.coords)
                if len(coords) >= 3:
                    # 轉換為相對座標
                    relative_coords = [[x - crop_x, y - crop_y] for x, y in coords]
                    result_polygons.append(relative_coords)
        
        return result_polygons
    
    def polygon_intersects_crop(self, points: List[List[float]], 
                              crop_x: int, crop_y: int) -> bool:
        """檢查polygon是否與切割區域相交"""
        # 使用shapely進行精確相交檢測
        crop_rect = box(
            crop_x, 
            crop_y, 
            crop_x + self.crop_size, 
            crop_y + self.crop_size
        )
        
        original_polygon = Polygon(points)
        if not original_polygon.is_valid:
            original_polygon = original_polygon.buffer(0)
            if not original_polygon.is_valid:
                return False
        
        return original_polygon.intersects(crop_rect)
    
    def get_crop_positions(self, img_width: int, img_height: int) -> List[Tuple[int, int]]:
        """
        計算所有切割位置
        考慮重疊和邊界調整
        """
        positions = []
        step_size = int(self.crop_size * (1 - self.overlap_ratio))
        
        for y in range(0, img_height, step_size):
            for x in range(0, img_width, step_size):
                # 調整位置以避免超出邊界
                adjusted_x = min(x, img_width - self.crop_size)
                adjusted_y = min(y, img_height - self.crop_size)
                
                # 確保不會有負數座標
                adjusted_x = max(0, adjusted_x)
                adjusted_y = max(0, adjusted_y)
                
                positions.append((adjusted_x, adjusted_y))
        
        # 移除重複位置
        positions = list(set(positions))
        return positions
    
    def process_single_image(self, image_path: str, annotation_path: str):
        """處理單張影像和對應的標註檔案"""
        print(f"處理影像: {image_path}")
        
        # 載入影像
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法載入影像: {image_path}")
            return
        
        img_height, img_width = image.shape[:2]
        
        # 載入標註
        annotation = self.load_annotation(annotation_path)
        
        # 獲取檔案名稱（不含副檔名）
        base_name = Path(image_path).stem
        
        # 獲取切割位置
        crop_positions = self.get_crop_positions(img_width, img_height)
        
        # 如果啟用可視化，產生原始影像的可視化圖片
        if self.visualize:
            original_vis_path = str(self.output_vis_original_folder / f"{base_name}_original.jpg")
            self.visualize_original_with_crops(image, annotation, crop_positions, original_vis_path)
        
        crop_count = 0
        for crop_x, crop_y in crop_positions:
            # 切割影像
            cropped_image = image[crop_y:crop_y + self.crop_size, 
                                crop_x:crop_x + self.crop_size]
            
            # 創建新的標註檔案
            new_annotation = {
                "version": annotation.get("version", "4.5.6"),
                "flags": annotation.get("flags", {}),
                "shapes": [],
                "imagePath": f"{base_name}_{crop_count:04d}.jpg",
                "imageHeight": self.crop_size,
                "imageWidth": self.crop_size
            }
            
            # 處理每個shape
            for shape in annotation.get("shapes", []):
                if shape["shape_type"] == "polygon":
                    points = shape["points"]
                    
                    # 檢查polygon是否與切割區域相交
                    if self.polygon_intersects_crop(points, crop_x, crop_y):
                        # 使用shapely裁剪多邊形
                        clipped_polygons = self.clip_polygon_to_crop(points, crop_x, crop_y)
                        
                        # 處理所有裁剪後的多邊形
                        for polygon_points in clipped_polygons:
                            if len(polygon_points) >= 3:
                                new_shape = {
                                    "label": shape["label"],
                                    "points": polygon_points,
                                    "group_id": shape.get("group_id"),
                                    "shape_type": "polygon",
                                    "flags": shape.get("flags", {})
                                }
                                new_annotation["shapes"].append(new_shape)
            
            # 儲存切割後的影像和標註（即使沒有標註也要儲存）
            output_image_name = f"{base_name}_{crop_count:04d}.jpg"
            output_json_name = f"{base_name}_{crop_count:04d}.json"
            
            # 儲存影像
            cv2.imwrite(str(self.output_ng_folder / output_image_name), cropped_image)
            
            # 儲存標註
            self.save_annotation(new_annotation, 
                               str(self.output_label_folder / output_json_name))
            
            # 如果啟用可視化，產生切割後影像的可視化圖片
            if self.visualize:
                vis_cropped_image = self.draw_annotations(cropped_image, new_annotation)
                
                # 添加資訊標籤
                info_height = 60
                info_panel = np.zeros((info_height, vis_cropped_image.shape[1], 3), dtype=np.uint8)
                cv2.putText(info_panel, f"Crop {crop_count:04d} - Position: ({crop_x}, {crop_y})", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Annotations: {len(new_annotation['shapes'])}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 合併資訊面板和切割影像
                vis_final = np.vstack([info_panel, vis_cropped_image])
                
                vis_output_name = f"{base_name}_{crop_count:04d}_vis.jpg"
                cv2.imwrite(str(self.output_vis_cropped_folder / vis_output_name), vis_final)
            
            crop_count += 1
        
        print(f"完成切割，共產生 {crop_count} 個小圖")
        if self.visualize:
            print(f"可視化圖片已儲存至: {self.output_vis_folder}")
    
    def process_all_images(self):
        """處理所有影像"""
        # 獲取所有影像檔案
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.ng_folder.glob(f"*{ext}"))
            image_files.extend(self.ng_folder.glob(f"*{ext.upper()}"))
        
        print(f"找到 {len(image_files)} 個影像檔案")
        
        processed_count = 0
        for image_path in image_files:
            # 尋找對應的標註檔案
            json_name = image_path.stem + ".json"
            json_path = self.label_folder / json_name
            
            if json_path.exists():
                try:
                    self.process_single_image(str(image_path), str(json_path))
                    processed_count += 1
                except Exception as e:
                    print(f"處理 {image_path} 時發生錯誤: {e}")
            else:
                print(f"找不到對應的標註檔案: {json_path}")
        
        print(f"處理完成！共處理了 {processed_count} 個影像檔案")

def main():
    parser = argparse.ArgumentParser(description="影像切割與標註轉換工具")
    parser.add_argument("--ng_folder", required=True, help="NG影像資料夾路徑")
    parser.add_argument("--label_folder", required=True, help="標註json檔案資料夾路徑")
    parser.add_argument("--output_folder", required=True, help="輸出資料夾路徑")
    parser.add_argument("--crop_size", type=int, default=644, help="切割影像大小 (預設: 644)")
    parser.add_argument("--overlap_ratio", type=float, default=0.1, 
                       help="重疊比例 (預設: 0.1, 即10%%)")
    parser.add_argument("--visualize", action="store_true", 
                       help="啟用可視化功能，產生標註可視化圖片")
    
    args = parser.parse_args()
    
    # 創建處理器並執行
    cropper = ImageCropperWithAnnotations(
        ng_folder=args.ng_folder,
        label_folder=args.label_folder,
        output_folder=args.output_folder,
        crop_size=args.crop_size,
        overlap_ratio=args.overlap_ratio,
        visualize=args.visualize
    )
    
    cropper.process_all_images()

if __name__ == "__main__":
    main()