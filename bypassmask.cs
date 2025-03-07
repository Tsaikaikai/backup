using OpenCvSharp;
using System.Text.Json;
using System.IO;

namespace ImageMasking
{
    public class ImageProcessor
    {
        // 定義 JSON 資料結構
        public class Shape
        {
            public string Label { get; set; }
            public double[][] Points { get; set; }
            public string ShapeType { get; set; }
        }

        public class JsonData
        {
            public string Version { get; set; }
            public Shape[] Shapes { get; set; }
            public string ImagePath { get; set; }
            public int ImageHeight { get; set; }
            public int ImageWidth { get; set; }
        }

        /// <summary>
        /// 將指定的 bypass 區域塗黑，若無對應JSON則直接輸出原圖
        /// </summary>
        /// <param name="srcImage">輸入影像</param>
        /// <param name="srcImagePath">輸入影像的完整路徑</param>
        /// <param name="bypassJsonFolder">存放bypass JSON檔案的資料夾</param>
        /// <param name="outputPath">輸出影像保存路徑</param>
        /// <returns>處理是否成功</returns>
        public static bool MaskBypassRegion(Mat srcImage, string srcImagePath, string bypassJsonFolder, string outputPath)
        {
            try
            {
                // 驗證輸入
                if (srcImage == null || srcImage.Empty())
                {
                    throw new ArgumentException("輸入影像無效");
                }
                if (string.IsNullOrEmpty(srcImagePath) || !File.Exists(srcImagePath))
                {
                    throw new ArgumentException("輸入影像路徑無效");
                }
                if (string.IsNullOrEmpty(bypassJsonFolder) || !Directory.Exists(bypassJsonFolder))
                {
                    throw new ArgumentException("bypass JSON資料夾無效");
                }

                // 從 srcImagePath 獲取檔案名稱（不含副檔名）
                string fileNameWithoutExt = Path.GetFileNameWithoutExtension(srcImagePath);
                // 構建 JSON 檔案路徑
                string jsonPath = Path.Combine(bypassJsonFolder, $"{fileNameWithoutExt}.json");

                // 如果 JSON 檔案不存在，直接輸出原圖
                if (!File.Exists(jsonPath))
                {
                    Console.WriteLine($"未找到對應JSON檔案: {jsonPath}，直接輸出原始影像");
                    return Cv2.ImWrite(outputPath, srcImage);
                }

                // 讀取並解析 JSON
                string jsonString = File.ReadAllText(jsonPath);
                JsonData data = JsonSerializer.Deserialize<JsonData>(jsonString);

                // 驗證影像尺寸
                if (srcImage.Height != data.ImageHeight || srcImage.Width != data.ImageWidth)
                {
                    throw new ArgumentException("影像尺寸與JSON描述不符");
                }

                // 創建遮罩
                using Mat mask = new Mat(srcImage.Size(), MatType.CV_8UC1, Scalar.All(0));

                // 處理所有 bypass 區域
                if (data.Shapes != null && data.Shapes.Length > 0)
                {
                    foreach (var shape in data.Shapes)
                    {
                        if (shape.Label == "Bypass" && shape.Points != null)
                        {
                            Point[] points = shape.Points
                                .Select(p => new Point((int)p[0], (int)p[1]))
                                .ToArray();
                            Cv2.FillPoly(mask, new Point[][] { points }, Scalar.All(255));
                        }
                    }
                }

                // 應用遮罩到原圖
                using Mat result = srcImage.Clone();
                result.SetTo(Scalar.All(0), mask);

                // 保存結果
                return Cv2.ImWrite(outputPath, result);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"錯誤: {ex.Message}");
                return false;
            }
        }
    }
}
