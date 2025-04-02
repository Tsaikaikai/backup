using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using System.IO;

namespace newSGM
{
    public class SGMAlgorithm
    {
        private static string GetTimestamp()
        {
            // 獲取當前時間戳記，精確到毫秒
            return DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
        }

        private static void Reconstruct(Mat binImage1, Mat binImage2, ref Mat result)
        {
            try
            {
                // 找到第一張影像中的輪廓
                Point[][] contours;
                HierarchyIndex[] hierarchy;
                Cv2.FindContours(binImage1, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                foreach (var contour in contours)
                {
                    // 創建與原影像大小相同的遮罩
                    Mat mask = Mat.Zeros(binImage1.Size(), MatType.CV_8UC1);
                    Cv2.DrawContours(mask, new[] { contour }, -1, Scalar.All(255), thickness: -1);

                    // 檢查該區域是否與第二張影像有重疊
                    Mat overlap = mask & binImage2;
                    if (Cv2.CountNonZero(overlap) > 0)
                    {
                        // 保留整個封閉區域
                        result = result | mask;
                    }
                    mask.Dispose();
                    overlap.Dispose();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Fail: " + ex.Message);
            }
            finally
            {

            }
        }

        public static int ProcessAndCropImage(string inputPath, string outputDir, int threshold, int windowSize, double overlap, int downScale)
        {
            Mat oriImg = new Mat();
            Mat img = new Mat();
            Mat oriGray = new Mat();
            Mat gray = new Mat();
            Mat binary = new Mat();
            Mat kernel = new Mat();
            Mat dilated = new Mat();
            Mat remask = new Mat();
            Mat labels = new Mat(), stats = new Mat(), centroids = new Mat();
            Mat edges = new Mat();
            try
            {
                // 讀取圖片
                oriImg = Cv2.ImRead(inputPath);
                if (oriImg.Empty())
                    throw new Exception($"無法讀取圖片: {inputPath}");

                // 縮小圖片
                Size newSize = new Size(oriImg.Width / downScale, oriImg.Height / downScale);
                img = oriImg.Resize(newSize, 0, 0, InterpolationFlags.Nearest);

                // 灰度化
                oriGray = oriImg.CvtColor(ColorConversionCodes.BGR2GRAY);
                gray = img.CvtColor(ColorConversionCodes.BGR2GRAY);

                // 二值化
                Cv2.Threshold(gray, binary, threshold, 255, ThresholdTypes.Binary);
                //Cv2.ImWrite(Path.ChangeExtension(inputPath, "_binary.jpg"), binary);

                // 形態學操作
                kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
                dilated = binary.MorphologyEx(MorphTypes.Open, kernel, iterations: 2);
                dilated = dilated.MorphologyEx(MorphTypes.Close, kernel, iterations: 2);
                //Cv2.ImWrite(Path.ChangeExtension(inputPath, "_dilated.jpg"), dilated);

                // 復原遮罩處理
                remask = Cv2.ImRead("D:\\newSGM\\reconstruct_mask.tif", ImreadModes.Grayscale);
                remask = remask.Resize(newSize, 0, 0, InterpolationFlags.Nearest);

                Mat constructResult = Mat.Zeros(dilated.Size(), MatType.CV_8UC1);
                Reconstruct(dilated, remask, ref constructResult);
                //Cv2.ImWrite(Path.ChangeExtension(inputPath, "_construct_result.jpg"), constructResult);

                // 找到連通區域
                int numLabels = Cv2.ConnectedComponentsWithStats(constructResult, labels, stats, centroids);
                constructResult.Dispose();

                // 找到最大的區域（排除背景）
                int maxLabel = 1, maxArea = 0;
                for (int i = 1; i < numLabels; i++)
                {
                    int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                    if (area > maxArea)
                    {
                        maxArea = area;
                        maxLabel = i;
                    }
                }

                // 創建只包含最大區域的遮罩
                Mat largestComponent = new Mat(labels.Size(), MatType.CV_8UC1);

                // 將矩陣中等於 maxLabel 的部分設置為白色 (255)，其餘部分為黑色 (0)
                Cv2.Compare(labels, maxLabel, largestComponent, CmpType.EQ);
                largestComponent.ConvertTo(largestComponent, MatType.CV_8UC1, 255);

                largestComponent = largestComponent.MorphologyEx(MorphTypes.Open, kernel, iterations: 2);
                largestComponent = largestComponent.MorphologyEx(MorphTypes.Close, kernel, iterations: 2);

                largestComponent = largestComponent.Resize(new Size(oriImg.Width, oriImg.Height), 0, 0, InterpolationFlags.Nearest);
                //Cv2.ImWrite(Path.ChangeExtension(inputPath, "_mask_smoo.jpg"), largestComponent);

                // 邊緣檢測
                edges = largestComponent.Canny(100, 200);
                largestComponent.Dispose();

                // 計算滑動窗口的步長（基於重疊率）
                int stride = (int)(windowSize * (1 - overlap));

                // 獲取圖像尺寸
                int height = edges.Rows;
                int width = edges.Cols;

                int count = 0;

                for (int y = 0; y <= height - windowSize; y += stride)
                {
                    for (int x = 0; x <= width - windowSize; x += stride)
                    {
                        // 提取當前窗口
                        Rect windowRect = new Rect(x, y, windowSize, windowSize);
                        Mat window = new Mat(edges, windowRect);
                        Mat nonZeroCoords = window.FindNonZero();
                        window.Dispose();

                        if (nonZeroCoords != null && nonZeroCoords.Rows > 0)
                        {
                            int sumX = 0, sumY = 0;

                            for (int i = 0; i < nonZeroCoords.Rows; i++)
                            {
                                var point = nonZeroCoords.At<Point>(i);
                                sumX += point.X;
                                sumY += point.Y;
                            }

                            // Calculate the center of the edge pixels
                            int centerX = sumX / nonZeroCoords.Rows;
                            int centerY = sumY / nonZeroCoords.Rows;

                            // Adjust the starting coordinates for cropping
                            int startY = Math.Max(0, Math.Min(y + centerY - windowSize / 2, height - windowSize));
                            int startX = Math.Max(0, Math.Min(x + centerX - windowSize / 2, width - windowSize));

                            // Crop the original grayscale image
                            Rect cropRect = new Rect(startX, startY, windowSize, windowSize);
                            Mat crop = new Mat();
                            crop = new Mat(oriGray, cropRect);

                            // Save the cropped image with a unique timestamp
                            string timestamp = GetTimestamp();
                            string outputPath = Path.Combine(outputDir, $"{timestamp}.png");
                            ImageEncodingParam[] Params = new ImageEncodingParam[]
                            {
                                new ImageEncodingParam(ImwriteFlags.PngCompression, 0)
                            };

                            if (!crop.IsContinuous())
                            {
                                Mat cropTemp = new Mat();
                                cropTemp = crop.Clone();
                            }

                            Cv2.ImWrite(outputPath, crop, Params);
                            crop.Dispose();
                            count++;
                        }
                        nonZeroCoords.Dispose();
                    }
                }
                return count;
            }
            catch (Exception ex)
            {
                throw new Exception("Fail: " + ex.Message);
            }
            finally
            {
                oriImg.Dispose();
                img.Dispose();
                oriGray.Dispose();
                gray.Dispose();
                binary.Dispose();
                kernel.Dispose();
                dilated.Dispose();
                remask.Dispose();
                labels.Dispose();
                stats.Dispose();
                centroids.Dispose();
                edges.Dispose();
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }
    }
}
