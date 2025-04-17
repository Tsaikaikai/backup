using OpenCvSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

public static List<Mat> ProcessAndCropImage(string inputPath, string outputDir, int threshold, int windowSize, int stepDistance, int downScale, int maxThreads = 0)
{
    // 如果maxThreads為0，使用環境處理器邏輯核心數
    if (maxThreads <= 0)
        maxThreads = Environment.ProcessorCount;

    Mat oriImg = null;
    Mat img = null;
    Mat oriGray = null;
    Mat gray = null;
    Mat binary = null;
    Mat kernel = null;
    Mat dilated = null;
    Mat remask = null;
    Mat labels = null;
    Mat stats = null;
    Mat centroids = null;
    Mat edges = null;
    
    // 使用ConcurrentBag來安全地收集多執行緒處理的結果
    ConcurrentBag<Mat> cropBag = new ConcurrentBag<Mat>();
    
    try
    {
        // 讀取圖片
        oriImg = Cv2.ImRead(inputPath);
        if (oriImg.Empty())
            throw new Exception($"無法讀取圖片: {inputPath}");

        // 縮小圖片
        Size newSize = new Size(oriImg.Width / downScale, oriImg.Height / downScale);
        img = oriImg.Resize(newSize, 0, 0, InterpolationFlags.Nearest);

        // 灰度化 - 直接處理
        oriGray = oriImg.CvtColor(ColorConversionCodes.BGR2GRAY);
        gray = img.CvtColor(ColorConversionCodes.BGR2GRAY);

        // 二值化 - 使用更高效的二值化方法
        binary = new Mat();
        Cv2.Threshold(gray, binary, threshold, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);

        // 形態學操作 - 改用更高效的結構元素和減少迭代次數
        kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
        dilated = binary.MorphologyEx(MorphTypes.Open, kernel);
        dilated = dilated.MorphologyEx(MorphTypes.Close, kernel);

        // 復原遮罩處理
        remask = Cv2.ImRead("D:\\newSGM\\reconstruct_mask.tif", ImreadModes.Grayscale);
        remask = remask.Resize(newSize, 0, 0, InterpolationFlags.Nearest);

        Mat constructResult = new Mat();
        Reconstruct(dilated, remask, ref constructResult);
        
        // 找到連通區域
        labels = new Mat();
        stats = new Mat();
        centroids = new Mat();
        int numLabels = Cv2.ConnectedComponentsWithStats(constructResult, labels, stats, centroids);
        constructResult.Dispose();

        // 找到最大的區域（排除背景）- 使用LINQ優化
        int maxLabel = Enumerable.Range(1, numLabels - 1)
            .Select(i => new { Label = i, Area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area) })
            .OrderByDescending(x => x.Area)
            .FirstOrDefault()?.Label ?? 1;

        // 創建只包含最大區域的遮罩
        Mat largestComponent = new Mat(labels.Size(), MatType.CV_8UC1);
        Cv2.Compare(labels, maxLabel, largestComponent, CmpType.EQ);
        largestComponent.ConvertTo(largestComponent, MatType.CV_8UC1, 255);

        // 優化形態學操作
        largestComponent = largestComponent.MorphologyEx(MorphTypes.Open, kernel);
        largestComponent = largestComponent.MorphologyEx(MorphTypes.Close, kernel);
        largestComponent = largestComponent.Resize(new Size(oriImg.Width, oriImg.Height), 0, 0, InterpolationFlags.Nearest);

        // 邊緣檢測
        edges = new Mat();
        Cv2.Canny(largestComponent, edges, 100, 200);
        largestComponent.Dispose();

        // 提取所有邊緣點
        Point[] edgePoints;
        using (var nonZeroCoords = edges.FindNonZero())
        {
            if (nonZeroCoords == null || nonZeroCoords.Rows == 0)
                return new List<Mat>();

            // 直接提取所有點，而不是一個個複製
            edgePoints = new Point[nonZeroCoords.Rows];
            for (int i = 0; i < nonZeroCoords.Rows; i++)
                edgePoints[i] = nonZeroCoords.At<Point>(i);
        }

        // 使用更高效的排序和取樣
        Array.Sort(edgePoints, (p1, p2) => {
            int xCompare = p1.X.CompareTo(p2.X);
            return xCompare != 0 ? xCompare : p1.Y.CompareTo(p2.Y);
        });

        // 進行高效率的點採樣
        List<Point> sampledPoints = new List<Point>(edgePoints.Length / stepDistance + 1);
        if (edgePoints.Length > 0)
        {
            sampledPoints.Add(edgePoints[0]);
            Point lastPoint = edgePoints[0];

            // 使用更高效的距離計算
            for (int i = 1; i < edgePoints.Length; i++)
            {
                Point currentPoint = edgePoints[i];
                int dx = currentPoint.X - lastPoint.X;
                int dy = currentPoint.Y - lastPoint.Y;
                int distanceSquared = dx * dx + dy * dy; // 避免開平方運算
                
                if (distanceSquared >= stepDistance * stepDistance)
                {
                    sampledPoints.Add(currentPoint);
                    lastPoint = currentPoint;
                }
            }
        }

        // 使用ConcurrentBag來存儲結果位置
        ConcurrentBag<(int x, int y)> cropPositions = new ConcurrentBag<(int x, int y)>();
        int height = oriImg.Rows;
        int width = oriImg.Cols;

        // 並行處理採樣點
        Parallel.ForEach(
            sampledPoints,
            new ParallelOptions { MaxDegreeOfParallelism = maxThreads },
            point => {
                int startY = Math.Max(0, Math.Min(point.Y - windowSize / 2, height - windowSize));
                int startX = Math.Max(0, Math.Min(point.X - windowSize / 2, width - windowSize));
                cropPositions.Add((startX, startY));
            }
        );

        // 創建繪圖用的影像副本
        Mat drawMap = oriImg.Clone();
        string name = Path.GetFileNameWithoutExtension(inputPath);
        
        // 設定PNG壓縮參數 - 只需建立一次
        ImageEncodingParam[] pngParams = new ImageEncodingParam[] {
            new ImageEncodingParam(ImwriteFlags.PngCompression, 0)
        };

        // 處理所有裁剪位置
        // 將裁剪位置轉換為列表以便處理
        var cropPositionsList = cropPositions.ToList();

        // 使用 Parallel.For 並行處理裁剪任務
        Parallel.For(0, cropPositionsList.Count, new ParallelOptions { MaxDegreeOfParallelism = maxThreads }, i => {
            var pos = cropPositionsList[i];
            int startX = pos.x;
            int startY = pos.y;
            
            // 確保裁剪區域在圖像範圍內
            if (startX + windowSize <= width && startY + windowSize <= height)
            {
                // 裁剪原始灰度圖像
                Mat crop = new Mat(oriGray, new Rect(startX, startY, windowSize, windowSize)).Clone();
                
                // 將裁剪結果添加到集合中
                cropBag.Add(crop);
                
                // 繪製裁剪位置
                lock (drawMap)
                {
                    Cv2.Rectangle(drawMap, new Point(startX, startY), 
                        new Point(startX + windowSize, startY + windowSize), 
                        new Scalar(0, 0, 255), 2);
                }

                // 生成唯一檔名
                string timestamp = GetTimestamp();
                string outputPath = Path.Combine(outputDir, $"{name}_{i:D4}_{timestamp}.png");
                
                // 保存裁剪後的圖像
                Cv2.ImWrite(outputPath, crop, pngParams);
            }
        });

        // 儲存標記的圖像
        Cv2.ImWrite(Path.Combine(outputDir, $"{name}_ROImap.png"), drawMap);
        
        // 將ConcurrentBag轉換為List
        return cropBag.ToList();
    }
    catch (Exception ex)
    {
        // 發生錯誤時，清理已收集的Mat對象
        foreach (var crop in cropBag)
        {
            crop?.Dispose();
        }
        throw new Exception("Processing failed: " + ex.Message);
    }
    finally
    {
        // 釋放資源
        oriImg?.Dispose();
        img?.Dispose();
        oriGray?.Dispose();
        gray?.Dispose();
        binary?.Dispose();
        kernel?.Dispose();
        dilated?.Dispose();
        remask?.Dispose();
        labels?.Dispose();
        stats?.Dispose();
        centroids?.Dispose();
        edges?.Dispose();
        
        // 強制垃圾回收
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }
}
