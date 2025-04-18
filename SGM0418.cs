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

        // 形態學操作
        kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
        dilated = binary.MorphologyEx(MorphTypes.Open, kernel, iterations: 2);
        dilated = dilated.MorphologyEx(MorphTypes.Close, kernel, iterations: 2);

        // 復原遮罩處理
        remask = Cv2.ImRead("D:\\newSGM\\reconstruct_mask.tif", ImreadModes.Grayscale);
        remask = remask.Resize(newSize, 0, 0, InterpolationFlags.Nearest);

        Mat constructResult = Mat.Zeros(dilated.Size(), MatType.CV_8UC1);
        Reconstruct(dilated, remask, ref constructResult);

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

        // 邊緣檢測
        edges = largestComponent.Canny(100, 200);
        
        // 找到所有輪廓
        Point[][] contours;
        HierarchyIndex[] hierarchy;
        Cv2.FindContours(edges, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        
        // 找到最大輪廓
        int maxContourIdx = 0;
        double maxContourArea = 0;
        for (int i = 0; i < contours.Length; i++)
        {
            double area = Cv2.ContourArea(contours[i]);
            if (area > maxContourArea)
            {
                maxContourArea = area;
                maxContourIdx = i;
            }
        }
        
        // 計算沿著輪廓的固定距離
        int contourDistance = (int)(windowSize * (1 - overlap));
        
        Mat drawMap = img.Clone();
        int count = 0;
        
        // 確保有輪廓可用
        if (contours.Length > 0)
        {
            Point[] maxContour = contours[maxContourIdx];
            
            // 計算沿輪廓的總長度
            double contourLength = Cv2.ArcLength(maxContour, true);
            
            // 計算需要採樣的點數
            int numSamples = Math.Max(1, (int)(contourLength / contourDistance));
            
            // 遍歷輪廓上的等距離點
            for (int i = 0; i < numSamples; i++)
            {
                // 計算當前點在輪廓上的位置
                int idx = (int)((i * 1.0 / numSamples) * maxContour.Length);
                if (idx >= maxContour.Length) idx = maxContour.Length - 1;
                
                Point contourPoint = maxContour[idx];
                
                // 縮放回原始圖片大小
                int origX = contourPoint.X * downScale;
                int origY = contourPoint.Y * downScale;
                
                // 計算裁切區域
                int startX = Math.Max(0, origX - windowSize / 2);
                int startY = Math.Max(0, origY - windowSize / 2);
                
                // 確保不超出圖片範圍
                if (startX + windowSize > oriImg.Width)
                    startX = oriImg.Width - windowSize;
                if (startY + windowSize > oriImg.Height)
                    startY = oriImg.Height - windowSize;
                
                // 裁切原始灰度圖
                Rect cropRect = new Rect(startX, startY, windowSize, windowSize);
                Mat crop = new Mat(oriGray, cropRect);
                
                // 在顯示圖上標記裁切位置
                Point topleft = new Point((int)(startX / downScale), (int)(startY / downScale));
                Point btmright = new Point((int)((startX + windowSize) / downScale), (int)((startY + windowSize) / downScale));
                Cv2.Rectangle(drawMap, topleft, btmright, new Scalar(0, 0, 255), 2);
                
                // 標記輪廓點
                Cv2.Circle(drawMap, contourPoint, 3, new Scalar(0, 255, 0), -1);
                
                // 保存裁切後的圖片
                string timestamp = GetTimestamp();
                string name = Path.GetFileNameWithoutExtension(inputPath);
                string outputPath = Path.Combine(outputDir, $"{name}{timestamp}.png");
                ImageEncodingParam[] Params = new ImageEncodingParam[]
                {
                    new ImageEncodingParam(ImwriteFlags.PngCompression, 0)
                };
                
                Cv2.ImWrite(outputPath, crop, Params);
                crop.Dispose();
                count++;
            }
        }
        
        Cv2.ImWrite(outputDir + Path.GetFileNameWithoutExtension(inputPath) + "_ROImap.png", drawMap);
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
