public static int ProcessAndCropImage(string inputPath, string outputDir, int threshold, int windowSize, int moveStep, int downScale)
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
        largestComponent.Dispose();

        // 獲取圖像尺寸
        int height = edges.Rows;
        int width = edges.Cols;
        int count = 0;
        Mat drawMap = img.Clone();

        // 找到所有邊緣像素點
        Mat nonZeroCoords = edges.FindNonZero();
        if (nonZeroCoords == null || nonZeroCoords.Rows == 0)
        {
            return 0; // 沒有找到邊緣
        }

        // 將邊緣像素點轉換為列表
        List<Point> edgePoints = new List<Point>();
        for (int i = 0; i < nonZeroCoords.Rows; i++)
        {
            edgePoints.Add(nonZeroCoords.At<Point>(i));
        }
        nonZeroCoords.Dispose();

        // 按x座標排序
        edgePoints.Sort((p1, p2) => p1.X.CompareTo(p2.X));

        // 計算邊緣的最小和最大x座標
        int minX = edgePoints.First().X;
        int maxX = edgePoints.Last().X;

        // 初始化已處理的點集合
        HashSet<string> processedAreas = new HashSet<string>();

        // 沿著邊緣每隔moveStep選取一個點作為窗口中心
        for (int x = minX; x <= maxX; x += moveStep)
        {
            // 找出所有x座標等於當前x的邊緣點
            var pointsAtX = edgePoints.Where(p => p.X == x).ToList();
            if (pointsAtX.Count == 0)
                continue;

            // 對於每個x座標的點，提取窗口
            foreach (var point in pointsAtX)
            {
                // 計算窗口的起始位置
                int startY = Math.Max(0, Math.Min(point.Y - windowSize / 2, height - windowSize));
                int startX = Math.Max(0, Math.Min(point.X - windowSize / 2, width - windowSize));

                // 生成區域識別碼
                string areaKey = $"{startX / (windowSize / 4)},{startY / (windowSize / 4)}";

                // 避免重複處理相近區域
                if (processedAreas.Contains(areaKey))
                    continue;

                processedAreas.Add(areaKey);

                // 裁剪原始灰度圖
                Rect cropRect = new Rect(startX, startY, windowSize, windowSize);
                Mat crop = new Mat(oriGray, cropRect);

                // 繪製裁剪位置
                Point topleft = new Point((int)(startX / downScale), (int)startY / downScale);
                Point btmright = new Point((int)(startX / downScale + windowSize / downScale), (int)(startY / downScale + windowSize / downScale));
                Cv2.Rectangle(drawMap, topleft, btmright, new Scalar(0, 0, 255), 2);

                // 保存裁剪圖片
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

        // 另外處理y方向的邊緣
        edgePoints.Sort((p1, p2) => p1.Y.CompareTo(p2.Y));
        int minY = edgePoints.First().Y;
        int maxY = edgePoints.Last().Y;

        for (int y = minY; y <= maxY; y += moveStep)
        {
            // 找出所有y座標等於當前y的邊緣點
            var pointsAtY = edgePoints.Where(p => p.Y == y).ToList();
            if (pointsAtY.Count == 0)
                continue;

            // 對於每個y座標的點，提取窗口
            foreach (var point in pointsAtY)
            {
                // 計算窗口的起始位置
                int startY = Math.Max(0, Math.Min(point.Y - windowSize / 2, height - windowSize));
                int startX = Math.Max(0, Math.Min(point.X - windowSize / 2, width - windowSize));

                // 生成區域識別碼
                string areaKey = $"{startX / (windowSize / 4)},{startY / (windowSize / 4)}";

                // 避免重複處理相近區域
                if (processedAreas.Contains(areaKey))
                    continue;

                processedAreas.Add(areaKey);

                // 裁剪原始灰度圖
                Rect cropRect = new Rect(startX, startY, windowSize, windowSize);
                Mat crop = new Mat(oriGray, cropRect);

                // 繪製裁剪位置
                Point topleft = new Point((int)(startX / downScale), (int)startY / downScale);
                Point btmright = new Point((int)(startX / downScale + windowSize / downScale), (int)(startY / downScale + windowSize / downScale));
                Cv2.Rectangle(drawMap, topleft, btmright, new Scalar(0, 0, 255), 2);

                // 保存裁剪圖片
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
