public static int ProcessAndCropImage(string inputPath, string outputDir, int threshold, int windowSize, int stepDistance, int downScale)
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

        // 提取所有邊緣點
        Mat nonZeroCoords = edges.FindNonZero();
        if (nonZeroCoords == null || nonZeroCoords.Rows == 0)
        {
            return 0;
        }

        // 將邊緣點轉換成列表，以便處理
        List<Point> edgePoints = new List<Point>();
        for (int i = 0; i < nonZeroCoords.Rows; i++)
        {
            edgePoints.Add(nonZeroCoords.At<Point>(i));
        }
        nonZeroCoords.Dispose();

        // 對邊緣點進行排序（按照x座標，然後按照y座標）
        edgePoints = edgePoints.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();

        // 沿著邊緣以固定間距選取點
        List<Point> sampledPoints = new List<Point>();
        if (edgePoints.Count > 0)
        {
            sampledPoints.Add(edgePoints[0]);
            Point lastPoint = edgePoints[0];

            for (int i = 1; i < edgePoints.Count; i++)
            {
                Point currentPoint = edgePoints[i];
                double distance = Math.Sqrt(Math.Pow(currentPoint.X - lastPoint.X, 2) + Math.Pow(currentPoint.Y - lastPoint.Y, 2));
                
                if (distance >= stepDistance)
                {
                    sampledPoints.Add(currentPoint);
                    lastPoint = currentPoint;
                }
            }
        }

        // 在取樣點位置放置窗口並進行切割
        int count = 0;
        Mat drawMap = oriImg.Clone();
        int height = oriImg.Rows;
        int width = oriImg.Cols;

        foreach (Point point in sampledPoints)
        {
            // 調整窗口起始位置，使窗口中心在邊緣點上
            int startY = Math.Max(0, Math.Min(point.Y - windowSize / 2, height - windowSize));
            int startX = Math.Max(0, Math.Min(point.X - windowSize / 2, width - windowSize));

            // 裁剪原始灰度圖像
            Rect cropRect = new Rect(startX, startY, windowSize, windowSize);
            Mat crop = new Mat(oriGray, cropRect);

            // 繪製裁剪位置
            Cv2.Rectangle(drawMap, new Point(startX, startY), new Point(startX + windowSize, startY + windowSize), new Scalar(0, 0, 255), 2);

            // 保存裁剪後的圖像
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
