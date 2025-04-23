public static List<Mat> ProcessAndCropImage(string inputPath, string outputDir, int threshold, int windowSize, double overlap, int downScale)
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
    List<Mat> croppedImages = new List<Mat>();

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

        // 邊緣檢測
        edges = largestComponent.Canny(50, 250);
        Cv2.ImWrite(Path.ChangeExtension(inputPath, "_edge.jpg"), edges);
        
        // 計算物體中心點，用於確定內外側方向
        Point objectCenter = GetObjectCenter(largestComponent);
        largestComponent.Dispose();

        // 找到所有邊緣點
        Mat nonZeroCoords = edges.FindNonZero();

        // 如果沒有找到邊緣點，則退出
        if (nonZeroCoords == null || nonZeroCoords.Rows == 0)
        {
            throw new Exception("未檢測到邊緣");
        }

        // 計算沿著邊緣的固定距離間隔
        int stepDistance = (int)(windowSize * (1 - overlap) / downScale);

        // 將邊緣點轉換為列表以便處理
        List<Point> edgePoints = new List<Point>();
        for (int i = 0; i < nonZeroCoords.Rows; i++)
        {
            edgePoints.Add(nonZeroCoords.At<Point>(i));
        }

        // 對邊緣點進行排序
        edgePoints = edgePoints.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();

        Mat drawMap = img.Clone();

        // 採樣點的列表，確保每兩點之間的距離大於等於 stepDistance
        List<Point> sampledPoints = new List<Point>();

        // 直接使用平方距離來比較，可以避免每次都計算平方根，提升效能
        double stepDistanceSquared = stepDistance * stepDistance;

        foreach (Point edgePoint in edgePoints)
        {
            bool isFarEnough = true;

            foreach (Point sample in sampledPoints)
            {
                double dx = edgePoint.X - sample.X;
                double dy = edgePoint.Y - sample.Y;
                double distanceSquared = dx * dx + dy * dy;
                if (distanceSquared < stepDistanceSquared)
                {
                    isFarEnough = false;
                    break;
                }
            }

            if (isFarEnough)
            {
                sampledPoints.Add(edgePoint);
            }
        }

        // 使用採樣點切割圖像
        foreach (Point point in sampledPoints)
        {
            // 為每個採樣點找最近的兩個採樣點
            List<Point> closestPoints = FindClosestPoints(point, sampledPoints, 2);
            
            // 計算角度
            double angle = CalculateAverageAngle(point, closestPoints, objectCenter);
            
            // 將縮小的座標轉回原始大小
            int origX = point.X * downScale;
            int origY = point.Y * downScale;
            
            // 建立旋轉矩形
            RotatedRect rotatedRect = new RotatedRect(
                new Point2f(origX, origY),
                new Size2f(windowSize, windowSize),
                (float)angle
            );

            // 取得旋轉矩形的四個頂點
            Point2f[] vertices = rotatedRect.Points();
            
            // 將頂點轉換為整數座標
            Point[] intVertices = Array.ConvertAll(vertices, p => new Point((int)p.X, (int)p.Y));
            
            // 將旋轉矩形的四個頂點繪製在地圖上
            for (int i = 0; i < 4; i++)
            {
                Cv2.Line(
                    drawMap,
                    new Point((int)(intVertices[i].X / downScale), (int)(intVertices[i].Y / downScale)),
                    new Point((int)(intVertices[(i + 1) % 4].X / downScale), (int)(intVertices[(i + 1) % 4].Y / downScale)),
                    new Scalar(0, 0, 255),
                    2
                );
            }
            
            // 標記邊緣點
            Cv2.Circle(drawMap, point, 3, new Scalar(0, 255, 0), -1);
            
            // 創建旋轉裁剪的仿射變換矩陣
            Mat rotationMatrix = Cv2.GetRotationMatrix2D(
                new Point2f(origX, origY),
                angle,
                1.0
            );
            
            // 旋轉整個圖像
            Mat rotatedImage = new Mat();
            Cv2.WarpAffine(
                oriGray,
                rotatedImage,
                rotationMatrix,
                oriGray.Size()
            );
            
            // 從旋轉後的圖像中裁剪出所需區域
            int startX = Math.Max(0, origX - windowSize / 2);
            int startY = Math.Max(0, origY - windowSize / 2);
            
            // 確保裁切區域不超出圖像範圍
            if (startX + windowSize > rotatedImage.Width)
                startX = rotatedImage.Width - windowSize;
            if (startY + windowSize > rotatedImage.Height)
                startY = rotatedImage.Height - windowSize;
                
            Rect cropRect = new Rect(startX, startY, windowSize, windowSize);
            Mat crop = new Mat(rotatedImage, cropRect).Clone();
            croppedImages.Add(crop);
            
            rotatedImage.Dispose();
            rotationMatrix.Dispose();
        }

        // 如果輸出目錄存在，保存標記圖
        if (!string.IsNullOrEmpty(outputDir))
        {
            Cv2.ImWrite(outputDir + Path.GetFileNameWithoutExtension(inputPath) + "_ROImap.png", drawMap);
        }

        return croppedImages;
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

// 計算物體中心點
private static Point GetObjectCenter(Mat binaryImage)
{
    Moments m = Cv2.Moments(binaryImage);
    int cx = (int)(m.M10 / m.M00);
    int cy = (int)(m.M01 / m.M00);
    return new Point(cx, cy);
}

// 找到最近的N個點
private static List<Point> FindClosestPoints(Point currentPoint, List<Point> allPoints, int count)
{
    return allPoints
        .Where(p => p != currentPoint)
        .OrderBy(p => Math.Pow(p.X - currentPoint.X, 2) + Math.Pow(p.Y - currentPoint.Y, 2))
        .Take(count)
        .ToList();
}

// 計算向量角度（弧度）
private static double CalculateAngle(Point start, Point end)
{
    double dx = end.X - start.X;
    double dy = end.Y - start.Y;
    return Math.Atan2(dy, dx);
}

// 計算角度的平均值，並確保矩形底邊朝向內部
private static double CalculateAverageAngle(Point currentPoint, List<Point> neighbors, Point center)
{
    // 計算與相鄰點的角度
    List<double> angles = new List<double>();
    foreach (var neighbor in neighbors)
    {
        angles.Add(CalculateAngle(currentPoint, neighbor));
    }
    
    // 計算平均角度
    double avgAngle = angles.Average();
    
    // 計算從邊緣點到中心點的角度
    double centerAngle = CalculateAngle(currentPoint, center);
    
    // 調整角度，使矩形底邊朝向內部
    // 將平均角度調整為垂直於指向中心的方向
    double perpendicularAngle = centerAngle + Math.PI/2;
    
    // 將角度轉換為度數
    double angleDegrees = perpendicularAngle * 180.0 / Math.PI;
    
    // 確保角度在0到360度之間
    while (angleDegrees < 0)
        angleDegrees += 360;
    while (angleDegrees >= 360)
        angleDegrees -= 360;
        
    return angleDegrees;
}
