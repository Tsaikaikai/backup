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

                //largestComponent = largestComponent.Resize(new Size(oriImg.Width, oriImg.Height), 0, 0, InterpolationFlags.Nearest);

                // 邊緣檢測
                edges = largestComponent.Canny(50, 250);
                Cv2.ImWrite(Path.ChangeExtension(inputPath, "_edge.jpg"), edges);
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
                    // 將縮小的座標轉回原始大小
                    int origX = point.X * downScale;
                    int origY = point.Y * downScale;

                    // 計算裁切區域，確保窗口以邊緣點為中心
                    int startX = Math.Max(0, origX - windowSize / 2);
                    int startY = Math.Max(0, origY - windowSize / 2);

                    // 確保裁切區域不超出圖像範圍
                    if (startX + windowSize > oriImg.Width)
                        startX = oriImg.Width - windowSize;
                    if (startY + windowSize > oriImg.Height)
                        startY = oriImg.Height - windowSize;

                    // 裁切原始灰度圖並添加到結果列表
                    Rect cropRect = new Rect(startX, startY, windowSize, windowSize);
                    Mat crop = new Mat(oriGray, cropRect).Clone(); // 使用clone()以確保獨立副本
                    croppedImages.Add(crop);

                    // 在顯示圖上標記裁切位置
                    Point topleft = new Point((int)(startX / downScale), (int)(startY / downScale));
                    Point btmright = new Point((int)((startX + windowSize) / downScale), (int)((startY + windowSize) / downScale));
                    Cv2.Rectangle(drawMap, topleft, btmright, new Scalar(0, 0, 255), 2);

                    // 標記邊緣點
                    Cv2.Circle(drawMap, point, 3, new Scalar(0, 255, 0), -1);
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