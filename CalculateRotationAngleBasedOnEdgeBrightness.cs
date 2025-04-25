    public static double CalculateRotationAngleBasedOnEdgeBrightness(Mat inputImage, int cornerSize)
    {
        // 確保 cornerSize 不超過影像尺寸的一半
        cornerSize = Math.Min(cornerSize, Math.Min(inputImage.Width, inputImage.Height) / 2);
        
        // 定義四個角落區域的座標 (左上、右上、右下、左下)
        Point[][] cornerRegions = new Point[4][]
        {
            new Point[] { new Point(0, 0), new Point(cornerSize, cornerSize) },                                  // 左上 (0)
            new Point[] { new Point(inputImage.Width - cornerSize, 0), new Point(inputImage.Width, cornerSize) }, // 右上 (1)
            new Point[] { new Point(inputImage.Width - cornerSize, inputImage.Height - cornerSize), new Point(inputImage.Width, inputImage.Height) }, // 右下 (2)
            new Point[] { new Point(0, inputImage.Height - cornerSize), new Point(cornerSize, inputImage.Height) }  // 左下 (3)
        };

        // 計算每個角落的平均亮度
        double[] cornerBrightness = new double[4];
        for (int i = 0; i < 4; i++)
        {
            Point topLeft = cornerRegions[i][0];
            Point bottomRight = cornerRegions[i][1];
            
            // 取出角落區域
            Rect roi = new Rect(topLeft.X, topLeft.Y, bottomRight.X - topLeft.X, bottomRight.Y - topLeft.Y);
            Mat cornerMat = new Mat(inputImage, roi);
            
            // 轉換為灰階圖片
            Mat grayCorner = new Mat();
            if (inputImage.Channels() > 1)
            {
                Cv2.CvtColor(cornerMat, grayCorner, ColorConversionCodes.BGR2GRAY);
            }
            else
            {
                grayCorner = cornerMat.Clone();
            }
            
            // 計算平均亮度
            Scalar mean = Cv2.Mean(grayCorner);
            cornerBrightness[i] = mean.Val0;
            
            // 釋放資源
            cornerMat.Dispose();
            grayCorner.Dispose();
        }

        // 定義四個邊緣（由相鄰的兩個角落組成）
        int[][] edges = new int[][] 
        {
            new int[] { 0, 1 },  // 上邊：左上、右上
            new int[] { 1, 2 },  // 右邊：右上、右下
            new int[] { 2, 3 },  // 下邊：右下、左下
            new int[] { 3, 0 }   // 左邊：左下、左上
        };

        // 計算每條邊的亮度總和
        double[] edgeBrightness = new double[4];
        for (int i = 0; i < 4; i++)
        {
            edgeBrightness[i] = cornerBrightness[edges[i][0]] + cornerBrightness[edges[i][1]];
        }

        // 找出亮度總和最高的邊
        int brightestEdgeIndex = Array.IndexOf(edgeBrightness, edgeBrightness.Max());

        // 根據最亮邊的位置決定旋轉角度
        double rotationAngle = 0;
        switch (brightestEdgeIndex)
        {
            case 0: // 上邊應該旋轉到底部
                rotationAngle = 180;
                break;
            case 1: // 右邊應該旋轉到底部
                rotationAngle = 90;
                break;
            case 2: // 下邊已經在底部
                rotationAngle = 0;
                break;
            case 3: // 左邊應該旋轉到底部
                rotationAngle = -90;
                break;
        }

        return rotationAngle;
    }
