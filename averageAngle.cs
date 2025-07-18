    public static double CalculateAverageAngle(double slope1, double slope2)
    {
        // 將斜率轉換為角度（弧度）
        double angle1 = Math.Atan(slope1);
        double angle2 = Math.Atan(slope2);
        
        // 將角度轉換為度數
        double degree1 = angle1 * 180.0 / Math.PI;
        double degree2 = angle2 * 180.0 / Math.PI;
        
        // 計算兩個角度的差異
        double diff = Math.Abs(degree1 - degree2);
        
        // 如果差異大於90度，需要考慮週期性
        // 因為直線的角度具有180度的週期性
        if (diff > 90)
        {
            // 調整其中一個角度，使差異最小
            if (degree1 > degree2)
            {
                degree2 += 180;
            }
            else
            {
                degree1 += 180;
            }
        }
        
        // 計算平均角度
        double averageAngle = (degree1 + degree2) / 2.0;
        
        // 將結果標準化到 [-90, 90] 範圍內
        while (averageAngle > 90)
        {
            averageAngle -= 180;
        }
        while (averageAngle < -90)
        {
            averageAngle += 180;
        }
        
        return averageAngle;
    }
