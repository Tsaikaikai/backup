using System;
using System.Collections.Generic;
using System.Linq;

public class DominantWavelengthCalculator
{
    // 光譜軌跡數據類
    public class SpectrumPoint
    {
        public double Wavelength { get; set; }
        public double X { get; set; }
        public double Y { get; set; }
    }

    // 計算結果類
    public class CalculationResult
    {
        public bool IsSuccess { get; set; }
        public double? Value { get; set; }
        public string ErrorMessage { get; set; }
    }

    /// <summary>
    /// 獲取 CIE 1931 標準光譜軌跡數據
    /// </summary>
    public static List<SpectrumPoint> GetCIE1931SpectrumData()
    {
        return new List<SpectrumPoint>
        {
            new SpectrumPoint { Wavelength = 380, X = 0.1741, Y = 0.0050 },
            new SpectrumPoint { Wavelength = 385, X = 0.1740, Y = 0.0050 },
            new SpectrumPoint { Wavelength = 390, X = 0.1738, Y = 0.0049 },
            new SpectrumPoint { Wavelength = 395, X = 0.1736, Y = 0.0049 },
            new SpectrumPoint { Wavelength = 400, X = 0.1733, Y = 0.0048 },
            new SpectrumPoint { Wavelength = 405, X = 0.1730, Y = 0.0048 },
            new SpectrumPoint { Wavelength = 410, X = 0.1726, Y = 0.0048 },
            new SpectrumPoint { Wavelength = 415, X = 0.1721, Y = 0.0048 },
            new SpectrumPoint { Wavelength = 420, X = 0.1714, Y = 0.0051 },
            new SpectrumPoint { Wavelength = 425, X = 0.1703, Y = 0.0058 },
            new SpectrumPoint { Wavelength = 430, X = 0.1689, Y = 0.0069 },
            new SpectrumPoint { Wavelength = 435, X = 0.1669, Y = 0.0086 },
            new SpectrumPoint { Wavelength = 440, X = 0.1644, Y = 0.0109 },
            new SpectrumPoint { Wavelength = 445, X = 0.1611, Y = 0.0138 },
            new SpectrumPoint { Wavelength = 450, X = 0.1566, Y = 0.0177 },
            new SpectrumPoint { Wavelength = 455, X = 0.1510, Y = 0.0227 },
            new SpectrumPoint { Wavelength = 460, X = 0.1440, Y = 0.0297 },
            new SpectrumPoint { Wavelength = 465, X = 0.1355, Y = 0.0399 },
            new SpectrumPoint { Wavelength = 470, X = 0.1241, Y = 0.0578 },
            new SpectrumPoint { Wavelength = 475, X = 0.1096, Y = 0.0868 },
            new SpectrumPoint { Wavelength = 480, X = 0.0913, Y = 0.1327 },
            new SpectrumPoint { Wavelength = 485, X = 0.0687, Y = 0.2007 },
            new SpectrumPoint { Wavelength = 490, X = 0.0454, Y = 0.2950 },
            new SpectrumPoint { Wavelength = 495, X = 0.0235, Y = 0.4127 },
            new SpectrumPoint { Wavelength = 500, X = 0.0082, Y = 0.5384 },
            new SpectrumPoint { Wavelength = 505, X = 0.0039, Y = 0.6548 },
            new SpectrumPoint { Wavelength = 510, X = 0.0139, Y = 0.7502 },
            new SpectrumPoint { Wavelength = 515, X = 0.0389, Y = 0.8120 },
            new SpectrumPoint { Wavelength = 520, X = 0.0743, Y = 0.8338 },
            new SpectrumPoint { Wavelength = 525, X = 0.1142, Y = 0.8262 },
            new SpectrumPoint { Wavelength = 530, X = 0.1547, Y = 0.8059 },
            new SpectrumPoint { Wavelength = 535, X = 0.1929, Y = 0.7816 },
            new SpectrumPoint { Wavelength = 540, X = 0.2296, Y = 0.7543 },
            new SpectrumPoint { Wavelength = 545, X = 0.2658, Y = 0.7243 },
            new SpectrumPoint { Wavelength = 550, X = 0.3016, Y = 0.6923 },
            new SpectrumPoint { Wavelength = 555, X = 0.3373, Y = 0.6589 },
            new SpectrumPoint { Wavelength = 560, X = 0.3731, Y = 0.6245 },
            new SpectrumPoint { Wavelength = 565, X = 0.4087, Y = 0.5896 },
            new SpectrumPoint { Wavelength = 570, X = 0.4441, Y = 0.5547 },
            new SpectrumPoint { Wavelength = 575, X = 0.4788, Y = 0.5202 },
            new SpectrumPoint { Wavelength = 580, X = 0.5125, Y = 0.4866 },
            new SpectrumPoint { Wavelength = 585, X = 0.5448, Y = 0.4544 },
            new SpectrumPoint { Wavelength = 590, X = 0.5752, Y = 0.4242 },
            new SpectrumPoint { Wavelength = 595, X = 0.6029, Y = 0.3965 },
            new SpectrumPoint { Wavelength = 600, X = 0.6270, Y = 0.3725 },
            new SpectrumPoint { Wavelength = 605, X = 0.6482, Y = 0.3514 },
            new SpectrumPoint { Wavelength = 610, X = 0.6658, Y = 0.3340 },
            new SpectrumPoint { Wavelength = 615, X = 0.6801, Y = 0.3197 },
            new SpectrumPoint { Wavelength = 620, X = 0.6915, Y = 0.3083 },
            new SpectrumPoint { Wavelength = 625, X = 0.7006, Y = 0.2993 },
            new SpectrumPoint { Wavelength = 630, X = 0.7079, Y = 0.2920 },
            new SpectrumPoint { Wavelength = 635, X = 0.7140, Y = 0.2859 },
            new SpectrumPoint { Wavelength = 640, X = 0.7190, Y = 0.2809 },
            new SpectrumPoint { Wavelength = 645, X = 0.7230, Y = 0.2770 },
            new SpectrumPoint { Wavelength = 650, X = 0.7260, Y = 0.2740 },
            new SpectrumPoint { Wavelength = 655, X = 0.7283, Y = 0.2717 },
            new SpectrumPoint { Wavelength = 660, X = 0.7300, Y = 0.2700 },
            new SpectrumPoint { Wavelength = 665, X = 0.7311, Y = 0.2689 },
            new SpectrumPoint { Wavelength = 670, X = 0.7320, Y = 0.2680 },
            new SpectrumPoint { Wavelength = 675, X = 0.7327, Y = 0.2673 },
            new SpectrumPoint { Wavelength = 680, X = 0.7334, Y = 0.2666 },
            new SpectrumPoint { Wavelength = 685, X = 0.7340, Y = 0.2660 },
            new SpectrumPoint { Wavelength = 690, X = 0.7344, Y = 0.2656 },
            new SpectrumPoint { Wavelength = 695, X = 0.7346, Y = 0.2654 },
            new SpectrumPoint { Wavelength = 700, X = 0.7347, Y = 0.2653 }
        };
    }

    /// <summary>
    /// 計算主波長
    /// </summary>
    /// <param name="x_test">測試點的 x 色度座標</param>
    /// <param name="y_test">測試點的 y 色度座標</param>
    /// <param name="x_white">白點的 x 色度座標</param>
    /// <param name="y_white">白點的 y 色度座標</param>
    /// <param name="spectrumData">光譜軌跡數據列表</param>
    /// <returns>計算結果，包含主波長值或錯誤信息</returns>
    public static CalculationResult CalculateDominantWavelength(
        double x_test, double y_test,
        double x_white, double y_white,
        List<SpectrumPoint> spectrumData)
    {
        try
        {
            // 檢查光譜數據
            if (spectrumData == null || spectrumData.Count < 2)
            {
                return new CalculationResult
                {
                    IsSuccess = false,
                    ErrorMessage = "光譜數據不足"
                };
            }

            // 方向向量
            double dx = x_test - x_white;
            double dy = y_test - y_white;

            // 檢查是否為同一點
            if (Math.Abs(dx) < 0.0000001 && Math.Abs(dy) < 0.0000001)
            {
                return new CalculationResult
                {
                    IsSuccess = false,
                    ErrorMessage = "測試點與白點重合"
                };
            }

            // 尋找最接近的光譜軌跡點
            double minDistance = double.MaxValue;
            double foundWavelength = 0;
            double foundX = 0;
            double foundY = 0;

            // 遍歷所有光譜軌跡點
            foreach (var point in spectrumData)
            {
                double wavelength = point.Wavelength;
                double x_spec = point.X;
                double y_spec = point.Y;

                // 過濾無效數據
                if (wavelength == 0 || x_spec == 0)
                    continue;

                // 計算參數 t
                double t;
                bool use_x = false;

                // 使用 x 計算 t
                double t_x = double.MaxValue;
                if (Math.Abs(dx) > 0.00001)
                {
                    t_x = (x_spec - x_white) / dx;
                    use_x = true;
                }

                // 使用 y 計算 t
                double t_y = double.MaxValue;
                if (Math.Abs(dy) > 0.00001)
                {
                    t_y = (y_spec - y_white) / dy;
                }

                // 選擇較可靠的 t 值
                t = use_x ? t_x : t_y;

                // 只考慮 t > 1 (在測試點之外，朝向光譜軌跡)
                if (t <= 1)
                    continue;

                // 限制 t 過大(避免數值誤差)
                if (t > 100)
                    continue;

                // 計算射線上的點
                double x_calc = x_white + t * dx;
                double y_calc = y_white + t * dy;

                // 計算與光譜軌跡點的距離
                double distance = Math.Sqrt(
                    Math.Pow(x_calc - x_spec, 2) + 
                    Math.Pow(y_calc - y_spec, 2)
                );

                // 更新最小距離
                if (distance < minDistance)
                {
                    minDistance = distance;
                    foundWavelength = wavelength;
                    foundX = x_spec;
                    foundY = y_spec;
                }
            }

            // 檢查是否找到合理的交點
            // 距離閾值，根據實際需求調整
            if (minDistance > 0.1)
            {
                return new CalculationResult
                {
                    IsSuccess = false,
                    ErrorMessage = "未找到有效的主波長交點"
                };
            }

            return new CalculationResult
            {
                IsSuccess = true,
                Value = foundWavelength
            };
        }
        catch (Exception ex)
        {
            return new CalculationResult
            {
                IsSuccess = false,
                ErrorMessage = $"計算錯誤: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// 計算純度
    /// </summary>
    /// <param name="x_test">測試點的 x 色度座標</param>
    /// <param name="y_test">測試點的 y 色度座標</param>
    /// <param name="x_white">白點的 x 色度座標</param>
    /// <param name="y_white">白點的 y 色度座標</param>
    /// <param name="dominantWL">主波長值</param>
    /// <param name="spectrumData">光譜軌跡數據列表</param>
    /// <returns>計算結果，包含純度值(百分比)或錯誤信息</returns>
    public static CalculationResult CalculatePurity(
        double x_test, double y_test,
        double x_white, double y_white,
        double? dominantWL,
        List<SpectrumPoint> spectrumData)
    {
        try
        {
            // 檢查主波長是否有效
            if (!dominantWL.HasValue || dominantWL.Value == 0)
            {
                return new CalculationResult
                {
                    IsSuccess = false,
                    ErrorMessage = "主波長無效"
                };
            }

            // 檢查光譜數據
            if (spectrumData == null || spectrumData.Count < 2)
            {
                return new CalculationResult
                {
                    IsSuccess = false,
                    ErrorMessage = "光譜數據不足"
                };
            }

            // 尋找主波長對應的光譜座標
            double x_spec = 0;
            double y_spec = 0;
            bool found = false;

            // 精確查找最接近的點
            var closestPoint = spectrumData
                .Where(p => Math.Abs(p.Wavelength - dominantWL.Value) < 0.5)
                .FirstOrDefault();

            if (closestPoint != null)
            {
                x_spec = closestPoint.X;
                y_spec = closestPoint.Y;
                found = true;
            }
            else
            {
                // 如果沒有精確值，使用最接近的
                double minDiff = double.MaxValue;
                foreach (var point in spectrumData)
                {
                    double diff = Math.Abs(point.Wavelength - dominantWL.Value);
                    if (diff < minDiff)
                    {
                        minDiff = diff;
                        x_spec = point.X;
                        y_spec = point.Y;
                        found = true;
                    }
                }
            }

            if (!found)
            {
                return new CalculationResult
                {
                    IsSuccess = false,
                    ErrorMessage = "找不到對應的光譜座標"
                };
            }

            // 計算距離
            // 從白點到測試點的距離
            double d_test = Math.Sqrt(
                Math.Pow(x_test - x_white, 2) + 
                Math.Pow(y_test - y_white, 2)
            );

            // 從白點到光譜軌跡的距離
            double d_spectrum = Math.Sqrt(
                Math.Pow(x_spec - x_white, 2) + 
                Math.Pow(y_spec - y_white, 2)
            );

            // 計算純度 (百分比)
            if (d_spectrum > 0.000001)
            {
                double purity = (d_test / d_spectrum) * 100.0;

                // 限制在 0-100% 範圍
                purity = Math.Max(0, Math.Min(100, purity));

                return new CalculationResult
                {
                    IsSuccess = true,
                    Value = purity
                };
            }
            else
            {
                return new CalculationResult
                {
                    IsSuccess = false,
                    ErrorMessage = "光譜距離過小"
                };
            }
        }
        catch (Exception ex)
        {
            return new CalculationResult
            {
                IsSuccess = false,
                ErrorMessage = $"計算錯誤: {ex.Message}"
            };
        }
    }
}