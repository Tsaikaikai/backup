using System;

public class Example
{
    public static void Main()
    {
        // 獲取 CIE 1931 標準光譜數據
        var spectrumData = DominantWavelengthCalculator.GetCIE1931SpectrumData();

        // 測試點座標
        double x_test = 0.4;
        double y_test = 0.35;

        // 白點座標 (D65)
        double x_white = 0.3127;
        double y_white = 0.3290;

        // 計算主波長
        var dwResult = DominantWavelengthCalculator.CalculateDominantWavelength(
            x_test, y_test, x_white, y_white, spectrumData);

        if (dwResult.IsSuccess)
        {
            Console.WriteLine($"主波長: {dwResult.Value} nm");

            // 計算純度
            var purityResult = DominantWavelengthCalculator.CalculatePurity(
                x_test, y_test, x_white, y_white, dwResult.Value, spectrumData);

            if (purityResult.IsSuccess)
            {
                Console.WriteLine($"純度: {purityResult.Value:F2}%");
            }
            else
            {
                Console.WriteLine($"純度計算失敗: {purityResult.ErrorMessage}");
            }
        }
        else
        {
            Console.WriteLine($"主波長計算失敗: {dwResult.ErrorMessage}");
        }
    }
}