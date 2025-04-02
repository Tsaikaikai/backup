using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace newSGM
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string inputDirectory = "D:\\newSGM\\Data\\ColorSlice_Image0";
            string outputDirectory = "D:\\newSGM\\Data\\ColorSlice_Image0\\crop";

            try
            {
                var start = DateTime.Now;
                ProcessDirectory(inputDirectory, outputDirectory);
                var end = DateTime.Now;
                Console.WriteLine($"共耗費 {(end - start).TotalSeconds} 秒，處理完成！");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"處理過程中發生錯誤: {ex.Message}");
            }
        }

        public static void ProcessDirectory(string inputDir, string outputDir, int threshold = 20, int windowSize = 644, double overlap = 0.5, int downScale = 15)
        {
            Directory.CreateDirectory(outputDir);
            int totalCrops = 0;
            string[] files = Directory.GetFiles(inputDir, "*.tiff");

            // 設定最大同步執行數量，例如：4
            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = 4
            };

            foreach (string file in files)
            {
            //Parallel.ForEach(files, parallelOptions, file =>
            //{
               try
               {
                   int crops = SGMAlgorithm.ProcessAndCropImage(file, outputDir, threshold, windowSize, overlap, downScale);
                   totalCrops += crops;
                   Console.WriteLine($"成功處理 {file}，產生了 {crops} 張切割圖片");
               }
               catch (Exception ex)
               {
                   Console.WriteLine($"處理 {file} 時發生錯誤: {ex.Message}");
               }
            //});
            }
        }
    }
}
