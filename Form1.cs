using System;
using System.Windows.Forms;
using OpenCvSharp;
using System.IO;
using SAM2Prediction;

namespace SAM2OnnxInference
{
    public partial class Form1 : Form
    {
        SAM2Predictor Sam2 = null;
        Point[] Points = new Point[0];
        int[] Labels = new int[0]; // 1=前景, 0=背景
        string LoadImagePath = "";
        string VideoDir = "";
        Mat SrcImage = new Mat();
        Mat MaskImage = new Mat();
        Mat VisualizedImage = new Mat();
        float Score = 0.0f;
        bool LoadedModel = false;

        public Form1()
        {
            InitializeComponent();
        }

        private void btn_loadModel_Click(object sender, EventArgs e)
        {
            if (LoadedModel)
            {
                //釋放模型資源
                Sam2.Dispose();
                Sam2 = null;
                LoadedModel = false;

                //將btn背景設為預設色
                btn_loadModel.BackColor = System.Drawing.SystemColors.Control;

                //將btn text改為load model
                btn_loadModel.Text = "Load Model";
                btn_run.Enabled = false;
                buttonRunVideo.Enabled = false;
                buttonBatchRun.Enabled = false;
                return;
            }
            else
            {
                //載入SAM2模型
                // 設定模型路徑
                string encoderPath = "./model/sam2.1_hiera_tiny_encoder.onnx";
                string decoderPath = "./model/sam2.1_hiera_tiny_decoder.onnx";

                // 建立預測器
                Sam2 = new SAM2Predictor(encoderPath, decoderPath);
                LoadedModel = true;

                //將btn背景設為綠色
                btn_loadModel.BackColor = System.Drawing.Color.LightGreen;

                //將btn text改為dispose model
                btn_loadModel.Text = "Dispose Model";
                btn_run.Enabled = true;
                buttonRunVideo.Enabled = true;
                buttonBatchRun.Enabled = true;
            }
        }

        private void btn_run_Click(object sender, EventArgs e)
        {
            try
            {
                Console.WriteLine("=== SAM2 C# Demo (OpenCvSharp) ===\n");

                // 載入測試圖片
                if (LoadImagePath == "")
                {
                    MessageBox.Show("請先載入影像");
                    return;
                }
                SrcImage = Cv2.ImRead(LoadImagePath);

                for (int i = 0; i < Points.Length; i++)
                {
                    string labelText = Labels[i] == 1 ? "前景" : "背景";
                    Console.WriteLine($"  點 {i + 1}: ({Points[i].X}, {Points[i].Y}) - {labelText}");
                }

                //計算時間
                var watch = System.Diagnostics.Stopwatch.StartNew();

                // 執行預測
                Sam2.Predict(SrcImage, Points, Labels, ref MaskImage, ref Score);
                // 進行形態學處理 (膨脹後侵蝕)
                MaskImage = Sam2.OpenCloseMask(MaskImage, kernelSize:51, iterations: 3);

                // 停止計時
                watch.Stop();

                // 視覺化並儲存結果
                VisualizedImage = Sam2.Visualize(SrcImage, MaskImage, Points, Labels);
                pictureBoxRst.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(VisualizedImage);

                //顯示分數
                label_score.Text = $"{Score:F4}";
                //顯示時間
                var elapsedMs = watch.ElapsedMilliseconds;
                label_tecttime.Text = $"{elapsedMs} ms";

                Console.WriteLine($"\n=== 預測完成 ===");
                Console.WriteLine($"IOU 分數: {Score:F4}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"錯誤: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
            finally
            {
            }
        }

        #region UX
        //讓使用者點選picturebox來取得座標點，將座標點顯示在圖片上並且將座標儲存至Point[] points，按左鍵紀錄為前景點，案右鍵紀錄為背景點
        private void pictureBoxSrc_MouseClick(object sender, MouseEventArgs e)
        {
            if (SrcImage.Empty())
            {
                MessageBox.Show("請先載入影像");
                return;
            }

            // 計算點擊位置對應到原始影像的座標
            PictureBox pb = sender as PictureBox;
            float xRatio = (float)SrcImage.Width / pb.Width;
            float yRatio = (float)SrcImage.Height / pb.Height;
            int imgX = (int)(e.X * xRatio);
            int imgY = (int)(e.Y * yRatio);

            // 紀錄點和標籤
            Array.Resize(ref Points, Points.Length + 1);
            Array.Resize(ref Labels, Labels.Length + 1);
            Points[Points.Length - 1] = new Point(imgX, imgY);
            Labels[Labels.Length - 1] = e.Button == MouseButtons.Left ? 1 : 0; // 左鍵=前景, 右鍵=背景

            // 在圖片上繪製點
            Mat displayImg = SrcImage.Clone();
            for (int i = 0; i < Points.Length; i++)
            {
                Scalar color = Labels[i] == 1 ? new Scalar(0, 255, 0) : new Scalar(0, 0, 255); // 綠色=前景, 紅色=背景
                Cv2.Circle(displayImg, Points[i], 10, color, -1);
            }

            // update UI
            pictureBoxSrc.Invalidate();
            pictureBoxSrc.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(displayImg);
            Application.DoEvents();

            displayImg.Dispose();

            //inference
            btn_run_Click(null,null);
        }


        #endregion

        #region UI
        private void pictureBoxSrc_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.Copy;
        }

        private void pictureBoxSrc_DragDrop(object sender, DragEventArgs e)
        {
            //清除之前的點以及Labels以及結果圖片
            Points = new Point[0];
            Labels = new int[0];
            pictureBoxRst.Image = null;

            string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
            if (files.Length > 0)
            {
                string filePath = files[0];
                string ext = Path.GetExtension(filePath).ToLower();
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                {
                    LoadImagePath = filePath;
                    pictureBoxSrc.Image = System.Drawing.Image.FromFile(filePath);
                    SrcImage = Cv2.ImRead(filePath);
                }
                else
                {
                    MessageBox.Show("請拖曳影像檔案 (jpg, png, bmp)");
                }
            }
        }
        #endregion

        private void btn_SaveRst_Click(object sender, EventArgs e)
        {
            //判斷路徑是否存在，不存在則建立
            string outputPath = "./Result";
            if (!Directory.Exists(outputPath))
            {
                Directory.CreateDirectory(outputPath);
            }

            //取得當下時間
            string timeNow = DateTime.Now.ToString("yyyyMMdd_HHmmss");

            // 儲存遮罩
            string maskSavePath = Path.Combine(outputPath, timeNow + "_mask.png");
            Cv2.ImWrite(maskSavePath, MaskImage);

            // 儲存結果
            string visualSavePath = Path.Combine(outputPath, timeNow + "_visual.png");
            Cv2.ImWrite(visualSavePath, VisualizedImage);

            // 打開結果資料夾
            System.Diagnostics.Process.Start("explorer.exe", Path.GetFullPath(outputPath));
        }

        private void btn_clearPoint_Click(object sender, EventArgs e)
        {
            // 清除所有點
            Points = new Point[0];
            Labels = new int[0];
            //更新picturebox顯示原始圖片
            pictureBoxSrc.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(SrcImage);
            //若有result影像則清除
            pictureBoxRst.Image = null;
        }

        private void buttonBatchRun_Click(object sender, EventArgs e)
        {
            //讓使用者選擇資料夾
            FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
            {
                string folderPath = folderBrowserDialog.SelectedPath;
                string[] imageFiles = Directory.GetFiles(folderPath, "*.*", SearchOption.TopDirectoryOnly);
                Array.Resize(ref Points, 1);
                Array.Resize(ref Labels, 1);
                Labels[0] = 1;
                Points[0] = new Point(964, 964);
                foreach (string imageFile in imageFiles)
                {
                    string ext = Path.GetExtension(imageFile).ToLower();
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                    {
                        //載入圖片
                        LoadImagePath = imageFile;
                        SrcImage = Cv2.ImRead(LoadImagePath);

                        //執行預測
                        btn_run_Click(null, null);

                        //儲存結果
                        //判斷路徑是否存在，不存在則建立
                        string outputPath = "./Result";
                        if (!Directory.Exists(outputPath))
                        {
                            Directory.CreateDirectory(outputPath);
                        }
                        string fileName = Path.GetFileNameWithoutExtension(imageFile);
                        string visualSavePath = Path.Combine(outputPath, fileName + ".png");
                        Cv2.ImWrite(visualSavePath, VisualizedImage);
                    }
                }
            }
            //批次處理資料夾中的所有圖片

        }

        private void buttonRunVideo_Click(object sender, EventArgs e)
        {
            //使用者選擇照片資料夾，若VideoDir有上次記憶則開啟對話框時開啟至該路徑
            FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
            if (VideoDir != "")
            {
                folderBrowserDialog.SelectedPath = VideoDir;
            }
            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
            {
                VideoDir = folderBrowserDialog.SelectedPath;
            }
            else
            {
                return;
            }

            //建立結果資料夾
            string outputDir = "./VideoResult";
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            // 步驟 1: 初始化影片狀態
            var inferenceState = Sam2.InitVideoState(VideoDir);

            // 步驟 2: 在第一幀上添加標註點
            int annotationFrameIdx = 0;  // 在第 0 幀上標註
            int objectId = 1;  // 物件 ID

            // 定義標註點於影像中心
            Point[] points = new Point[]
            {
                new Point(966,966),
            };

            int[] labels = new int[] { 1 };  // 1 = 正向點, 0 = 負向點

            // 在第一幀上添加標註
            var (firstMask, iouScore) = Sam2.AddPointsToFrame(
                inferenceState,
                annotationFrameIdx,
                objectId,
                points,
                labels
            );

            Console.WriteLine($"第一幀標註完成, IOU: {iouScore:F4}");

            // 視覺化第一幀的結果
            using (var firstFrame = Cv2.ImRead(inferenceState.FramePaths[annotationFrameIdx]))
            using (var firstMaskMat = Sam2.MaskToMat(firstMask))
            using (var visualized = Sam2.Visualize(firstFrame, firstMaskMat, points, labels))
            {
                string outputPath = Path.Combine(outputDir, "frame_0_annotated.jpg");
                Cv2.ImWrite(outputPath, visualized);
                Console.WriteLine($"第一幀視覺化結果已儲存: {outputPath}");
            }

            // 步驟 3: 在整個影片中傳播追蹤
            var videoSegments = Sam2.PropagateInVideo(inferenceState);

            // 步驟 4: 儲存追蹤結果
            Console.WriteLine("\n儲存追蹤結果...");

            int visFrameStride = 1;  // 每 30 幀儲存一次視覺化結果

            for (int frameIdx = 0; frameIdx < inferenceState.FramePaths.Count; frameIdx++)
            {
                if (frameIdx % visFrameStride != 0 && frameIdx != 0)
                    continue;

                using (var frame = Cv2.ImRead(inferenceState.FramePaths[frameIdx]))
                {
                    // 對每個追蹤物件
                    foreach (var kvp in videoSegments[frameIdx])
                    {
                        int objId = kvp.Key;
                        float[,] maskArray = kvp.Value;

                        using (var maskMat = Sam2.MaskToMat(maskArray))
                        {
                            //將mask套用openclose
                            var processedMask = Sam2.OpenCloseMask(maskMat, kernelSize: 51, iterations: 3);
                            // 建立彩色遮罩
                            using (var colorMask = new Mat(maskMat.Size(), MatType.CV_8UC3, new Scalar(0, 255, 0)))
                            using (var maskedColor = new Mat())
                            {
                                Cv2.BitwiseAnd(colorMask, colorMask, maskedColor, processedMask);

                                using (var result = new Mat())
                                {
                                    Cv2.AddWeighted(frame, 0.7, maskedColor, 0.3, 0, result);

                                    string outputPath = Path.Combine(outputDir, $"frame_{frameIdx:D5}_obj{objId}.jpg");
                                    Cv2.ImWrite(outputPath, result);
                                    Console.WriteLine($"已儲存影格 {frameIdx}");
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine("\n影片追蹤完成!");
            Console.WriteLine($"結果已儲存到: {outputDir}");
        }
    }
}
