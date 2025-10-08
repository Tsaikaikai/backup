using Brandy;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using PatternMatch;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Text.Json;
using System.Windows.Forms;

namespace AIF_2D_AOI
{
    public class AIFAlgorithm
    {
        #region Initial
        CAlgorithmConfig Config = new CAlgorithmConfig();
        ChangeDetectionModel Core = null;

        public AIFAlgorithm()
        {

        }

        public bool LoadConfig(string ConfigPath)
        {
            try
            {
                Config = Config.ReadXmlConfig(ConfigPath);
                return true;
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        public class CutImage : IDisposable
        {
            public Mat ImageA;
            public Mat ImageB;
            public float[] floatImageA;
            public float[] floatImageB;
            public int X;
            public int Y;
            public Bitmap Result;
            bool _disposed;

            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }
            ~CutImage()
            {
                Dispose();
            }
            protected virtual void Dispose(bool disposing)
            {
                if (_disposed) return;
                if (disposing)
                {
                    if (ImageA != null)
                        ImageA.Dispose();
                    if (ImageB != null)
                        ImageB.Dispose();
                }
                _disposed = true;
            }
        }
        public class Defect
        {
            public int Area;
            public int PosX;
            public int PosY;
            public int Width;
            public int Height;
        }

        // 定義 JSON 資料結構
        public class Shape
        {
            public string label { get; set; }
            public double[][] points { get; set; }
            public string shape_type { get; set; }
        }

        public class JsonData
        {
            public string version { get; set; }
            public Shape[] shapes { get; set; }
            public string imagePath { get; set; }
            public int imageHeight { get; set; }
            public int imageWidth { get; set; }
        }
        #endregion


        #region Main Function
        public bool DoLoadAIModel()
        {
            try
            {
                if (File.Exists(Config.AIModelPath) != true)
                {
                    throw new Exception("AI model file not exist.");
                }

                Core = new ChangeDetectionModel(Config.AIModelPath, Config.UseGPU);

                if (Core.InputShape[2] != Core.InputShape[3])
                {
                    throw new Exception("The width and height should be the same in model.");
                }

                if(Core.InputShape[2] != Config.AIInputSize || Core.InputShape[3] != Config.AIInputSize)
                {
                    throw new Exception("The model input is differnce than setting input size. Model size is " +
                        Core.InputShape[2] + " x " + Core.InputShape[3] + ", but setting is " + Config.AIInputSize + " x "+ Config.AIInputSize);
                }

                //Do first inference to ensure execution time is stable
                Random random = new Random();
                float[] array = new float[Core.InputShape[1] * Core.InputShape[2] * Core.InputShape[3] * Config.BatchSize];

                Parallel.For(0, array.Length, j =>
                {
                    array[j] = (float)random.NextDouble();
                });
                Core.BatchInference(array, array, Config.BatchSize);

                return true;
            }
            catch(Exception ex)
            {
                throw new Exception("Load AI Model Fail : " + ex.Message);
            }
        }

        public bool DoAutoAlignment(Mat SrcImage, Mat CprImage, ref Mat AlignSrcImage, ref Mat AlignCprImage, ref PatternMatchResult AlignResult)
        {
            List<Rect> allPatternRegion = new List<Rect>();
            Rect patternCropRoi = new Rect();
            List<PatternMatchResult> resultList = new List<PatternMatchResult>();
            Mat rotateCprImage = new Mat();
            Mat copySrcImage = new Mat(); 
            Mat copyCprImage = new Mat();
            Mat patternImage = new Mat();
            int patternCenterX = 0;
            int patternCenterY = 0;

            try
            {
                copySrcImage = SrcImage.Clone();
                copyCprImage = CprImage.Clone();

                allPatternRegion = GetAlignRegion(copySrcImage.Width, copySrcImage.Height);

                for (int i = 0; i < allPatternRegion.Count(); i++)
                {
                    Rect alignSrcCropRegion = new Rect(Config.Buffer, 
                                                       Config.Buffer, 
                                                       SrcImage.Width - Config.Buffer * 2,
                                                       SrcImage.Height - Config.Buffer * 2);

                    Rect patternSrcCropRegion = allPatternRegion[i];


                    AlignSrcImage = CropImage(copySrcImage, alignSrcCropRegion);
                    patternImage = CropImage(copySrcImage, patternSrcCropRegion);

                    if (Config.ScaleMode)
                    {
                        if (FastPatternMatchScale(patternImage, copyCprImage, Config.AlignScale, ref resultList))
                        {
                            patternCropRoi = allPatternRegion[i];
                            patternCenterX = Convert.ToInt32(patternCropRoi.X + patternCropRoi.Width / 2);
                            patternCenterY = Convert.ToInt32(patternCropRoi.Y + patternCropRoi.Height / 2);
                        }
                    }
                    else
                    {
                        if (FastPatternMatch(patternImage, copyCprImage, ref resultList))
                        {
                            patternCropRoi = allPatternRegion[i];
                            patternCenterX = Convert.ToInt32(patternCropRoi.X + patternCropRoi.Width / 2);
                            patternCenterY = Convert.ToInt32(patternCropRoi.Y + patternCropRoi.Height / 2);
                        }
                    }


                    if (resultList.Count != 1)
                    {
                        if (i == allPatternRegion.Count() - 1)
                        {
                            throw new Exception("Align Fail. No match any region.");
                        }
                        else continue;
                    }

                    AlignResult = resultList[0];
                    int matchCenterX = Convert.ToInt32(resultList[0].ptCenter.X);
                    int matchCenterY = Convert.ToInt32(resultList[0].ptCenter.Y);
                    double matchRotate = Math.Round(resultList[0].Angle, 3);

                    //save aligned cpr image
                    if (Config.ScaleMode)
                    {
                        rotateCprImage = RotateImageScale(copyCprImage, matchCenterX, matchCenterY, -matchRotate, Config.AlignScale);
                    }
                    else
                    {
                        rotateCprImage = RotateImage(copyCprImage, matchCenterX, matchCenterY, -matchRotate);
                    }

                    Rect cprCrop = new Rect(Config.Buffer + matchCenterX - patternCenterX,
                                            Config.Buffer + matchCenterY - patternCenterY,
                                            CprImage.Width - Config.Buffer * 2,
                                            CprImage.Height - Config.Buffer * 2);

                    AlignCprImage = CropImage(rotateCprImage, cprCrop);

                    if (AlignCprImage == null)
                    {
                        if (i == allPatternRegion.Count() - 1)
                        {
                            throw new Exception("Align Fail. No match any region.");
                        }
                        else continue;
                    }
                    break;
                }
                return true;
            }
            catch (Exception ex)
            {
                throw new Exception("Alignment Fail : " + ex.Message);
            }
            finally
            {
                rotateCprImage?.Dispose();
                resultList?.Clear();
                copySrcImage.Dispose();
                copyCprImage.Dispose();
                patternImage.Dispose();
            }
        }

        public bool DoInspection(Mat alignSrcImage, Mat alignCprImage, ref Bitmap resultMask, ref Mat processSrcImage, ref Mat processCprImage)
        {
            try
            {
                if (Core == null)
                {
                    throw new Exception("No change detection model.");
                }
                processSrcImage = alignSrcImage.Clone();
                processCprImage = alignCprImage.Clone();

                //縮小輸入影像
                if (Config.ResizeMode)
                {
                    OpenCvSharp.Size scaleSize = new OpenCvSharp.Size(896, 896);
                    Cv2.Resize(processSrcImage, processSrcImage, scaleSize);
                    Cv2.Resize(processCprImage, processCprImage, scaleSize);
                }
                //影像強化
                if (Config.EnhancementMode)
                {
                    Cv2.ConvertScaleAbs(processSrcImage, processSrcImage, Config.Contrast, Config.Brightness);
                    Cv2.ConvertScaleAbs(processCprImage, processCprImage, Config.Contrast, Config.Brightness);

                    processSrcImage = GammaCorrection(processSrcImage, Config.GammaCorrection);
                    processCprImage = GammaCorrection(processCprImage, Config.GammaCorrection);
                }
                //亮度平衡
                if (Config.BalanceMode)
                {
                    processSrcImage = BrightBalance(processSrcImage);
                    processCprImage = BrightBalance(processCprImage);
                }

                List<CutImage> cutImages = ImageCutting(processSrcImage, processCprImage, Core.InputShape[2], Core.InputShape[3], Config.OverlapRate);
                ChangeDetection(Core, cutImages, Config.BatchSize, Config.CDThreshold);
                resultMask = ImageStitching(cutImages, processSrcImage.Width, processSrcImage.Height);

                return true;
            }
            catch (Exception ex)
            {
                throw new Exception("Inspection Fail : " + ex.Message);
            }
            finally
            {
            }
        }

        public bool DoOutputResult(Mat srcImage, Bitmap resultMask, string location, ref Bitmap resultOnSrc, ref List<Defect> defectList)
        {
            Mat copySrcImage = new Mat();
            Bitmap copyResultMask = null;
            Mat bypassPaddingMaskResult = new Mat();
            Mat paddingMaskResult = new Mat();
            Mat bypassMask = new Mat();
            try
            {
                copySrcImage = srcImage.Clone();
                copyResultMask = (Bitmap)resultMask.Clone();
                paddingMaskResult = PaddingMask(copyResultMask, copySrcImage);
                (bypassPaddingMaskResult, bypassMask) = MaskBypassRegion(paddingMaskResult, location);
                resultOnSrc = DrawMaskResultToSrc(bypassPaddingMaskResult, copySrcImage, bypassMask);
                defectList = ShowDefect(bypassPaddingMaskResult);
                return true;
            }
            catch (Exception ex)
            {
                throw new Exception("Output Result Fail : " + ex.Message);
            }
            finally
            {
                copySrcImage?.Dispose();
                copyResultMask?.Dispose();
                bypassPaddingMaskResult?.Dispose();
                paddingMaskResult?.Dispose();
                bypassMask?.Dispose();
            }
        }
        
        public bool DoDisposeAIModel()
        {
            try
            {
                Core?.Dispose();
                return true;
            }
            catch(Exception ex)
            {
                throw new Exception("Dispose AI Model fail." + ex.Message);
            }
        }
        #endregion


        #region Algorithm
        public bool FastPatternMatch(Mat patternImage, Mat compareImage,ref List<PatternMatchResult> resultList)
        {
            PatternMatchFinder finder = new PatternMatchFinder();
            Mat grayImageP = new Mat();
            Mat grayImageC = new Mat();

            try
            {
                Cv2.CvtColor(patternImage, grayImageP, ColorConversionCodes.BGR2GRAY);
                byte[] data1 = MatToByteArray(grayImageP);

                Cv2.CvtColor(compareImage, grayImageC, ColorConversionCodes.BGR2GRAY);
                byte[] data2 = MatToByteArray(grayImageC);

                MatchImage ptnImage = new MatchImage(data1, grayImageP.Channels(), grayImageP.Width, grayImageP.Height);
                MatchImage cprImage = new MatchImage(data2, grayImageC.Channels(), grayImageC.Width, grayImageC.Height);

                finder.MaxPos = Config.MaxPos;
                finder.Score = Config.MatchMinScore;
                finder.ToleranceAngle = Config.ToleranceAngle;
                finder.MinReduceArea = Config.MinReduceArea;
                finder.MaxOverlap = Config.MaxOverlap;
                finder.LearnPattern(ptnImage);
                finder.Match(cprImage);
                resultList = finder.GetResult();

                ptnImage.Dispose();
                cprImage.Dispose();

                if (resultList.Count == 1)
                    return true;
                else
                    return false;
            }
            catch (Exception ex)
            {
                throw new Exception("Fast Pattern Match fail" + ex.Message);
            }
            finally
            {
                finder?.Dispose();
                grayImageP?.Dispose();
                grayImageC?.Dispose();
            }
        }

        public bool FastPatternMatchScale(Mat patternImage, Mat compareImage, double scale, ref List<PatternMatchResult> resultList)
        {
            PatternMatchFinder finder = new PatternMatchFinder();
            Mat grayImageP = new Mat();
            Mat grayImageC = new Mat();

            try
            {
                // 轉換為灰度圖
                Cv2.CvtColor(patternImage, grayImageP, ColorConversionCodes.BGR2GRAY);
                Cv2.CvtColor(compareImage, grayImageC, ColorConversionCodes.BGR2GRAY);

                // 縮小圖像
                OpenCvSharp.Size smallSizeC = new OpenCvSharp.Size(compareImage.Width * scale, compareImage.Height * scale);
                OpenCvSharp.Size smallSizeP = new OpenCvSharp.Size(patternImage.Width * scale, patternImage.Height * scale);

                Mat smallPatternImage = new Mat();
                Mat smallCompareImage = new Mat();
                Cv2.Resize(grayImageP, smallPatternImage, smallSizeP, 0, 0, InterpolationFlags.Linear);
                Cv2.Resize(grayImageC, smallCompareImage, smallSizeC, 0, 0, InterpolationFlags.Linear);

                byte[] data1 = MatToByteArray(smallPatternImage);
                byte[] data2 = MatToByteArray(smallCompareImage);

                MatchImage ptnImage = new MatchImage(data1, smallPatternImage.Channels(), smallPatternImage.Width, smallPatternImage.Height);
                MatchImage cprImage = new MatchImage(data2, smallCompareImage.Channels(), smallCompareImage.Width, smallCompareImage.Height);

                finder.MaxPos = Config.MaxPos;
                finder.Score = Config.MatchMinScore;
                finder.ToleranceAngle = Config.ToleranceAngle;
                finder.MinReduceArea = Config.MinReduceArea;
                finder.MaxOverlap = Config.MaxOverlap;

                finder.LearnPattern(ptnImage);
                finder.Match(cprImage);

                List<PatternMatchResult> scaledResultList = finder.GetResult();

                // 將結果座標映射回原始尺寸
                resultList = ScaleResults(scaledResultList, scale);

                ptnImage.Dispose();
                cprImage.Dispose();

                return resultList.Count == 1;
            }
            catch (Exception ex)
            {
                throw new Exception("Fast Pattern Match fail: " + ex.Message);
            }
            finally
            {
                finder?.Dispose();
                grayImageP?.Dispose();
                grayImageC?.Dispose();
            }
        }

        private void ChangeDetection(ChangeDetectionModel core, List<CutImage> cutimages, int batchSize, float threshold)
        {
            try
            {
                Parallel.For(0, cutimages.Count(), j =>
                {
                    cutimages[j].floatImageA = MatToFloatArray(cutimages[j].ImageA);
                    cutimages[j].floatImageB = MatToFloatArray(cutimages[j].ImageB);
                });

                //batch inference
                int runBatchSize;
                int runTime = cutimages.Count() / batchSize;
                if (cutimages.Count() % batchSize != 0) runTime++;

                for (int epoch = 0; epoch < runTime; epoch++)
                {
                    if (epoch == runTime - 1)
                    {
                        runBatchSize = cutimages.Count() % batchSize;
                        if (runBatchSize == 0) runBatchSize = batchSize;
                    }
                    else
                    {
                        runBatchSize = batchSize;
                    }

                    float[] batchDataA = new float[cutimages[0].floatImageA.Length * runBatchSize];
                    float[] batchDataB = new float[cutimages[0].floatImageB.Length * runBatchSize];

                    Parallel.For(epoch * batchSize, epoch * batchSize + runBatchSize, i =>
                    {
                        Array.Copy(cutimages[i].floatImageA, 0, batchDataA, cutimages[i].floatImageA.Length * (i % batchSize), cutimages[i].floatImageA.Length);
                        Array.Copy(cutimages[i].floatImageB, 0, batchDataB, cutimages[i].floatImageB.Length * (i % batchSize), cutimages[i].floatImageB.Length);
                    });

                    core.BatchInference(batchDataB, batchDataA, runBatchSize);
                    List<BrandyImage> outputs = core.GetOutputImage(threshold, runBatchSize);

                    Parallel.For(0, outputs.Count(), i =>
                    //for (int i = 0; i < outputs.Count(); i++)
                    {
                        cutimages[epoch * batchSize + i].Result = outputs[i].Bitmap;
                    });
                    //}

                    outputs.ForEach(x => x?.Dispose());
                    outputs.Clear();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Inference error : " + ex);
            }
        }
        #endregion


        #region Tools
        public List<Rect> GetAlignRegion(int imageWidth, int imageHeight)
        {
            List<Rect> allRegion = new List<Rect>();
            Rect firstRegion = new Rect(Config.Buffer, Config.Buffer, imageWidth - Config.Buffer * 2, imageHeight - Config.Buffer * 2);
            allRegion.Add(firstRegion);
            int interval = Convert.ToInt32(Config.AlignSearchRegionSize * Config.AlignSearchOverlap);

            for (int i = Config.Buffer; i+ Config.AlignSearchRegionSize <= imageWidth - Config.Buffer; i+= interval)
            {
                for (int j = Config.Buffer; j+ Config.AlignSearchRegionSize <= imageHeight - Config.Buffer; j+= interval)
                {
                    Rect region = new Rect(i,j, Config.AlignSearchRegionSize, Config.AlignSearchRegionSize);
                    allRegion.Add(region);
                }
            }
            return allRegion;
        }

        public Mat PaddingMask(Bitmap mask, Mat srcImage)
        {
            Mat result = new Mat(srcImage.Width, srcImage.Height, srcImage.Type(), new Scalar(0, 0, 0));
            Mat mMask = BitmapConverter.ToMat(mask);
            if (Config.ResizeMode)
            {
                OpenCvSharp.Size size = new OpenCvSharp.Size(1892, 1892);
                Cv2.Resize(mMask, mMask, size);
            }
            int padding = (srcImage.Width - mMask.Width) / 2;
            mMask.CopyTo(result[new Rect(padding, padding, mMask.Cols, mMask.Rows)]);
            return result;
        }

        public Bitmap DrawMaskResultToSrc(Mat resultMask, Mat srcImage, Mat bypassMask)
        {
            Mat maskToSrcResult = new Mat();
            try
            {
                Cv2.Split(srcImage, out Mat[] imageChannels);
                Cv2.Split(resultMask, out Mat[] maskChannels);
                Cv2.AddWeighted(imageChannels[0], 0.9 , maskChannels[0], 0, 0, imageChannels[0]);//B
                Cv2.AddWeighted(imageChannels[1], 0.9, bypassMask, 0.2, 0, imageChannels[1]);//G
                Cv2.AddWeighted(imageChannels[2], 0.9, maskChannels[2], 0.6, 0, imageChannels[2]);//R
                Cv2.Merge(imageChannels, maskToSrcResult);
                return BitmapConverter.ToBitmap(maskToSrcResult);
            }
            catch (Exception ex)
            {
                throw new Exception("Draw MaskResult To Src Image Fail: " + ex.Message);
            }
        }

        public List<Defect> ShowDefect(Mat mMask)
        {
            List<Defect> defectList = new List<Defect>();
            Mat binaryImage = new Mat(mMask.Width, mMask.Height, MatType.CV_8UC1);
            Cv2.CvtColor(mMask, binaryImage, ColorConversionCodes.BGR2GRAY);
            Cv2.FindContours(binaryImage,
                             out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy,
                             RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            for (int i = 0; i < contours.Length; i++)
            {
                Defect defect = new Defect();
                double area = Cv2.ContourArea(contours[i]);
                if (area == 0)
                {
                    defect.PosX = contours[i][0].X;
                    defect.PosY = contours[i][0].Y;
                    defect.Area = 0;
                    defect.Width = 1;
                    defect.Height = 1;
                }
                else
                {
                    Moments moments = Cv2.Moments(contours[i]);
                    double centerX = moments.M10 / moments.M00;
                    double centerY = moments.M01 / moments.M00;

                    Rect boundingRect = Cv2.BoundingRect(contours[i]);

                    defect.PosX = Convert.ToInt32(centerX);
                    defect.PosY = Convert.ToInt32(centerY);
                    defect.Area = Convert.ToInt32(area);
                    defect.Width = Convert.ToInt32(boundingRect.Width);
                    defect.Height = Convert.ToInt32(boundingRect.Height);
                }

                defectList.Add(defect);
            }
            return defectList;
        }

        private Mat CropImage(Mat sourceImage, Rect cropArea)
        {
            // Create a Region of Interest (ROI) based on the crop area
            if (cropArea.X < 0 ||
                cropArea.Y < 0 ||
                cropArea.X + cropArea.Width > sourceImage.Width ||
                cropArea.Y + cropArea.Height > sourceImage.Height)
                return null;

            Mat croppedImage = new Mat(sourceImage, cropArea);
            return croppedImage;
        }

        private List<CutImage> ImageCutting(Mat imageA, Mat imageB, int cutWidth, int cutHeight, double overlapRate)
        {
            if (imageA.Width != imageB.Width || imageA.Height != imageB.Height)
            {
                throw new Exception("ImageCutting Fail : input image A and B size not match.!");
            }

            List<CutImage> cutImages = new List<CutImage>();
            int strideX = Convert.ToInt32(cutWidth * (1.0 - overlapRate));
            int strideY = Convert.ToInt32(cutHeight * (1.0 - overlapRate));

            for (int y = 0; y <= imageA.Height - cutHeight; y += strideY)
            {
                for (int x = 0; x <= imageA.Width - cutWidth; x += strideX)
                {
                    // 切割小图
                    CutImage cutImage = new CutImage();
                    Rect roi = new Rect(x, y, cutWidth, cutHeight);
                    cutImage.X = x;
                    cutImage.Y = y;
                    cutImage.ImageA = new Mat(imageA, roi);
                    cutImage.ImageB = new Mat(imageB, roi);
                    cutImages.Add(cutImage);
                }
            }
            return cutImages;
        }

        private Bitmap ImageStitching(List<CutImage> cutImages, int mapWidth, int mapHeight)
        {
            Bitmap stitchedBitmap = new Bitmap(mapWidth, mapHeight, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(stitchedBitmap))
            {
                foreach (CutImage cutimage in cutImages)
                {
                    g.DrawImage(cutimage.Result, cutimage.X, cutimage.Y);
                }
            }
            return stitchedBitmap;
        }

        private Mat RotateImage(Mat image, int rotateX, int rotateY, double angle) 
        {
            // 設定旋轉中心點
            Point2f center = new Point2f(rotateY, rotateX);

            // 計算旋轉矩陣
            Mat rotationMatrix = Cv2.GetRotationMatrix2D(center, angle, 1.0);

            // 進行圖片旋轉
            Mat rotatedImage = new Mat();
            Cv2.WarpAffine(image, rotatedImage, rotationMatrix, image.Size());
            return rotatedImage;
        }

        private Mat RotateImageScale (Mat image, int rotateX, int rotateY, double angle, double scale)
        {
            // 原始圖像尺寸
            OpenCvSharp.Size originalSize = image.Size();

            // 縮小圖像
            OpenCvSharp.Size smallSize = new OpenCvSharp.Size(image.Width * scale, image.Height * scale);
            Mat smallImage = new Mat();
            Cv2.Resize(image, smallImage, smallSize, 0, 0, InterpolationFlags.Linear);

            // 調整旋轉中心點
            Point2f center = new Point2f((float)(rotateY * scale), (float)(rotateX * scale));

            // 計算旋轉矩陣
            Mat rotationMatrix = Cv2.GetRotationMatrix2D(center, angle, 1.0);

            // 進行縮小後圖片的旋轉
            Mat smallRotatedImage = new Mat();
            Cv2.WarpAffine(smallImage, smallRotatedImage, rotationMatrix, smallSize);

            // 將旋轉後的圖片放大回原始尺寸
            Mat rotatedImage = new Mat();
            Cv2.Resize(smallRotatedImage, rotatedImage, originalSize, 0, 0, InterpolationFlags.Linear);

            return rotatedImage;
        }

        private static float[] MatToFloatArray(Mat image)
        {
            Mat[] channels = image.Split();
            float[] input = new float[image.Rows * image.Cols * channels.Count()];
            Parallel.For(0, channels.Count(), i =>
            {
                byte[] byteArray = new byte[channels[i].Width * channels[i].Height];
                Marshal.Copy(channels[i].Data, byteArray, 0, byteArray.Length);
                float[] floatMat = byteArray.Select(x => (float)x).ToArray();
                Array.Copy(floatMat, 0, input, channels[i].Width * channels[i].Height * i, channels[i].Width * channels[i].Height);
            });
            return input;
        }

        private byte[] MatToByteArray(Mat image)
        {
            byte[] data = new byte[image.Rows * image.Cols * image.Channels()];
            Marshal.Copy(image.Data, data, 0, image.Rows * image.Cols * image.Channels());
            return data;
        }

        private List<PatternMatchResult> ScaleResults(List<PatternMatchResult> scaledResults, double scale)
        {
            List<PatternMatchResult> originalResults = new List<PatternMatchResult>();

            foreach (var result in scaledResults)
            {
                PatternMatchResult originalResult = new PatternMatchResult();

                // 縮放回原始座標
                originalResult.ptLT = new PointF((float)(result.ptLT.X / scale), (float)(result.ptLT.Y / scale));
                originalResult.ptRT = new PointF((float)(result.ptRT.X / scale), (float)(result.ptRT.Y / scale));
                originalResult.ptRB = new PointF((float)(result.ptRB.X / scale), (float)(result.ptRB.Y / scale));
                originalResult.ptLB = new PointF((float)(result.ptLB.X / scale), (float)(result.ptLB.Y / scale));
                originalResult.ptCenter = new PointF((float)(result.ptCenter.X / scale), (float)(result.ptCenter.Y / scale));

                // 角度和分數保持不變
                originalResult.MatchedAngle = result.MatchedAngle;
                originalResult.MatchScore = result.MatchScore;

                originalResults.Add(originalResult);
            }

            return originalResults;
        }

        public (Mat,Mat) MaskBypassRegion(Mat maskResultImage, string location)
        {
            try
            {
                // 創建遮罩
                Mat bypassMask = new Mat(maskResultImage.Size(), MatType.CV_8UC1, Scalar.All(0));

                if (Config.BypassMode == false)
                {
                    return (maskResultImage, bypassMask);
                }

                // 驗證輸入
                if (maskResultImage == null || maskResultImage.Empty())
                {
                    throw new ArgumentException("輸入影像無效");
                }
                if (string.IsNullOrEmpty(Config.BypassJsonFolder) || !Directory.Exists(Config.BypassJsonFolder))
                {
                    throw new ArgumentException("bypass JSON 資料夾無效");
                }

                // 初始化 JSON 檔案路徑
                string jsonPath = null;

                // 判斷 location 是否有效
                if (!string.IsNullOrEmpty(location))
                {
                    // 將 location 加上 .json 組成檔案路徑
                    jsonPath = Path.Combine(Config.BypassJsonFolder, $"{location}.json");
                }

                // 如果 JSON 檔案不存在或 location 為 null
                if (string.IsNullOrEmpty(jsonPath) || !File.Exists(jsonPath))
                {
                    return (maskResultImage, bypassMask);
                }

                // 讀取並解析 JSON
                string jsonString = File.ReadAllText(jsonPath);
                JsonData data = JsonSerializer.Deserialize<JsonData>(jsonString);

                // 驗證影像尺寸
                if (maskResultImage.Height != data.imageHeight || maskResultImage.Width != data.imageWidth)
                {
                    throw new ArgumentException("影像尺寸與 JSON 描述不符");
                }

                // 處理所有 bypass 區域
                if (data.shapes != null && data.shapes.Length > 0)
                {
                    foreach (var shape in data.shapes)
                    {
                        if (shape.label.ToLower() == "bypass" && shape.points != null)
                        {
                            if (shape.shape_type.ToLower() == "polygon")
                            {
                                OpenCvSharp.Point[] points = shape.points
                                    .Select(p => new OpenCvSharp.Point((int)p[0], (int)p[1]))
                                    .ToArray();
                                Cv2.FillPoly(bypassMask, new OpenCvSharp.Point[][] { points }, Scalar.All(255));
                            }

                            if (shape.shape_type.ToLower() == "rectangle")
                            {
                                OpenCvSharp.Point pt1 = new OpenCvSharp.Point((int)shape.points[0][0], (int)shape.points[0][1]);
                                OpenCvSharp.Point pt2 = new OpenCvSharp.Point((int)shape.points[1][0], (int)shape.points[1][1]);
                                Cv2.Rectangle(bypassMask, pt1, pt2, Scalar.All(255), -1);
                            }

                            if (shape.shape_type.ToLower() == "circle")
                            {
                                OpenCvSharp.Point center = new OpenCvSharp.Point((int)shape.points[0][0], (int)shape.points[0][1]);
                                int radius = (int)Math.Sqrt(Math.Pow(shape.points[0][0] - shape.points[1][0], 2) + Math.Pow(shape.points[0][1] - shape.points[1][1], 2));
                                Cv2.Circle(bypassMask, center, radius, Scalar.All(255), -1);
                            }
                        }
                    }
                }

                // 應用遮罩到原圖
                Mat maskOnSrc = maskResultImage.Clone();
                maskOnSrc.SetTo(Scalar.All(0), bypassMask);

                // 返回結果
                return (maskOnSrc, bypassMask);
            }
            catch (Exception ex)
            {
                throw new Exception("Output Result Fail: " + ex.Message);
            }
        }

        private Mat GammaCorrection(Mat src, double gamma)
        {
            Mat lut = new Mat(1, 256, MatType.CV_8UC1);

            for (int i = 0; i < 256; i++)
            {
                lut.Set<byte>(0, i, Convert.ToByte(Math.Pow(i / 255.0D, gamma) * 255.0D));
            }

            Mat dst = new Mat();
            Cv2.LUT(src, lut, dst);
            return dst;
        }

        private Mat BrightBalance(Mat srcImage)
        {
            Mat resultImage = new Mat();
            Mat hsvImage = new Mat();
            Mat[] hsvChannels = new Mat[3];
            Mat vChannel = new Mat();
            Mat blurredV = new Mat();
            Mat correctedV = new Mat();

            try
            {
                if (srcImage == null || srcImage.Empty())
                {
                    throw new ArgumentException("輸入影像不能為空");
                }

                // 確保模糊核心大小為奇數
                int blurKernelSize = Config.BlurKernelSize;
                if (Config.BlurKernelSize % 2 == 0)
                {
                    blurKernelSize = Config.BlurKernelSize + 1;
                }

                // 將影像轉換為HSV色彩空間
                Cv2.CvtColor(srcImage, hsvImage, ColorConversionCodes.BGR2HSV);

                // 分離HSV通道
                Cv2.Split(hsvImage, out hsvChannels);

                // 取得V通道(亮度通道)並轉換為float32
                hsvChannels[2].ConvertTo(vChannel, MatType.CV_32F);

                // 對V通道進行高斯模糊以估計亮度分布
                OpenCvSharp.Size kernelSize = new OpenCvSharp.Size(blurKernelSize, blurKernelSize);
                Cv2.GaussianBlur(vChannel, blurredV, kernelSize, 0);

                // 進行亮度補償：corrected_v = (v / (blurred + 1e-6)) * compensate_strength
                // 防止除以零，加上一個很小的值
                Cv2.Add(blurredV, new Scalar(1e-6), blurredV);
                Cv2.Divide(vChannel, blurredV, correctedV);
                Cv2.Multiply(correctedV, new Scalar(Config.AddBright), correctedV);

                // 限制值的範圍在0-255之間並轉回uint8
                correctedV = correctedV.Clone();
                correctedV.ConvertTo(hsvChannels[2], MatType.CV_8U);

                // 合併HSV通道
                Cv2.Merge(hsvChannels, hsvImage);

                // 轉換回BGR色彩空間
                Cv2.CvtColor(hsvImage, resultImage, ColorConversionCodes.HSV2BGR);

                return resultImage.Clone();
            }
            catch (Exception ex)
            {
                throw new Exception($"亮度平衡處理失敗: {ex.Message}");
            }
            finally
            {
                // 釋放資源
                hsvImage?.Dispose();
                vChannel?.Dispose();
                blurredV?.Dispose();
                correctedV?.Dispose();

                if (hsvChannels != null)
                {
                    foreach (var channel in hsvChannels)
                    {
                        channel?.Dispose();
                    }
                }
            }
        }
    }
    #endregion
}
