using AIF_2D;
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
            List<Rect> allAignRegion = new List<Rect>();
            Rect patternCropRoi = new Rect();
            List<PatternMatchResult> resultList = new List<PatternMatchResult>();
            Mat rotateCprImage = new Mat();
            Mat copySrcImage = new Mat(); 
            Mat copyCprImage = new Mat();
            int patternCenterX = 0;
            int patternCenterY = 0;

            try
            {
                copySrcImage = SrcImage.Clone();
                copyCprImage = CprImage.Clone();

                allAignRegion = GetAlignRegion(copySrcImage.Width, copySrcImage.Height);

                for (int i = 0; i < allAignRegion.Count(); i++)
                {
                    Rect alignSrcCropRegion = new Rect(Config.Buffer, 
                                                       Config.Buffer, 
                                                       SrcImage.Width - Config.Buffer * 2,
                                                       SrcImage.Height - Config.Buffer * 2);

                    AlignSrcImage = CropImage(copySrcImage, alignSrcCropRegion);

                    if (FastPatternMatch(AlignSrcImage, copyCprImage, ref resultList))
                    {
                        patternCropRoi = allAignRegion[i];
                        patternCenterX = Convert.ToInt32(patternCropRoi.X + patternCropRoi.Width / 2);
                        patternCenterY = Convert.ToInt32(patternCropRoi.Y + patternCropRoi.Height / 2);
                    }

                    if (resultList.Count != 1)
                    {
                        throw new Exception("Fast pattern match fail.");
                    }

                    AlignResult = resultList[0];
                    int matchCenterX = Convert.ToInt32(resultList[0].ptCenter.X);
                    int matchCenterY = Convert.ToInt32(resultList[0].ptCenter.Y);
                    double matchRotate = Math.Round(resultList[0].Angle, 3);

                    //save aligned cpr image
                    rotateCprImage = RotateImage(copyCprImage, matchCenterX, matchCenterY, -matchRotate);
                    Rect cprCrop = new Rect(Config.Buffer + matchCenterX - patternCenterX,
                                            Config.Buffer + matchCenterY - patternCenterY,
                                            CprImage.Width - Config.Buffer * 2,
                                            CprImage.Height - Config.Buffer * 2);

                    AlignCprImage = CropImage(rotateCprImage, cprCrop);

                    if (AlignCprImage == null)
                    {
                        if (i == allAignRegion.Count() - 1)
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
            }
        }

        public bool DoInspection(Mat alignSrcImage, Mat alignCprImage, ref Bitmap resultMask)
        {
            Mat copyAlignSrcImage = new Mat();
            Mat copyAlignCprImage = new Mat();

            try
            {
                if (Core == null)
                {
                    throw new Exception("No change detection model.");
                }
                copyAlignSrcImage = alignSrcImage.Clone();
                copyAlignCprImage = alignCprImage.Clone();
                List<CutImage> cutImages = ImageCutting(copyAlignSrcImage, copyAlignCprImage, Core.InputShape[2], Core.InputShape[3], Config.OverlapRate);
                ChangeDetection(Core, cutImages, Config.BatchSize, Config.CDThreshold);
                resultMask = ImageStitching(cutImages, copyAlignSrcImage.Width, copyAlignSrcImage.Height);
                return true;
            }
            catch(Exception ex)
            {
                throw new Exception("Inspection Fail : " + ex.Message);
            }
            finally
            {
                copyAlignSrcImage.Dispose();
                copyAlignCprImage.Dispose();
            }
        }

        public bool DoOutputResult(Mat srcImage, Bitmap resultMask, ref Bitmap resultImage, ref List<Defect> defectList)
        {
            Mat copySrcImage = new Mat();
            Bitmap copyResultMask = null;
            try
            {
                copySrcImage = srcImage.Clone();
                copyResultMask = (Bitmap)resultMask.Clone();
                resultImage = DrawResult(copyResultMask, copySrcImage);
                defectList = ShowDefect(copyResultMask);
                return true;
            }
            catch (Exception ex)
            {
                throw new Exception("Output Result Fail : " + ex.Message);
            }
            finally
            {
                copySrcImage.Dispose();
                copyResultMask.Dispose();
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
                finder.Dispose();
                grayImageP.Dispose();
                grayImageC.Dispose();
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

        public Bitmap DrawResult(Bitmap mask, Mat srcImage)
        {
            Mat result = new Mat(srcImage.Width, srcImage.Height, srcImage.Type(), new Scalar(0, 0, 0));
            Mat mMask = BitmapConverter.ToMat(mask);
            int padding = (srcImage.Width - mask.Width) / 2;
            mMask.CopyTo(result[new Rect(padding, padding, mMask.Cols, mMask.Rows)]);

            Cv2.Split(srcImage, out Mat[] imageChannels);
            Cv2.Split(result, out Mat[] maskChannels);
            Cv2.AddWeighted(imageChannels[0], 0.5, maskChannels[0], 0, 0, imageChannels[0]);
            Cv2.AddWeighted(imageChannels[1], 0.5, maskChannels[1], 0, 0, imageChannels[1]);
            Cv2.AddWeighted(imageChannels[2], 0.5, maskChannels[2], 0.5, 0, imageChannels[2]);
            Cv2.Merge(imageChannels, result);
            return BitmapConverter.ToBitmap(result);
        }

        public List<Defect> ShowDefect(Bitmap mask)
        {
            List<Defect> defectList = new List<Defect>();
            Mat mMask = BitmapConverter.ToMat(mask);
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
        #endregion
    }
}
