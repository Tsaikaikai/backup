using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace SAM2OnnxInference
{
    public class SAM2Predictor : IDisposable
    {
        private InferenceSession encoderSession;
        private InferenceSession decoderSession;
        private const int ImageSize = 1024;

        // ImageNet 標準化參數
        private readonly float[] mean = { 0.485f, 0.456f, 0.406f };
        private readonly float[] std = { 0.229f, 0.224f, 0.225f };

        // 影片追蹤狀態
        public class VideoInferenceState
        {
            public List<string> FramePaths { get; set; }
            public Dictionary<int, (DenseTensor<float> imageEmbed, DenseTensor<float> highResFeats0, DenseTensor<float> highResFeats1)> FrameEmbeddings { get; set; }
            public Dictionary<int, Dictionary<int, ObjectAnnotation>> Annotations { get; set; } // frameIdx -> objId -> annotation
            public int VideoWidth { get; set; }
            public int VideoHeight { get; set; }
        }

        public class ObjectAnnotation
        {
            public int ObjectId { get; set; }
            public Point[] Points { get; set; }
            public int[] Labels { get; set; }
            public float[,] LastMask { get; set; }
        }

        /*public SAM2Predictor(string encoderPath, string decoderPath, bool useFP16)
        {
            try
            {
                var options = new SessionOptions();
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                options.AppendExecutionProvider_CUDA(0);

                encoderSession = new InferenceSession(encoderPath, options);
                decoderSession = new InferenceSession(decoderPath, options);

                Console.WriteLine("SAM2 - 模型載入成功!");
                PrintModelInfo();
            }
            catch (Exception ex) 
            { 
                throw new Exception(ex.Message);
            }
        }*/

        public SAM2Predictor(string encoderPath, string decoderPath, bool enableOptimization = false)
        {
            var options = new SessionOptions();

            // 啟用最高級別的圖優化
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            if (enableOptimization)
            {
                // 啟用記憶體優化
                options.EnableMemoryPattern = true;
                options.EnableCpuMemArena = true;

                // 設定執行模式為平行（多執行緒）
                options.ExecutionMode = ExecutionMode.ORT_PARALLEL;

                // 設定執行緒數（根據 CPU 核心數調整）
                options.IntraOpNumThreads = Environment.ProcessorCount;
            }

            // 配置 CUDA
            var cudaProviderOptions = new OrtCUDAProviderOptions();
            var cudaOptions = new Dictionary<string, string>
            {
                { "device_id", "0" },
                { "gpu_mem_limit", "4294967296" }, // 4GB
                { "arena_extend_strategy", "kSameAsRequested" },
                { "cudnn_conv_algo_search", "EXHAUSTIVE" }, // 使用最佳 conv 算法
                { "do_copy_in_default_stream", "1" },
                { "cudnn_conv_use_max_workspace", "1" }
            };

            cudaProviderOptions.UpdateOptions(cudaOptions);
            options.AppendExecutionProvider_CUDA(cudaProviderOptions);

            Console.WriteLine("正在載入 SAM2 模型...");
            var startTime = DateTime.Now;

            encoderSession = new InferenceSession(encoderPath, options);
            decoderSession = new InferenceSession(decoderPath, options);

            var loadTime = (DateTime.Now - startTime).TotalSeconds;

            Console.WriteLine($"✓ SAM2 模型載入成功! (耗時: {loadTime:F2}s)");
            Console.WriteLine($"  - 使用 FP32 精度 + CUDA 加速");
            Console.WriteLine($"  - 圖優化等級: ORT_ENABLE_ALL");
            Console.WriteLine($"  - cuDNN 卷積優化: EXHAUSTIVE");

            PrintModelInfo();
        }

        private void PrintModelInfo()
        {
            Console.WriteLine("\n=== Encoder 輸入/輸出 ===");
            foreach (var input in encoderSession.InputMetadata)
            {
                Console.WriteLine($"輸入: {input.Key} - {string.Join("x", input.Value.Dimensions)}");
            }
            foreach (var output in encoderSession.OutputMetadata)
            {
                Console.WriteLine($"輸出: {output.Key} - {string.Join("x", output.Value.Dimensions)}");
            }

            Console.WriteLine("\n=== Decoder 輸入/輸出 ===");
            foreach (var input in decoderSession.InputMetadata)
            {
                Console.WriteLine($"輸入: {input.Key} - {string.Join("x", input.Value.Dimensions)}");
            }
            foreach (var output in decoderSession.OutputMetadata)
            {
                Console.WriteLine($"輸出: {output.Key} - {string.Join("x", output.Value.Dimensions)}");
            }
            Console.WriteLine();
        }

        private DenseTensor<float> PreprocessImage(Mat image, out float scaleX, out float scaleY)
        {
            int originalHeight = image.Height;
            int originalWidth = image.Width;

            scaleX = (float)ImageSize / originalWidth;
            scaleY = (float)ImageSize / originalHeight;

            Mat resized = new Mat();
            Cv2.Resize(image, resized, new Size(ImageSize, ImageSize), 0, 0, InterpolationFlags.Linear);

            Mat rgb = new Mat();
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

            var tensor = new DenseTensor<float>(new[] { 1, 3, ImageSize, ImageSize });

            unsafe
            {
                byte* data = (byte*)rgb.DataPointer;
                int step = (int)rgb.Step();

                for (int y = 0; y < ImageSize; y++)
                {
                    for (int x = 0; x < ImageSize; x++)
                    {
                        int idx = y * step + x * 3;
                        tensor[0, 0, y, x] = (data[idx] / 255f - mean[0]) / std[0];
                        tensor[0, 1, y, x] = (data[idx + 1] / 255f - mean[1]) / std[1];
                        tensor[0, 2, y, x] = (data[idx + 2] / 255f - mean[2]) / std[2];
                    }
                }
            }

            resized.Dispose();
            rgb.Dispose();

            return tensor;
        }

        private (DenseTensor<float> imageEmbed, DenseTensor<float> highResFeats0, DenseTensor<float> highResFeats1)
            EncodeImage(DenseTensor<float> imageTensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", imageTensor)
            };

            var results = encoderSession.Run(inputs);

            var highResFeats0 = (DenseTensor<float>)results.First(x => x.Name == "high_res_feats_0").AsTensor<float>();
            var highResFeats1 = (DenseTensor<float>)results.First(x => x.Name == "high_res_feats_1").AsTensor<float>();
            var imageEmbed = (DenseTensor<float>)results.First(x => x.Name == "image_embed").AsTensor<float>();

            var output0 = new DenseTensor<float>(highResFeats0.Dimensions.ToArray());
            var output1 = new DenseTensor<float>(highResFeats1.Dimensions.ToArray());
            var output2 = new DenseTensor<float>(imageEmbed.Dimensions.ToArray());

            highResFeats0.Buffer.Span.CopyTo(output0.Buffer.Span);
            highResFeats1.Buffer.Span.CopyTo(output1.Buffer.Span);
            imageEmbed.Buffer.Span.CopyTo(output2.Buffer.Span);

            return (output2, output0, output1);
        }

        private (float[,] mask, float iouScore) DecodeMask(
            DenseTensor<float> imageEmbed,
            DenseTensor<float> highResFeats0,
            DenseTensor<float> highResFeats1,
            Point[] points,
            int[] labels,
            int originalWidth,
            int originalHeight)
        {
            int numPoints = points.Length;

            var pointCoords = new DenseTensor<float>(new[] { 1, numPoints, 2 });
            for (int i = 0; i < numPoints; i++)
            {
                pointCoords[0, i, 0] = points[i].X;
                pointCoords[0, i, 1] = points[i].Y;
            }

            var pointLabels = new DenseTensor<float>(new[] { 1, numPoints });
            for (int i = 0; i < numPoints; i++)
            {
                pointLabels[0, i] = labels[i];
            }

            int maskH = imageEmbed.Dimensions[2] * 4;
            int maskW = imageEmbed.Dimensions[3] * 4;
            var maskInput = new DenseTensor<float>(new[] { 1, 1, maskH, maskW });

            var hasMaskInput = new DenseTensor<float>(new[] { 1 });
            hasMaskInput[0] = 0f;

            var imgSize = new DenseTensor<int>(new[] { 2 });
            imgSize[0] = originalHeight;
            imgSize[1] = originalWidth;

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image_embed", imageEmbed),
                NamedOnnxValue.CreateFromTensor("high_res_feats_0", highResFeats0),
                NamedOnnxValue.CreateFromTensor("high_res_feats_1", highResFeats1),
                NamedOnnxValue.CreateFromTensor("point_coords", pointCoords),
                NamedOnnxValue.CreateFromTensor("point_labels", pointLabels),
                NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
                NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInput),
                NamedOnnxValue.CreateFromTensor("orig_im_size", imgSize)
            };

            var results = decoderSession.Run(inputs);

            var masks = results.First(x => x.Name == "masks").AsTensor<float>();
            var iouPred = results.First(x => x.Name == "iou_predictions").AsTensor<float>();

            int height = masks.Dimensions[2];
            int width = masks.Dimensions[3];
            var mask = new float[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    mask[y, x] = masks[0, 0, y, x];
                }
            }

            return (mask, iouPred[0, 0]);
        }

        /// <summary>
        /// 單張圖片預測（原有功能）
        /// </summary>
        public void Predict(Mat srcImage, Point[] points, int[] labels, ref Mat mask, ref float iouScore)
        {
            try
            {
                if (srcImage.Empty())
                {
                    throw new Exception($"無法載入圖片");
                }

                int originalHeight = srcImage.Height;
                int originalWidth = srcImage.Width;
                Console.WriteLine($"原始圖片大小: {originalWidth} x {originalHeight}");

                var imageTensor = PreprocessImage(srcImage, out float scaleX, out float scaleY);
                Console.WriteLine("圖片預處理完成");

                var (imageEmbed, highResFeats0, highResFeats1) = EncodeImage(imageTensor);
                Console.WriteLine($"圖片編碼完成");

                Point[] scaledPoints = new Point[points.Length];
                for (int i = 0; i < points.Length; i++)
                {
                    scaledPoints[i] = new Point(
                        (int)(points[i].X * scaleX),
                        (int)(points[i].Y * scaleY)
                    );
                }

                var (maskArray, iou) = DecodeMask(
                    imageEmbed, highResFeats0, highResFeats1,
                    scaledPoints, labels,
                    originalWidth, originalHeight
                );

                iouScore = iou;
                Console.WriteLine($"遮罩生成完成, IOU 分數: {iou:F4}");

                int maskHeight = maskArray.GetLength(0);
                int maskWidth = maskArray.GetLength(1);
                mask = new Mat(maskHeight, maskWidth, MatType.CV_8UC1);

                unsafe
                {
                    byte* data = (byte*)mask.DataPointer;
                    for (int y = 0; y < maskHeight; y++)
                    {
                        for (int x = 0; x < maskWidth; x++)
                        {
                            data[y * maskWidth + x] = maskArray[y, x] > 0 ? (byte)255 : (byte)0;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"預測失敗: {ex.Message}");
            }
        }

        /// <summary>
        /// 初始化影片追蹤狀態
        /// </summary>
        public VideoInferenceState InitVideoState(string videoDir)
        {
            Console.WriteLine($"初始化影片追蹤狀態: {videoDir}");

            // 掃描所有 JPEG 圖片
            var framePaths = Directory.GetFiles(videoDir)
                .Where(f => f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase) ||
                           f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => int.Parse(Path.GetFileNameWithoutExtension(f)))
                .ToList();

            if (framePaths.Count == 0)
            {
                throw new Exception("找不到任何 JPEG 圖片檔案");
            }

            // 取得影片尺寸
            using (var firstFrame = Cv2.ImRead(framePaths[0]))
            {
                var state = new VideoInferenceState
                {
                    FramePaths = framePaths,
                    FrameEmbeddings = new Dictionary<int, (DenseTensor<float>, DenseTensor<float>, DenseTensor<float>)>(),
                    Annotations = new Dictionary<int, Dictionary<int, ObjectAnnotation>>(),
                    VideoWidth = firstFrame.Width,
                    VideoHeight = firstFrame.Height
                };

                Console.WriteLine($"找到 {framePaths.Count} 個影格, 尺寸: {state.VideoWidth}x{state.VideoHeight}");
                return state;
            }
        }

        /// <summary>
        /// 在指定影格上添加標註點
        /// </summary>
        public (float[,] mask, float iouScore) AddPointsToFrame(
            VideoInferenceState state,
            int frameIdx,
            int objectId,
            Point[] points,
            int[] labels)
        {
            Console.WriteLine($"在影格 {frameIdx} 上為物件 {objectId} 添加標註點");

            // 載入並編碼該影格（如果尚未編碼）
            if (!state.FrameEmbeddings.ContainsKey(frameIdx))
            {
                using (var frame = Cv2.ImRead(state.FramePaths[frameIdx]))
                {
                    var imageTensor = PreprocessImage(frame, out float scaleX, out float scaleY);
                    state.FrameEmbeddings[frameIdx] = EncodeImage(imageTensor);
                    Console.WriteLine($"影格 {frameIdx} 編碼完成");
                }
            }

            var (imageEmbed, highResFeats0, highResFeats1) = state.FrameEmbeddings[frameIdx];

            // 將點座標轉換到 1024x1024 空間
            float scaleX2 = (float)ImageSize / state.VideoWidth;
            float scaleY2 = (float)ImageSize / state.VideoHeight;

            Point[] scaledPoints = new Point[points.Length];
            for (int i = 0; i < points.Length; i++)
            {
                scaledPoints[i] = new Point(
                    (int)(points[i].X * scaleX2),
                    (int)(points[i].Y * scaleY2)
                );
            }

            // 生成遮罩
            var (maskArray, iou) = DecodeMask(
                imageEmbed, highResFeats0, highResFeats1,
                scaledPoints, labels,
                state.VideoWidth, state.VideoHeight
            );

            // 儲存標註
            if (!state.Annotations.ContainsKey(frameIdx))
            {
                state.Annotations[frameIdx] = new Dictionary<int, ObjectAnnotation>();
            }

            state.Annotations[frameIdx][objectId] = new ObjectAnnotation
            {
                ObjectId = objectId,
                Points = points,
                Labels = labels,
                LastMask = maskArray
            };

            Console.WriteLine($"物件 {objectId} 在影格 {frameIdx} 的遮罩生成完成, IOU: {iou:F4}");

            return (maskArray, iou);
        }

        /// <summary>
        /// 在整個影片中傳播追蹤
        /// </summary>
        public Dictionary<int, Dictionary<int, float[,]>> PropagateInVideo(VideoInferenceState state)
        {
            Console.WriteLine("開始在影片中傳播追蹤...");

            var videoSegments = new Dictionary<int, Dictionary<int, float[,]>>();

            // 處理每個影格
            for (int frameIdx = 0; frameIdx < state.FramePaths.Count; frameIdx++)
            {
                Console.WriteLine($"處理影格 {frameIdx + 1}/{state.FramePaths.Count}");

                // 編碼當前影格（如果尚未編碼）
                if (!state.FrameEmbeddings.ContainsKey(frameIdx))
                {
                    using (var frame = Cv2.ImRead(state.FramePaths[frameIdx]))
                    {
                        var imageTensor = PreprocessImage(frame, out _, out _);
                        state.FrameEmbeddings[frameIdx] = EncodeImage(imageTensor);
                    }
                }

                var (imageEmbed, highResFeats0, highResFeats1) = state.FrameEmbeddings[frameIdx];
                videoSegments[frameIdx] = new Dictionary<int, float[,]>();

                // 對每個已標註的物件進行追蹤
                foreach (var objAnnotations in state.Annotations.Values)
                {
                    foreach (var annotation in objAnnotations.Values)
                    {
                        // 使用原始標註點進行預測
                        float scaleX = (float)ImageSize / state.VideoWidth;
                        float scaleY = (float)ImageSize / state.VideoHeight;

                        Point[] scaledPoints = new Point[annotation.Points.Length];
                        for (int i = 0; i < annotation.Points.Length; i++)
                        {
                            scaledPoints[i] = new Point(
                                (int)(annotation.Points[i].X * scaleX),
                                (int)(annotation.Points[i].Y * scaleY)
                            );
                        }

                        var (maskArray, _) = DecodeMask(
                            imageEmbed, highResFeats0, highResFeats1,
                            scaledPoints, annotation.Labels,
                            state.VideoWidth, state.VideoHeight
                        );

                        videoSegments[frameIdx][annotation.ObjectId] = maskArray;
                    }
                }
            }

            Console.WriteLine("影片追蹤傳播完成!");
            return videoSegments;
        }

        /// <summary>
        /// 將追蹤結果轉換為 OpenCV Mat
        /// </summary>
        public Mat MaskToMat(float[,] maskArray)
        {
            int height = maskArray.GetLength(0);
            int width = maskArray.GetLength(1);
            Mat mask = new Mat(height, width, MatType.CV_8UC1);

            unsafe
            {
                byte* data = (byte*)mask.DataPointer;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        data[y * width + x] = maskArray[y, x] > 0 ? (byte)255 : (byte)0;
                    }
                }
            }

            return mask;
        }

        public Mat OpenCloseMask(Mat mask, int kernelSize, int iterations)
        {
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(kernelSize, kernelSize));

            Mat dilated = new Mat();
            Cv2.Dilate(mask, dilated, kernel, iterations: iterations);

            Mat eroded = new Mat();
            Cv2.Erode(dilated, eroded, kernel, iterations: iterations * 2);

            Mat closed = new Mat();
            Cv2.Dilate(eroded, closed, kernel, iterations: iterations);

            dilated.Dispose();
            eroded.Dispose();
            kernel.Dispose();

            return closed;
        }

        public Mat Visualize(Mat image, Mat mask, Point[] points, int[] labels)
        {
            Mat colorMask = new Mat(mask.Size(), MatType.CV_8UC3, new Scalar(0, 255, 0));
            Mat maskedColor = new Mat();
            Cv2.BitwiseAnd(colorMask, colorMask, maskedColor, mask);

            Mat result = new Mat();
            Cv2.AddWeighted(image, 0.7, maskedColor, 0.3, 0, result);

            for (int i = 0; i < points.Length; i++)
            {
                Scalar color = labels[i] == 1 ? new Scalar(0, 255, 0) : new Scalar(0, 0, 255);
                Cv2.Circle(result, points[i], 8, color, -1);
                Cv2.Circle(result, points[i], 8, new Scalar(255, 255, 255), 2);
            }

            colorMask.Dispose();
            maskedColor.Dispose();
            return result;
        }

        public void Dispose()
        {
            encoderSession?.Dispose();
            decoderSession?.Dispose();
        }
    }
}
