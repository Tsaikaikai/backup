using AgingAOI.Core.Interfaces;
using AgingAOI.Core.Models;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using LogLevel = AgingAOI.Core.Interfaces.LogLevel;   // OpenCvSharp 也有 LogLevel，需指定

namespace AgingAOI.Sample
{
    /// <summary>
    /// 【給平台人員的呼叫示範】—— 模擬「平台」如何使用 IAlgorithm。
    ///
    /// 平台一次檢測的完整流程：
    ///   1. 建立演算法、載入參數（開機或換機型時做一次）
    ///   2. 連拍取得 N 張影像，收進 List&lt;Mat&gt;
    ///   3. 呼叫 Execute 一次（傳入整包、emptyZones、檢測類別）
    ///   4. 回傳 false → 記為 Fault（設備 / 流程異常，不計良率，不上報 Zone 結果）
    ///   5. 回傳 true  → 由 zones 推導整體 OK / NG，組上報欄位
    ///   6. 依 frameDetails 存圖、寫報告
    ///   7. 釋放：輸入的 frames 與輸出 frameDetails 內的 Mat 都由平台釋放
    ///
    /// 執行：PlatformCallDemo.Run(@"C:\temp\demo");
    /// 本檔使用 MockAlgorithm 代替真實演算法；換成真實演算法只需改 Step 1 的 new。
    /// </summary>
    public static class PlatformCallDemo
    {
        // 模擬取像用的張數與解析度（實機為 5472x3648、約 50 張；示範用小圖）
        private const int FrameCount = 5;
        private const int Width = 640;
        private const int Height = 480;

        public static void Run(string outputDir)
        {
            ILogService log = new ConsoleLogService();
            Directory.CreateDirectory(outputDir);

            // ════ Step 1: 建立演算法、載入參數（開機 / 換機型時做一次）════════
            using (IAlgorithm algorithm = new MockAlgorithm(log))
            {
                string configPath = Path.Combine(outputDir, "AlgorithmConfig.xml");
                if (!File.Exists(configPath))
                    algorithm.Config.Save(configPath);       // 第一次先寫出預設值，供現場調參
                algorithm.Config.Load(configPath);           // 檔案不存在時 Load 直接 return，不拋例外

                log.Info("演算法：" + algorithm.Name);

                // Zone 數由 Config 決定；每個 Zone 各自一組參數（Config 內的 Zones[i]），
                // 演算法檢測第 i 個 Zone 時只套用第 i 組。emptyZones 長度必須等於 ZoneCount。
                int zoneCount = algorithm.Config.ZoneCount;
                log.Info("Zone 數 = " + zoneCount);

                // ── 範例 A：線缺陷 + 面缺陷一次檢測（產線一般做法）──────────
                InspectOnce(algorithm, log, outputDir, "A001",
                            InspectMode.LineDefect | InspectMode.SurfaceDefect,
                            emptyZones: EmptyZones(zoneCount));

                // ── 範例 B：只做線缺陷，且最後一個 Zone 為空 ───────────────
                InspectOnce(algorithm, log, outputDir, "A002",
                            InspectMode.LineDefect,
                            emptyZones: EmptyZones(zoneCount, zoneCount - 1));

                // ── 範例 C：錯誤輸入（mode 未指定）→ 回傳 false → Fault ───
                InspectOnce(algorithm, log, outputDir, "A003",
                            (InspectMode)0,
                            emptyZones: EmptyZones(zoneCount));

                // ── 範例 D：錯誤輸入（emptyZones 長度與 ZoneCount 不符）→ Fault ──
                InspectOnce(algorithm, log, outputDir, "A004",
                            InspectMode.LineDefect,
                            emptyZones: EmptyZones(zoneCount + 1));

                // ── 範例 E：4 片的機型，第 2 片（Zone 1）為空 ──────────────
                // 換機型 = 重新 Config.Load；載入後 ZoneCount 可能改變，平台要重新讀取。
                SwitchRecipe(algorithm, log, outputDir, 4);
                InspectOnce(algorithm, log, outputDir, "A005",
                            InspectMode.LineDefect | InspectMode.SurfaceDefect,
                            emptyZones: EmptyZones(algorithm.Config.ZoneCount, 1));

                // ── 範例 F：3 片的機型，第 1 片（Zone 0）為空 ──────────────
                SwitchRecipe(algorithm, log, outputDir, 3);
                InspectOnce(algorithm, log, outputDir, "A006",
                            InspectMode.LineDefect | InspectMode.SurfaceDefect,
                            emptyZones: EmptyZones(algorithm.Config.ZoneCount, 0));

                // ── 範例 G：4 片的機型，第 2、3 片（Zone 1、Zone 2）為空 ────
                // EmptyZones 的第二個參數起可一次指定多個空 Zone：{false, true, true, false}
                SwitchRecipe(algorithm, log, outputDir, 4);
                InspectOnce(algorithm, log, outputDir, "A007",
                            InspectMode.LineDefect | InspectMode.SurfaceDefect,
                            emptyZones: EmptyZones(algorithm.Config.ZoneCount, 1, 2));
            }
        }

        /// <summary>平台對「一個產品」的一次完整檢測。</summary>
        private static void InspectOnce(IAlgorithm algorithm, ILogService log, string outputDir,
                                        string taskId, InspectMode mode, bool[] emptyZones)
        {
            log.Info("──── 檢測開始 id=" + taskId + "，mode=" + mode + " ────");

            // 平台自己控制逾時；演算法需定期檢查 token，被取消時回傳 false
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30)))
            {
                // ════ Step 2: 連拍取像，收進 List<Mat> ═══════════════════════
                //   格式：BGR、8UC3、整張原圖，依拍攝順序。N 張同時留在記憶體。
                var frames = new List<Mat>();
                ZoneResult[] zones = null;
                FrameDetail[] frameDetails = null;
                try
                {
                    for (int i = 0; i < FrameCount; i++)
                        frames.Add(GrabFrame(i));

                    // ════ Step 3: 呼叫 Execute（一次，整包）═════════════════
                    int defectCount;
                    bool ok = algorithm.Execute(frames, emptyZones, mode,
                                                out zones, out frameDetails,
                                                out defectCount, cts.Token);

                    // ════ Step 4: false → Fault ══════════════════════════════
                    if (!ok)
                    {
                        // 此時 zones / frameDetails 為空陣列（非 null），不可使用其內容。
                        // Fault = 設備 / 流程異常：本片沒有有效判定，不計入良率，不上報 Zone 結果。
                        log.Error("[" + taskId + "] 結果：Fault（ALGORITHM_FAILED）");
                        return;
                    }

                    // ════ Step 5: 由 zones 推導整體結果與上報欄位 ═══════════
                    bool overallOk = zones.All(z => z.IsOk);   // 全部 Zone 皆 "OK" 才算 OK

                    // 各 Zone 是否通過：true;false;…
                    string isPass = string.Join(";", zones.Select(z => z.IsOk ? "true" : "false"));
                    // 失敗說明：各 Zone 的 Type 以 "-" 串接，例如 "OK-TM1500"
                    string failure = overallOk
                        ? ""
                        : "1;" + string.Join("-", zones.Select(z => z.Type)) + " NG";
                    // MCMQ 缺陷碼：最後一個 NG Zone 的 Type；缺陷數：defectCount
                    string mcmqCode = zones.Where(z => !z.IsOk).Select(z => z.Type).LastOrDefault() ?? "OK";

                    log.Info("[" + taskId + "] 結果：" + (overallOk ? "OK" : "NG"));
                    log.Info("  Is_Pass     = " + isPass);
                    log.Info("  Failure_Set = " + failure);
                    log.Info("  DefectCode  = " + mcmqCode + "，DefectCount = " + defectCount);

                    // ════ Step 6: 依 frameDetails 存圖、寫報告 ═══════════════
                    SaveImages(outputDir, taskId, frameDetails);
                    // 逐張明細（Pattern、HSV、AvgHsvList…）可序列化進 DetailReport.txt，此處略

                    // 位置鍵：frameDetails[i] 對應 frames[i]；Zones[j] 對應 ROI 順序
                }
                finally
                {
                    // ════ Step 7: 釋放 ═══════════════════════════════════════
                    // 輸入影像：呼叫端（平台）自己建立，自己釋放。
                    foreach (var f in frames) f.Dispose();

                    // 輸出影像：frameDetails 內每個 ZoneFrame 的 Image / DebugImage，
                    // 所有權在 Execute 回傳後交給平台，用完要 Dispose。
                    DisposeOutputImages(frameDetails);
                }
            }
        }

        // ── 平台端輔助 ────────────────────────────────────────────

        /// <summary>
        /// 【僅示範用】模擬「換成另一個機型」：產生一份 N 個 Zone 的 Config 檔並載入。
        /// 真實平台是把各機型自己的 Config 檔路徑傳給 algorithm.Config.Load(path)，
        /// 不會（也不需要）認識具體的 Config 類別。
        /// </summary>
        private static void SwitchRecipe(IAlgorithm algorithm, ILogService log, string outputDir, int zoneCount)
        {
            var cfg = new MockAlgorithmConfig();
            for (int i = 0; i < zoneCount; i++)
                cfg.Zones.Add(new MockZoneParameter { Name = "Zone " + i, SimulateLineNg = (i % 2 == 1) });

            string path = Path.Combine(outputDir, "Config_" + zoneCount + "zones.xml");
            cfg.Save(path);
            algorithm.Config.Load(path);
            log.Info("切換機型：" + zoneCount + " 片，ZoneCount = " + algorithm.Config.ZoneCount);
        }

        /// <summary>建立 emptyZones：長度 = Zone 數，預設全部 false（要檢測），指定的序號設為 true（空 Zone）。</summary>
        private static bool[] EmptyZones(int zoneCount, params int[] emptyIndexes)
        {
            var empty = new bool[zoneCount];
            foreach (int i in emptyIndexes)
                if (i >= 0 && i < zoneCount) empty[i] = true;
            return empty;
        }

        /// <summary>模擬取像：每張產生不同亮度的 BGR 8UC3 影像。</summary>
        private static Mat GrabFrame(int index)
        {
            return new Mat(Height, Width, MatType.CV_8UC3, new Scalar(40 * index, 40 * index, 40 * index));
        }

        /// <summary>依舊版命名規則存檔：{id}_Zone{zz}_Frame{fff}_{Type}.png，NG 與 OK 分資料夾。</summary>
        private static void SaveImages(string outputDir, string taskId, FrameDetail[] frameDetails)
        {
            string okDir = Path.Combine(outputDir, "OK");
            string ngDir = Path.Combine(outputDir, "NG");
            Directory.CreateDirectory(okDir);
            Directory.CreateDirectory(ngDir);

            foreach (var frame in frameDetails)
            {
                foreach (var zf in frame.Zones)
                {
                    if (zf.Image == null) continue;                    // Empty Zone 等沒有影像
                    if (zf.Type == DefectCode.Drop || zf.Type == DefectCode.NoAlign) continue;

                    string dir  = zf.Type.StartsWith("TM") ? ngDir : okDir;
                    string name = string.Format("{0}_Zone{1:00}_Frame{2:000}_{3}.png",
                                                taskId, zf.Index, frame.Index, zf.Type);
                    Cv2.ImWrite(Path.Combine(dir, name), zf.Image);
                }
            }
        }

        private static void DisposeOutputImages(FrameDetail[] frameDetails)
        {
            if (frameDetails == null) return;
            foreach (var frame in frameDetails)
            {
                if (frame == null || frame.Zones == null) continue;
                foreach (var zf in frame.Zones)
                {
                    if (zf == null) continue;
                    if (zf.Image != null) zf.Image.Dispose();
                    if (zf.DebugImage != null) zf.DebugImage.Dispose();
                }
            }
        }

        /// <summary>示範用的簡易日誌；平台請改用自己的 ILogService 實作。</summary>
        private class ConsoleLogService : ILogService
        {
            public void Log(LogLevel level, string message)
            {
                Console.WriteLine("[" + level + "] " + message);
            }
            public void Debug(string message)   { Log(LogLevel.Debug, message); }
            public void Info(string message)    { Log(LogLevel.Info, message); }
            public void Warning(string message) { Log(LogLevel.Warning, message); }
            public void Error(string message)   { Log(LogLevel.Error, message); }
        }
    }
}
