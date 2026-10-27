using AgingAOI.Core.Models;
using OpenCvSharp;
using System.Collections.Generic;
using System.Threading;

namespace AgingAOI.Core.Interfaces
{
    /// <summary>
    /// 演算法抽象介面。
    /// 所有演算法都必須實作此介面，確保平台端呼叫方式統一，演算法可無縫替換。
    ///
    /// 一次輸入「同一產品的 N 張連拍影像」，輸出逐 Zone 的檢測結果。
    /// 演算法只管「輸入影像 + 參數 → 檢測結果」，不處理機台名稱、檔案路徑、
    /// 網路、報告與流程；影像存檔、報告、上報全部由平台依輸出資料處理。
    ///
    /// ════════════════════════════════════════════════════════════
    /// 【例外契約】—— 本介面採「不拋例外」設計
    ///
    ///   Execute 內部必須用 try-catch 把所有例外攔下來，
    ///   轉成「回傳 false」交給平台層處理（平台記為 Fault，不計入良率）。
    ///
    ///   ✕ 錯誤示範： catch (Exception ex) { throw new Exception(...); }
    ///   ○ 正確示範： catch (Exception ex) { /* 記錄、釋放已建立的輸出 Mat */ return false; }
    ///
    ///   唯一例外：建構子載入模型 / 樣板失敗時，可丟 AlgorithmException
    ///   （啟動階段問題，應讓系統直接進 Error 狀態）。
    /// ════════════════════════════════════════════════════════════
    ///
    /// 【判定責任】
    ///   本介面由演算法直接判定每個 Zone 的 OK / NG（ZoneResult.Type）。
    ///   不再有單一 score，也不使用 RecipeData.NgThreshold。
    ///   整體 OK = 全部 Zone 的 Type 皆為 "OK"，由平台推導，演算法不另外輸出。
    /// </summary>
    public interface IAlgorithm : IDevice
    {
        /// <summary>
        /// 演算法參數。平台端透過此屬性統一存取 / 儲存參數，
        /// 不需要知道具體是哪一個 Config 類別。
        /// </summary>
        IAlgorithmConfig Config { get; }

        /// <summary>
        /// 執行演算法主流程。
        /// ──────────────────────────────────────────────
        /// 參數排序慣例：
        ///   1. 輸入參數（影像、設定）
        ///   2. 輸出參數（out）依「結果 → 視覺 → 診斷」排列
        ///   3. CancellationToken 一律放最後
        /// ──────────────────────────────────────────────
        /// </summary>
        /// <param name="frames">
        /// N 張輸入影像，依拍攝順序排列（frames[0] 為第一張）。
        /// 格式：BGR、8UC3、整張原圖。
        /// 所有權屬呼叫端：不可修改、不可 Dispose；要保留請 Clone()。
        /// 演算法不得在 Execute 返回後保留任何影像。
        /// </param>
        /// <param name="emptyZones">
        /// 長度 = Config.ZoneCount。true = 該 Zone 為空，不檢測，其 ZoneResult.Type 回傳 "OK"
        /// （frameDetails 中該 Zone 的 Image 為 null）。任何序號的 Zone 都可以設為空，
        /// 例如 4 片的第 2 片為空 = {false, true, false, false}；3 片的第 1 片為空 = {true, false, false}。
        /// 片數由機型的 Config 決定（3 片的機型 ZoneCount = 3）；長度與 ZoneCount 不符視為流程異常，
        /// 回傳 false。換機型（重新 Config.Load）後 ZoneCount 可能改變，平台須重新讀取。
        /// 每個 Zone 套用 Config 中「該 Zone 自己」的一組參數（第 i 個 Zone 用第 i 組），
        /// Zone 之間參數互相獨立；zones、frameDetails 內的 Zone 順序與此一致。
        /// </param>
        /// <param name="mode">
        /// 檢測類別：LineDefect（線缺陷）、SurfaceDefect（面缺陷），或兩者以 | 合併。
        /// 對所有 Zone 一律相同，不可逐 Zone 指定；Zone 各自的參數只影響「怎麼檢」，不影響「檢什麼類別」。
        /// 只檢測被指定的類別，結果中只會出現對應類別的缺陷代碼（見 DefectCode）。
        /// 不可傳入 0（未指定任何類別），否則視為流程異常，回傳 false。
        /// </param>
        /// <param name="zones">
        /// 【結果】逐 Zone 最終結果，長度 = Zone 數，順序 = ROI 順序。
        /// 平台上報只依 ZoneResult.Type。
        /// </param>
        /// <param name="frameDetails">
        /// 【視覺】逐張逐 Zone 明細與影像，長度 = frames.Count，frameDetails[i] 對應 frames[i]。
        /// 其中的 Mat 所有權交給呼叫端。
        /// </param>
        /// <param name="defectCount">
        /// 【診斷】本次內 Type 為 TM 代碼的 (Zone, 張) 數量，每次呼叫從 0 起算。
        /// </param>
        /// <param name="cancellationToken">取消信號；長迴圈請定期檢查，被取消時回傳 false。</param>
        /// <returns>
        /// true  = 演算法流程完整走完，輸出參數有效。<br/>
        /// false = 流程中途異常（影像為空、尺寸不符、參數 / 樣板缺失、被取消等），
        ///         輸出參數無效：zones、frameDetails 為空陣列（不得為 null），
        ///         且演算法已自行釋放所有已建立的輸出 Mat。
        ///         實作端絕對不可以改成拋例外，請見上方例外契約。
        /// </returns>
        bool Execute(IReadOnlyList<Mat> frames, bool[] emptyZones, InspectMode mode,
                     out ZoneResult[] zones, out FrameDetail[] frameDetails,
                     out int defectCount, CancellationToken cancellationToken);
    }
}
