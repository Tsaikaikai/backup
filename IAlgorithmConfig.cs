namespace AgingAOI.Core.Interfaces
{
    /// <summary>
    /// 演算法參數的統一存取介面（配置分離規範）。
    ///
    /// ════════════════════════════════════════════════════════════
    /// 規範 2：演算法參數必須抽成獨立的 Config 類別，不可散在演算法內部
    ///
    ///   好處：
    ///     1. 平台端可以統一存 / 讀所有演算法的參數，不必認識每一個型別。
    ///     2. 參數類別加上 [Category] / [DisplayName] / [Description] 後，
    ///        UI 端可用 PropertyGrid 類控制項自動生成設定畫面。
    ///     3. 演算法邏輯與調機參數分離，工廠端調參不需要改程式。
    ///
    /// 實作 Config 類別時請遵守：
    ///   · 每個參數都標 [Category]（用 "1.xxx" 控制排序）、[DisplayName]、
    ///     [Description]（寫給現場調機人員看）。
    ///   · 成員須為 public 可讀寫、類別須有無參建構子（XmlSerializer 限制）；
    ///     不可使用 Dictionary、介面型別、Mat。需要「鍵值對」請用
    ///     List&lt;含 Name 欄位的類別&gt;。
    ///   · Load 必須「逐欄位複製」，不可換掉整個物件參考
    ///     （IAlgorithm.Config 已被 UI 繫結，換參考會讓繫結失效）。
    ///   · 不可放 OK/NG 判定以外的流程參數（相機曝光、增益、張數、間隔、
    ///     檔案路徑等屬平台 Recipe / 設備設定，不屬於演算法 Config）。
    ///   · 影像類資源（例如黃金樣板遮罩）不能存進 Config；請放在 Config
    ///     檔旁邊的同一資料夾，由演算法自行載入並快取。
    ///
    /// 【每個 Zone 各自一組參數】
    ///   · Config 內以 List&lt;ZoneParameter&gt; 保存各 Zone 的參數（Zones[i] 即第 i 個 Zone，
    ///     內含該 Zone 的 ROI 與所有檢測參數）；檢測時每個 Zone 只套用自己那一組，
    ///     Zone 之間互不共用、互不影響。
    ///   · Zone 參數只放「該 Zone 的檢測參數」（ROI、各項門檻等）。
    ///     不可放檢測類別（線缺陷 / 面缺陷）或其啟用開關：檢測類別由 Execute 的 mode
    ///     對所有 Zone 統一指定，Zone 之間一律相同。
    ///   · Zone 數 = Zones.Count，必須透過 ZoneCount 對外公開。平台建立
    ///     emptyZones、解讀 ZoneResult / FrameDetail 都以此為準，順序與 Zones 一致。
    ///   · 與 Zone 無關的整體參數（例如對整張影像生效的前處理）另外放在 Config 最上層，
    ///     不要在每個 ZoneParameter 重複。
    ///   · Zone 內可有 Name 欄位，便於 UI 顯示；但上報與結果仍以序號（Index）為準。
    ///   · ZoneParameter 建議加 [TypeConverter(typeof(ExpandableObjectConverter))]，
    ///     PropertyGrid 才能展開成每個 Zone 一個群組。
    ///   · Load 時 Zones 以「清空後逐一加入複本」方式更新，不可換掉 List 物件參考。
    ///   · ★ XmlSerializer 陷阱：反序列化會把檔案內容「加進」屬性上已存在的 List。
    ///     因此 Zones 屬性必須宣告為空列表（不可用集合初始設定式放預設 Zone），
    ///     預設值改在演算法建構時才填入，否則讀檔後 Zone 會重複、ZoneCount 變錯。
    /// ════════════════════════════════════════════════════════════
    /// </summary>
    public interface IAlgorithmConfig
    {
        /// <summary>
        /// Zone 數量（= 每個 Zone 各自一組參數的組數）。
        /// 平台傳給 IAlgorithm.Execute 的 emptyZones 長度必須等於此值。
        /// </summary>
        int ZoneCount { get; }

        /// <summary>將目前參數寫入檔案。</summary>
        void Save(string path);

        /// <summary>
        /// 從檔案讀取參數並覆寫目前值。
        /// 檔案不存在時直接 return，保留目前值（不拋例外）。
        /// </summary>
        void Load(string path);
    }
}
