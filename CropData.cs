using System;

/// <summary>
/// 表示切割後的小圖資料，包含位置和狀態資訊
/// </summary>
public class CropData
{
    /// <summary>
    /// 切割圖片的唯一識別碼
    /// </summary>
    public string CropID { get; set; }
    
    /// <summary>
    /// 切片在序列中的索引
    /// </summary>
    public int SliceIndex { get; set; }
    
    /// <summary>
    /// 起始位置的 X 座標
    /// </summary>
    public int StartX { get; set; }
    
    /// <summary>
    /// 起始位置的 Y 座標
    /// </summary>
    public int StartY { get; set; }
    
    /// <summary>
    /// 切割圖片的寬度
    /// </summary>
    public int Width { get; set; }
    
    /// <summary>
    /// 切割圖片的高度
    /// </summary>
    public int Height { get; set; }
    
    /// <summary>
    /// 切割圖片的二進制資料
    /// </summary>
    public byte[] ImageData { get; set; }
    
    /// <summary>
    /// 表示切割圖片是否為 NG (不良品)，預設為 false
    /// </summary>
    public bool IsNG { get; set; } = false;
    
    /// <summary>
    /// 預設建構函式
    /// </summary>
    public CropData()
    {
    }
    
    /// <summary>
    /// 帶參數的建構函式
    /// </summary>
    public CropData(string cropID, int sliceIndex, int startX, int startY, int width, int height, byte[] imageData)
    {
        CropID = cropID;
        SliceIndex = sliceIndex;
        StartX = startX;
        StartY = startY;
        Width = width;
        Height = height;
        ImageData = imageData;
    }
}
