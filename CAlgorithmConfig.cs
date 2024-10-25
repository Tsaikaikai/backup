using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Serialization;

namespace MeasureWaferRecipe
{
    public class CAlgorithmConfig
    {
        #region Initial
        public CAlgorithmConfig()
        {

        }

        public bool WriteXmlConfig(string configFileName)
        {
            try
            {
                if (!File.Exists(configFileName))
                {
                    configFileName = configFileName.Replace("\\\\", "/").Replace("\\", "/");
                    int index = configFileName.LastIndexOf("/");
                    Directory.CreateDirectory(configFileName.Substring(0, index));
                }
                // Write config value into xml.
                using (FileStream fileStream = new FileStream(configFileName, FileMode.Create))
                {
                    XmlSerializer xmlSerializer = new XmlSerializer(typeof(CAlgorithmConfig));
                    xmlSerializer.Serialize(fileStream, this);
                    fileStream.Close();
                }
                return true;
            }
            catch (Exception ex)
            {
                throw new Exception($"[WriteXmlConfig]{ex.Message}");
            }
        }

        public CAlgorithmConfig ReadXmlConfig(string configFileName)
        {
            CAlgorithmConfig tempConfig;
            try
            {
                // If file does not exists, create default one.
                if (!File.Exists(configFileName))
                    this.WriteXmlConfig(configFileName);

                // Open and read a crypto-xml file.
                using (FileStream fileStream = new FileStream(configFileName, FileMode.Open))
                using (XmlTextReader xmlTextReader = new XmlTextReader(fileStream))
                {
                    XmlSerializer xmlSerializer = new XmlSerializer(typeof(CAlgorithmConfig));
                    tempConfig = (CAlgorithmConfig)xmlSerializer.Deserialize(xmlTextReader);

                    xmlTextReader.Close();
                    fileStream.Close();
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"[ReadXmlConfig]{ex.Message}");
            }

            return tempConfig;
        }
        #endregion

        #region Base
        [Category("0.Base"), DisplayName("Line Scan Image Height (pix)")]
        public int LineScanHeight { get; set; }

        [Category("0.Base"), DisplayName("Line Scan Image Width (pix)")]
        public int LineScanWidth { get; set; }

        [Category("0.Base"), DisplayName("Line Scan Image Slice Number")]
        public int LineScanSlice { get; set; }

        [Category("0.Base"), DisplayName("Line Scan Pixel Size (um/pix)")]
        public double LineScanPixelSize { get; set; }

        [Category("0.Base"), DisplayName("Edge Camera Pixel Size (um/pix)")]
        public double EdgePixelSize { get; set; }

        [Category("0.Base"), DisplayName("Edge Camera Image Height (pix)")]
        public int EdgeImageHeight { get; set; }

        [Category("0.Base"), DisplayName("Edge Camera Image Width (pix)")]
        public int EdgeImageWidth { get; set; }

        [Category("0.Base"), DisplayName("Edge Camera Image Slice Number")]
        public int EdgeImageSlice { get; set; }

        [Category("0.Base"), DisplayName("Notch Camera Pixel Size (um/pix)")]
        public double NotchMeasurePixSize { get; set; }

        [Category("0.Base"), DisplayName("Notch Camera Image Height (pix)")]
        public int NotchImageHeight { get; set; }

        [Category("0.Base"), DisplayName("Notch Camera Image Width (pix)")]
        public int NotchImageWidth { get; set; }

        [Category("0.Base"), DisplayName("Notch Camera Image Slice Number")]
        public double NotchImageSlice { get; set; }

        #endregion

        #region 1.辨識研磨前後
        //CheckBeforeOrAfterGrinding

        [Category("1.CheckGrinding"), DisplayName("UnGrinding Model Path")]
        public string UnGrindingModelPath { get; set; }

        [Category("1.CheckGrinding"), DisplayName("Grinding Model Path")]
        public string GrindingModelPath { get; set; }

        [Category("1.CheckGrinding"), DisplayName("CheckGrinding Min Score (0-1)")]
        public double GrindingMinScore { get; set; }

        [Category("1.CheckGrinding"), DisplayName("Grinding Rotate (deg)")]
        public double GrindingRotate { get; set; }

        [Category("1.CheckGrinding"), DisplayName("Open and Close region (pix)")]
        public int GrindingOpenClose { get; set; }
        #endregion

        #region 2.拼接對位

        //CalculateEdgePntAndCropNotch

        [Category("2.TileAlign"), DisplayName("Notch Model Path")]
        public string NotchModelPath { get; set; }

        [Category("2.TileAlign"), DisplayName("Crop Notch Width (pix)")]
        public int CropNotchWidth { get; set; }

        [Category("2.TileAlign"), DisplayName("Inside Inspect Width (mm)")]
        public double InsideInspectWidth { get; set; }

        [Category("2.TileAlign"), DisplayName("Crop Size (pix)")]
        public int CropSize { get; set; }

        [Category("2.TileAlign"), DisplayName("Align MinScore (0-1)")]
        public double AlignMinScore { get; set; }

        [Category("2.TileAlign"), DisplayName("Search Scale Range +-Rate (0-1)")]
        public double SearchScaleRange { get; set; }

        [Category("2.TileAlign"), DisplayName("Search Angle +-Range (deg)")]
        public double SearchAngleRange { get; set; }

        //CatchEdgeImageIndex

        [Category("2.TileAlign"), DisplayName("Area Total Grab Number")]
        public int AreaTotalNum { get; set; }

        [Category("2.TileAlign"), DisplayName("Measurement Catch Number")]
        public int CatchNum { get; set; }

        [Category("2.TileAlign"), DisplayName("Notch Shift Degree Angle (deg)")]
        public double NotchShiftDegreeAngle { get; set; }

        [Category("2.TileAlign"), DisplayName("Measure Edge Threshold")]
        public int MeasureThrshold { get; set; }

        [Category("2.TileAlign"), DisplayName("Measure Edge Sigma")]
        public int MeasureSigma { get; set; }
        #endregion

        #region 3.直徑檢
        [Category("3.Diameter Inspect"), DisplayName("White to Black Direction (L/R)")]
        public string Direction { get; set; }

        [Category("3.Diameter Inspect"), DisplayName("Gaussian Smooth Sigma")]
        public double Sigma { get; set; }

        [Category("3.Diameter Inspect"), DisplayName("Black White Amplitude")]
        public double Amplitude { get; set; }

        [Category("3.Diameter Inspect"), DisplayName("Measurement Length (pix)")]
        public int MeasurePosLength { get; set; }

        [Category("3.Diameter Inspect"), DisplayName("Measurement Width (pix)")]
        public int MeasurePosWidth { get; set; }

        [Category("3.Diameter Inspect"), DisplayName("Measurement Gap (pix)")]
        public int MeasureGap { get; set; }

        [Category("3.Diameter Inspect"), DisplayName("Notch Width (pix)")]
        public int NotchWidth { get; set; }

        [Category("3.Diameter Inspect"), DisplayName("Diameter tranfer alpha")]
        public double Alpha { get; set; }

        [Category("3.Diameter Inspect"), DisplayName("Diameter tranfer beta")]
        public double Beta { get; set; }
        #endregion

        #region 4.瑕疵檢

        //CropDomainImage

        [Category("4.Defect Inspect"), DisplayName("AI Draw Start X (pix)")]
        public int AIDrawStartX  { get; set; }

        //InspectWaferEdgeDefectAndDrawAi

        [Category("4.Defect Inspect"), DisplayName("Display Dilation (pix)")]
        public int DisplayDilation { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Paint Result (bool)")]
        public bool PaintResult { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Have Second Edge (bool)")]
        public bool HaveSecondEdge { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Only Inspect Edge (bool)")]
        public bool OnlyInspectEdge { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Second Line Bypass Width")]
        public int LineBypassWidth { get; set; }

        [Category("4.Defect Inspect"), DisplayName("AI Defect Size")]
        public int AIDefectSize { get; set; }

        [Category("4.Defect Inspect"), DisplayName("AI Min Gray")]
        public int AIMinGray { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Shift Edge")]
        public int ShiftEdge { get; set; }

        //Top
        [Category("4.Defect Inspect"), DisplayName("Top Inner Defect Area Min Size")]
        public int InnerDefectAreaSizeT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top ROI Width to Bevel Center (pix)")]
        public int ROIWidthtoBevelCenterT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top Bevel Width (pix)")]
        public int BevelWidthT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top Bevel Defect Max Gray")]
        public int BevelDefectMaxGrayT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top Inner Defect Max Gray")]
        public int InnerDefectMaxGrayT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top Gamma Correction")]
        public double GammaT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top Find Line Mask")]
        public int FindLineMaskT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top Find Line Low")]
        public double FindLineLowT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top Find Line High")]
        public double FindLineHighT { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Top Second E2E Distance")]
        public int SecondEdgeToEdgeDistanceT { get; set; }

        //Bottom
        [Category("4.Defect Inspect"), DisplayName("Bottom Inner Defect Area Min Size")]
        public int InnerDefectAreaSizeB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom ROI Width to Bevel Center (pix)")]
        public int ROIWidthtoBevelCenterB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom Bevel Width (pix)")]
        public int BevelWidthB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom Bevel Defect Max Gray")]
        public int BevelDefectMaxGrayB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom Inner Defect Max Gray")]
        public int InnerDefectMaxGrayB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom Gamma Correction")]
        public double GammaB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom Find Line Mask")]
        public int FindLineMaskB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom Find Line Low")]
        public double FindLineLowB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom Find Line High")]
        public double FindLineHighB { get; set; }

        [Category("4.Defect Inspect"), DisplayName("Bottom Second E2E Distance")]
        public int SecondEdgeToEdgeDistanceB { get; set; }
        #endregion

        #region 5.面幅檢

        //MeasureWaferEdgeB
        [Category("5.Edge Measurement"), DisplayName("Edge region rate")]
        public double EdgeRegionRate { get; set; }

        [Category("5.Edge Measurement"), DisplayName("Rotate Angle (deg)")]
        public int EdgeRotateAngle { get; set; }

        [Category("5.Edge Measurement"), DisplayName("Gray Max")]
        public double EdgeGrayMax { get; set; }

        [Category("5.Edge Measurement"), DisplayName("ROI Y1")]
        public double EdgeRoiY1 { get; set; }

        [Category("5.Edge Measurement"), DisplayName("ROI X1")]
        public double EdgeRoiX1 { get; set; }

        [Category("5.Edge Measurement"), DisplayName("ROI Y2")]
        public double EdgeRoiY2 { get; set; }

        [Category("5.Edge Measurement"), DisplayName("ROI X2")]
        public double EdgeRoiX2 { get; set; }

        [Category("5.Edge Measurement"), DisplayName("A1 Angle")]
        public double A1Ang { get; set; }

        [Category("5.Edge Measurement"), DisplayName("A2 Angle")]
        public double A2Ang { get; set; }

        [Category("5.Edge Measurement"), DisplayName("C1 Angle")]
        public double C1Ang { get; set; }

        [Category("5.Edge Measurement"), DisplayName("C2 Angle")]
        public double C2Ang { get; set; }

        [Category("5.Edge Measurement"), DisplayName("bu11")]
        public double Bu11 { get; set; }

        [Category("5.Edge Measurement"), DisplayName("bv11")]
        public double Bv11 { get; set; }

        [Category("5.Edge Measurement"), DisplayName("bu22")]
        public double Bu22 { get; set; }

        [Category("5.Edge Measurement"), DisplayName("bv22")]
        public double Bv22 { get; set; }

        [Category("5.Edge Measurement"), DisplayName("R Exclude Angle")]
        public double ExAngle{ get; set; }

        [Category("5.Edge Measurement"), DisplayName("Scale Width")]
        public double ScaleW { get; set; }

        [Category("5.Edge Measurement"), DisplayName("Scale Height")]
        public double ScaleH { get; set; }

        [Category("5.Edge Measurement"), DisplayName("Bevel Measure Number")]
        public double BevelMeasureNum { get; set; }

        #endregion

        #region 6.notch量測

        //MeasureNotch
        [Category("6.Notch Measurement"), DisplayName("Notch Measurement Check Region Rate")]
        public double NotchMeasureCheckRate { get; set; }

        [Category("6.Notch Measurement"), DisplayName("Notch Measurement Rotate Angle (deg)")]
        public double NotchMeasureRotate { get; set; }

        [Category("6.Notch Measurement"), DisplayName("Notch Area Model Path")]
        public string NotchAreaModelPath { get; set; }

        [Category("6.Notch Measurement"), DisplayName("Notch Match Minmun Score")]
        public double NotchMatchMinScore { get; set; }

        [Category("6.Notch Measurement"), DisplayName("Notch Measurement Resize Rate (0-1)")]
        public double NotchMeasureResizeRate { get; set; }

        [Category("6.Notch Measurement"), DisplayName("Notch Backlight Region Gray Min")]
        public double NotchGrayMin { get; set; }

        [Category("6.Notch Measurement"), DisplayName("B1 Angle (deg)")]
        public double B1Ang { get; set; }

        [Category("6.Notch Measurement"), DisplayName("B2 Angle (deg)")]
        public double B2Ang { get; set; }

        [Category("6.Notch Measurement"), DisplayName("U1 Start Rate (0-1)")]
        public double U1 { get; set; }

        [Category("6.Notch Measurement"), DisplayName("U2 Start Rate (0-1)")]
        public double U2 { get; set; }

        [Category("6.Notch Measurement"), DisplayName("Radius Width Left (um)")]
        public double RWidthLeft { get; set; }

        [Category("6.Notch Measurement"), DisplayName("Radius Width Right (um)")]
        public double RWidthRight { get; set; }

        [Category("6.Notch Measurement"), DisplayName("Pin Radius (mm)")]
        public double PinR { get; set; }

        [Category("6.Notch Measurement"), DisplayName("VR Angle (deg)")]
        public double VRAng { get; set; }

        #endregion

        #region 7.notch瑕疵檢

        //InspectNotchDefect_v2
        [Category("7.Notch Inspect"), DisplayName("Notch Inspect Check Region Rate")]
        public double NotchInspectCheckRate { get; set; }

        [Category("7.Notch Inspect"), DisplayName("Back White Min Gray")]
        public double BackWhiteMinGray { get; set; }

        [Category("7.Notch Inspect"), DisplayName("Notch Inspect Display Dilation (pix)")]
        public int NotchInspectDisplayDilation { get; set; }

        [Category("7.Notch Inspect"), DisplayName("Notch Inner Defect Max Gray")]
        public double NotchInnerDefectMaxGray { get; set; }

        [Category("7.Notch Inspect"), DisplayName("Notch Vm Y Up (pix)")]
        public int NotchVmRowUp { get; set; }

        [Category("7.Notch Inspect"), DisplayName("Notch Vm Y Down (pix)")]
        public int NotchVmRowDown { get; set; }

        [Category("7.Notch Inspect"), DisplayName("Notch Black Width (pix)")]
        public int NotchBlackWidth { get; set; }

        [Category("7.Notch Inspect"), DisplayName("Inner Defect Min Width (pix)")]
        public int InnerDefectMinWidth { get; set; }

        [Category("7.Notch Inspect"), DisplayName("Out Defect Min Width (pix)")]
        public int OutDefectMinWidth { get; set; }

        #endregion
    }
}
