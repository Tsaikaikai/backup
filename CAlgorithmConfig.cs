using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Serialization;

namespace AIF_2D_AOI
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
        [Category("0.Base"), DisplayName("Image Width")]
        public int ImageWidth { get; set; }

        [Category("0.Base"), DisplayName("Image Height")]
        public int ImageHeight { get; set; }

        #endregion

        #region 1.Alignment

        [Category("1.Alignment"), DisplayName("Search Region Size (pix)")]
        public int AlignSearchRegionSize { get; set; }

        [Category("1.Alignment"), DisplayName("Search Overlap (pix)")]
        public double AlignSearchOverlap { get; set; }

        [Category("1.Alignment"), DisplayName("MaxPos")]
        public int MaxPos { get; set; }

        [Category("1.Alignment"), DisplayName("ToleranceAngle")]
        public int ToleranceAngle { get; set; }

        [Category("1.Alignment"), DisplayName("Score (0-1)")]
        public double MatchMinScore { get; set; }

        [Category("1.Alignment"), DisplayName("MinReduceArea")]
        public int MinReduceArea { get; set; }

        [Category("1.Alignment"), DisplayName("Max Overlap")]
        public double MaxOverlap { get; set; }

        [Category("1.Alignment"), DisplayName("Buffer (pix)")]
        public int Buffer { get; set; }

        [Category("1.Alignment"), DisplayName("Scale Mode")]
        public bool ScaleMode { get; set; }

        [Category("1.Alignment"), DisplayName("Align Scale")]
        public double AlignScale { get; set; }

        #endregion

        #region 2.Inspection

        [Category("2.Inspection"), DisplayName("AI Model Path")]
        public string AIModelPath { get; set; }

        [Category("2.Inspection"), DisplayName("UseGPU")]
        public int UseGPU { get; set; }

        [Category("2.Inspection"), DisplayName("Batch Size")]
        public int BatchSize { get; set; }

        [Category("2.Inspection"), DisplayName("AI Input Size")]
        public int AIInputSize { get; set; }

        [Category("2.Inspection"), DisplayName("Overlap Rate")]
        public double OverlapRate { get; set; }

        [Category("2.Inspection"), DisplayName("CD Threshold")]
        public float CDThreshold { get; set; }

        [Category("2.Inspection"), DisplayName("Resize Mode")]
        public bool ResizeMode { get; set; }

        [Category("2.Inspection"), DisplayName("Bypass Mode")]
        public bool BypassMode { get; set; }

        [Category("2.Inspection"), DisplayName("Bypass Json Folder")]
        public string BypassJsonFolder { get; set; }

        [Category("3.Image Enhancement"), DisplayName(".Enhancement Mode")]
        public bool EnhancementMode { get; set; }

        [Category("3.Image Enhancement"), DisplayName("Contrast")]
        public float Contrast { get; set; }

        [Category("3.Image Enhancement"), DisplayName("Gamma Correction")]
        public float GammaCorrection { get; set; }

        [Category("3.Image Enhancement"), DisplayName("Brightness")]
        public int Brightness { get; set; }

        [Category("4.Bright Balance"), DisplayName(".Balance Mode")]
        public bool BalanceMode { get; set; }

        [Category("4.Bright Balance"), DisplayName("Kernel Size")]
        public int BlurKernelSize { get; set; }

        [Category("4.Bright Balance"), DisplayName("Add Bright")]
        public int AddBright { get; set; }

        #endregion
    }
}
