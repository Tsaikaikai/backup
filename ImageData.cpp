#include "Stdafx.h"
#include "ImageData.h"

using namespace ImageProcessData;

ImageData::ImageData(ImageSetting ^ setting, SystemData ^ data, int index)
{
	if (setting == nullptr)
	{
		throw gcnew ArgumentNullException("setting");
	}

	if (data == nullptr)
	{
		throw gcnew ArgumentNullException("systemData");
	}

	this->isAllocated = false;

	this->systemData = data;

	this->index = index;

	this->dataName = setting->Name;

	this->fixable = setting->Fixable;

	if (setting->ImageWidth < 1)
	{
		throw gcnew ArgumentException("imageWidth < 1");
	}
	this->imageWidth = setting->ImageWidth;

	if (setting->ImageHeight < 1)
	{
		throw gcnew ArgumentException("ImageHeight < 1");
	}
	this->imageHeight = setting->ImageHeight;

	if (setting->ImageChannel != 1
		&& setting->ImageChannel != 3)
	{
		throw gcnew ArgumentException("ImageChannel not support.");
	}
	this->imageChannel = setting->ImageChannel;

	if (setting->ImageBit != 8
		&& setting->ImageBit != 16
		&& setting->ImageBit != 32)
	{
		throw gcnew ArgumentException("ImageBit not support.");
	}
	this->imageBit = setting->ImageBit;

	if (setting->ImageCount < 1)
	{
		throw gcnew ArgumentException("ImageCount < 1");
	}
	this->imageCount = setting->ImageCount;

	this->imageDirection = setting->ImageDirection;	

	this->extraCount = setting->ExtraCount;

	if (this->fixable)
	{
		this->AllocateData();
	}
}

ImageData::~ImageData()
{
	if (this->MData)
	{
		for (int cnt = 0; cnt < this->imageChannel; cnt++)
		{
			delete this->MData[cnt];
			this->MData[cnt] = nullptr;
		}

		delete this->MData;
		this->MData = nullptr;
	}

	if (this->PData)
	{
		for (int cnt = 0; cnt < this->imageChannel; cnt++)
		{
			delete this->PData[cnt];
			this->PData[cnt] = nullptr;
		}

		delete this->PData;
		this->PData = nullptr;
	}	
}

void ImageData::AllocateData(ImageSetting ^ setting)
{
	if (!this->fixable)
	{
		if (setting == nullptr)
		{
			throw gcnew ArgumentNullException("setting");
		}

		this->dataName = setting->Name;

		if (setting->ImageWidth < 1)
		{
			throw gcnew ArgumentException("imageWidth < 1");
		}
		this->imageWidth = setting->ImageWidth;

		if (setting->ImageHeight < 1)
		{
			throw gcnew ArgumentException("ImageHeight < 1");
		}
		this->imageHeight = setting->ImageHeight;

		if (setting->ImageChannel != 1
			&& setting->ImageChannel != 3)
		{
			throw gcnew ArgumentException("ImageChannel not support.");
		}
		this->imageChannel = setting->ImageChannel;

		if (setting->ImageBit != 8
			&& setting->ImageBit != 16
			&& setting->ImageBit != 32)
		{
			throw gcnew ArgumentException("ImageBit not support.");
		}
		this->imageBit = setting->ImageBit;

		if (setting->ImageCount < 1)
		{
			throw gcnew ArgumentException("ImageCount < 1");
		}
		this->imageCount = setting->ImageCount;

		this->imageDirection = setting->ImageDirection;

		this->extraCount = setting->ExtraCount;
	}

	this->AllocateData();
}

void ImageData::FreeData()
{
	if (this->fixable)
	{
		//	Fixable Data, clear only
		this->ClearData();
	}
	else
	{
		if (this->MData)
		{
			for (int cnt = 0; cnt < this->imageChannel; cnt++)
			{
				delete this->MData[cnt];
				this->MData[cnt] = nullptr;
			}

			delete this->MData;
			this->MData = nullptr;
		}

		if (this->PData)
		{
			for (int cnt = 0; cnt < this->imageChannel; cnt++)
			{
				delete this->PData[cnt];
				this->PData[cnt] = nullptr;
			}

			delete this->PData;
			this->PData = nullptr;
		}

		this->isAllocated = false;
	}	
}

void ImageData::ClearData()
{
	if (!this->PData) return;

	for (int cnt = 0; cnt < this->imageChannel; cnt++)
	{
		this->PData[cnt]->ClearData();
	}
}

void* ImageData::GetImagePtr(String^ name, int channel, int slice)
{
	if (channel < 0)
		throw gcnew Exception(String::Format(
			"GetImagePtr: Channel < 0"));

	if (channel > this->Channels - 1)
		throw gcnew Exception(String::Format(
			"GetImagePtr: Channel > Total Count."));

	if (slice < 0)
		throw gcnew Exception(String::Format(
			"GetImagePtr: Slice < 0."));

	if (slice > this->Count - 1)
		throw gcnew Exception(String::Format(
			"GetImagePtr: Slice > Total Count."));

	if (name == "PROCESS")
	{
		if (this->imageBit == 8)
		{
			return (char*)this->PData[channel]->ProcessData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else if (this->imageBit == 16)
		{
			return (short*)this->PData[channel]->ProcessData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else if (this->imageBit == 32)
		{
			return (int*)this->PData[channel]->ProcessData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else
		{
			throw gcnew Exception(String::Format(
				"GetImagePtr: imageBit != 8,16,32 ."));
		}
	}
	else if (name == "COMPARE")
	{
		if (this->imageBit == 8)
		{
			return (char*)this->PData[channel]->CompareData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else if (this->imageBit == 16)
		{
			return (short*)this->PData[channel]->CompareData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else if (this->imageBit == 32)
		{
			return (int*)this->PData[channel]->CompareData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else
		{
			throw gcnew Exception(String::Format(
				"GetImagePtr: imageBit != 8,16,32 ."));
		}
	}
	else if (name->Contains("EXTRA:"))
	{
		array<String^>^ tmpStringArray = name->Split(':');
		int extraIndex;

		if (tmpStringArray->Length != 2)
			throw gcnew Exception(String::Format(
				"GetImagePtr: Error Extra Count."));

		try
		{
			extraIndex = Convert::ToInt32(tmpStringArray[1]);
		}
		catch (Exception^)
		{
			throw gcnew Exception(String::Format(
				"GetImagePtr: Error Extra Index."));
		}

		if (extraIndex > this->extraCount - 1)
			throw gcnew Exception(String::Format(
				"GetImagePtr: Extra > Total Count."));

		if (this->imageBit == 8)
		{
			return (char*)this->PData[channel]->ExtraData[extraIndex] + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else if (this->imageBit == 16)
		{
			return (short*)this->PData[channel]->ExtraData[extraIndex] + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else if (this->imageBit == 32)
		{
			return (int*)this->PData[channel]->ExtraData[extraIndex] + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else
		{
			throw gcnew Exception(String::Format(
				"GetImagePtr: imageBit != 8,16,32 ."));
		}
	}
	else
	{
		if (this->imageBit == 8)
		{
			return (char*)this->PData[channel]->SourceData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else if (this->imageBit == 16)
		{
			return (short*)this->PData[channel]->SourceData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else if (this->imageBit == 32)
		{
			return (int*)this->PData[channel]->SourceData + 
				(this->imageWidth * this->imageHeight * slice);
		}
		else
		{
			throw gcnew Exception(String::Format(
				"GetImagePtr: imageBit != 8,16,32 ."));
		}
	}
}

void* ImageData::GetImagePtr(String^ name, int channel)
{
	if (channel < 0)
		throw gcnew Exception(String::Format(
			"GetImagePtr: Channel < 0"));

	if (channel > this->Channels - 1)
		throw gcnew Exception(String::Format(
			"GetImagePtr: Channel > Total Count."));

	if (name == "PROCESS")
	{
		return this->PData[channel]->ProcessData;
	}
	else if (name == "COMPARE")
	{
		return this->PData[channel]->CompareData;
	}
	else if (name->Contains("EXTRA:"))
	{
		array<String^>^ tmpStringArray = name->Split(':');
		int extraIndex;

		if (tmpStringArray->Length != 2)
			throw gcnew Exception(String::Format(
				"GetImagePtr: Error Extra Count."));

		try
		{
			extraIndex = Convert::ToInt32(tmpStringArray[1]);
		}
		catch (Exception^)
		{
			throw gcnew Exception(String::Format(
				"GetImagePtr: Error Extra Index."));
		}

		if (extraIndex > this->extraCount - 1)
			throw gcnew Exception(String::Format(
				"GetImagePtr: Extra > Total Count."));

		return this->PData[channel]->ExtraData[extraIndex];
	}
	else
	{
		return this->PData[channel]->SourceData;
	}
}


long long ImageData::GetMDataID(String^ name, int channel, int slice)
{
	if (!this->systemData->MilEnable)
		throw gcnew Exception(String::Format(
			"GetMDataID: M-System not Enable."));

	if (channel < 0)
		throw gcnew Exception(String::Format(
			"GetMDataID: Channel < 0"));

	if (channel > this->Channels - 1)
		throw gcnew Exception(String::Format(
			"GetMDataID: Channel > Total Count."));

	if (slice < 0)
		throw gcnew Exception(String::Format(
			"GetMDataID: Slice < 0."));

	if (slice > this->Count - 1)
		throw gcnew Exception(String::Format(
			"GetMDataID: Slice > Total Count."));

	if (name == "PROCESS")
	{
		return this->MData[channel]->ProcessSlice[slice];
	}
	else if (name == "COMPARE")
	{
		return this->MData[channel]->CompareSlice[slice];
	}
	else if (name->Contains("EXTRA:"))
	{
		array<String^> ^ tmpStringArray = name->Split(':');
		int extraIndex;

		if (tmpStringArray->Length != 2)
			throw gcnew Exception(String::Format(
				"GetMDataID: Error Extra Count."));

		try
		{
			extraIndex = Convert::ToInt32(tmpStringArray[1]);			
		}
		catch (Exception ^)
		{
			throw gcnew Exception(String::Format(
				"GetMDataID: Error Extra Index."));
		}

		if (extraIndex > this->extraCount - 1)
			throw gcnew Exception(String::Format(
				"GetMDataID: Extra > Total Count."));

		return this->MData[channel]->ExtraSlice[extraIndex][slice];
	}
	else
	{
		return this->MData[channel]->SourceSlice[slice];
	}
}

long long ImageData::GetMDataID(String^ name, int channel)
{
	if (!this->systemData->MilEnable)
		throw gcnew Exception(String::Format(
			"GetMDataID: M-System not Enable."));

	if (channel < 0)
		throw gcnew Exception(String::Format(
			"GetMDataID: Channel < 0"));

	if (channel > this->Channels - 1)
		throw gcnew Exception(String::Format(
			"GetMDataID: Channel > Total Count."));

	if (name == "PROCESS")
	{
		return this->MData[channel]->ProcessScan;
	}
	else if (name == "COMPARE")
	{
		return this->MData[channel]->CompareScan;
	}
	else if (name->Contains("EXTRA:"))
	{
		array<String^> ^ tmpStringArray = name->Split(':');
		int extraIndex;

		if (tmpStringArray->Length != 2)
			throw gcnew Exception(String::Format(
				"GetMDataID: Error Extra Count."));

		try
		{
			extraIndex = Convert::ToInt32(tmpStringArray[1]);
		}
		catch (Exception ^)
		{
			throw gcnew Exception(String::Format(
				"GetMDataID: Error Extra Index."));
		}

		if (extraIndex > this->extraCount - 1)
			throw gcnew Exception(String::Format(
				"GetMDataID: Extra > Total Count."));

		return this->MData[channel]->ExtraScan[extraIndex];
	}
	else
	{
		return this->MData[channel]->SourceScan;
	}
}

String^ ImageData::GetImageName(String^ name)
{
	if (name == "PROCESS")
	{
		return "pImage";
	}
	else if (name == "COMPARE")
	{
		return "cImage";
	}
	else if (name->Contains("EXTRA:"))
	{
		array<String^> ^ tmpStringArray = name->Split(':');
		int extraIndex;

		if (tmpStringArray->Length != 2)
			throw gcnew Exception(String::Format(
				"GetImageName: Error Extra Count."));

		try
		{
			extraIndex = Convert::ToInt32(tmpStringArray[1]);
		}
		catch (Exception ^)
		{
			throw gcnew Exception(String::Format(
				"GetImageName: Error Extra Index."));
		}

		if (extraIndex > this->extraCount - 1)
			throw gcnew Exception(String::Format(
				"GetImageName: Extra > Total Count."));

		return String::Format("e{0:00}Image", extraIndex);
	}
	else
	{
		return "sImage";
	}
}

void ImageData::SaveScanImage(String^ fullFileName, String^ type, int resize)
{
	if (resize < 1 || resize > 100) resize = 100;

	if (this->systemData->HalconEnable)
	{
		//	save by halcon
	}
	else if (this->systemData->MilEnable)
	{
		//	save by mil
		MIL_ID scanImage = M_NULL;
		MIL_ID sourceImage = M_NULL;

		MIL_INT scanWidth;
		MIL_INT scanHeight;
		MIL_INT sourceWidth;
		MIL_INT sourceHeight;

		switch (this->imageDirection)
		{
		case ImageDirectionType::HORIZONTAL:
			scanWidth = this->imageWidth * this->imageCount * resize / 100;
			scanHeight = this->imageHeight * resize / 100;

			sourceWidth = this->imageWidth * this->imageCount;
			sourceHeight = this->imageHeight;
			break;

		case ImageDirectionType::VERTICAL:
		default:
			scanWidth = this->imageWidth * resize / 100;
			scanHeight = this->imageHeight * this->imageCount * resize / 100;

			sourceWidth = this->imageWidth;
			sourceHeight = this->imageHeight * this->imageCount;
			break;
		}

		scanImage = MbufAllocColor(
			this->systemData->MilSystemData->System,
			this->Channels,
			scanWidth,
			scanHeight,
			this->imageBit + M_UNSIGNED,
			M_IMAGE + M_PROC,
			M_NULL);
		MbufClear(scanImage, 0);

		for (int band = 0; band < this->Channels; band++)
		{
			sourceImage = this->GetMDataID(type, band);
			
			MbufTransfer(
				sourceImage,
				scanImage,
				0,
				0,
				sourceWidth,
				sourceHeight,
				0,
				0,
				0,
				scanWidth,
				scanHeight,
				band,
				M_COPY + M_SCALE,
				M_DEFAULT,
				M_NULL,
				M_NULL);
		}

		this->MilSaveImage(fullFileName, scanImage);

		MbufFree(scanImage);
	}
	else
	{
		//	save by bitmap
	}
}

void ImageData::SaveSliceImage(String^ fullFileName, String^ type, int sliceIndex)
{
	if (this->systemData->HalconEnable)
	{
		//	save by halcon
	}
	else if (this->systemData->MilEnable)
	{
		//	save by mil
		MIL_ID sliceImage = M_NULL;
		MIL_ID sourceImage = M_NULL;

		MIL_INT sliceWidth;
		MIL_INT sliceHeight;
		MIL_INT sourceWidth;
		MIL_INT sourceHeight;

		sourceWidth = sliceWidth = this->imageWidth;
		sourceHeight = sliceHeight = this->imageHeight;

		sliceImage = MbufAllocColor(
			this->systemData->MilSystemData->System,
			this->Channels,
			sliceWidth,
			sliceHeight,
			this->imageBit + M_UNSIGNED,
			M_IMAGE + M_PROC,
			M_NULL);
		MbufClear(sliceImage, 0);

		for (int band = 0; band < this->Channels; band++)
		{
			sourceImage = this->GetMDataID(type, band, sliceIndex);
			
			MbufTransfer(
				sourceImage,
				sliceImage,
				0,
				0,
				sourceWidth,
				sourceHeight,
				0,
				0,
				0,
				sliceWidth,
				sliceHeight,
				band,
				M_COPY + M_SCALE,
				M_DEFAULT,
				M_NULL,
				M_NULL);
		}

		this->MilSaveImage(fullFileName, sliceImage);

		MbufFree(sliceImage);
	}
	else
	{
		//	save by bitmap
	}
}

void ImageData::SaveSmallImage(
	String^ fullFileName,
	String^ type,
	Rectangle zone,
	int sliceIndex)
{
	Point point;

	point.X = -1;
	point.Y = -1;

	if (zone.X < 0)
		throw gcnew Exception("zone X < 0");

	if (zone.X > this->imageWidth - 1)
		throw gcnew Exception("zone X > imageWidth - 1");

	if (zone.Y < 0)
		throw gcnew Exception("zone Y < 0");

	if (zone.Y > this->imageHeight - 1)
		throw gcnew Exception("zone Y > imageHeight - 1");

	if (zone.Width < 0)
		throw gcnew Exception("zone Width < 0");

	if ((zone.Width + zone.X) > this->imageWidth)
		throw gcnew Exception("(zone.Width + zone.X) > imageWidth");

	if (zone.Height < 0)
		throw gcnew Exception("zone Height < 0");

	if ((zone.Height + zone.Y) > this->imageHeight)
		throw gcnew Exception("(zone.Height + zone.Y) > imageHeight");

	if (sliceIndex < 0)
		throw gcnew Exception("sliceIndex < 0");

	if (sliceIndex > this->imageCount - 1)
		throw gcnew Exception("sliceIndex > imageCount - 1");

	if (this->systemData->HalconEnable)
	{
		//	save by halcon
	}
	else if (this->systemData->MilEnable)
	{
		//	save by mil
		MIL_ID smallImage = M_NULL;
		MIL_ID sourceImage = M_NULL;

		smallImage = MbufAllocColor(
			this->systemData->MilSystemData->System,
			this->Channels,
			zone.Width,
			zone.Height,
			this->imageBit + M_UNSIGNED,
			M_IMAGE + M_PROC,
			M_NULL);
		MbufClear(smallImage, 0);

		for (int band = 0; band < this->Channels; band++)
		{
			sourceImage = this->GetMDataID(type, band, sliceIndex);

			MbufTransfer(
				sourceImage,
				smallImage,
				zone.X,
				zone.Y,
				zone.Width,
				zone.Height,
				0,
				0,
				0,
				zone.Width,
				zone.Height,
				band,
				M_COPY + M_SCALE,
				M_DEFAULT,
				M_NULL,
				M_NULL);
		}
		
		this->MilSaveImage(fullFileName, smallImage);

		MbufFree(smallImage);
	}
	else
	{
		//	save by bitmap
	}
}

void ImageData::SaveAlignShiftImage(
	String^ fullFileName,
	String^ type,
	Rectangle zone,
	int sliceIndex,
	Point corss,
	String^ borderColor,
	int resize)
{
	if (resize < 1 || resize > 100) resize = 100;

	if (zone.X < 0)
		throw gcnew Exception("zone X < 0");

	if (zone.X > this->imageWidth - 1)
		throw gcnew Exception("zone X > imageWidth - 1");

	if (zone.Y < 0)
		throw gcnew Exception("zone Y < 0");

	if (zone.Y > this->imageHeight - 1)
		throw gcnew Exception("zone Y > imageHeight - 1");

	if (zone.Width < 0)
		throw gcnew Exception("zone Width < 0");

	if ((zone.Width + zone.X) > this->imageWidth)
		throw gcnew Exception("(zone.Width + zone.X) > imageWidth");

	if (zone.Height < 0)
		throw gcnew Exception("zone Height < 0");

	if ((zone.Height + zone.Y) > this->imageHeight)
		throw gcnew Exception("(zone.Height + zone.Y) > imageHeight");

	if (sliceIndex < 0)
		throw gcnew Exception("sliceIndex < 0");

	if (sliceIndex > this->imageCount - 1)
		throw gcnew Exception("sliceIndex > imageCount - 1");

	corss.X = corss.X * resize / 100;
	corss.Y = corss.Y * resize / 100;

	if (this->systemData->HalconEnable)
	{
		//	save by halcon
	}
	else if (this->systemData->MilEnable)
	{
		//	save by mil
		MIL_ID smallImage = M_NULL;
		MIL_ID sourceImage = M_NULL;

		MIL_INT scanWidth;
		MIL_INT scanHeight;

		scanWidth = zone.Width * resize / 100;
		scanHeight = zone.Height * resize / 100;

		smallImage = MbufAllocColor(
			this->systemData->MilSystemData->System,
			3,
			scanWidth,
			scanHeight,
			this->imageBit + M_UNSIGNED,
			M_IMAGE + M_PROC,
			M_NULL);
		MbufClear(smallImage, 0);

		for (int band = 0; band < 3; band++)
		{
			int channelIndex = band;

			if (channelIndex >= this->Channels)
				channelIndex = this->Channels - 1;
			sourceImage = this->GetMDataID(type, channelIndex, sliceIndex);

			MbufTransfer(
				sourceImage,
				smallImage,
				zone.X,
				zone.Y,
				zone.Width,
				zone.Height,
				0,
				0,
				0,
				scanWidth,
				scanHeight,
				band,
				M_COPY + M_SCALE,
				M_DEFAULT,
				M_NULL,
				M_NULL);
		}

		Rectangle border;
		border.X = 0;
		border.Y = 0;
		border.Width = (int) scanWidth - 1;
		border.Height = (int) scanHeight - 1;
		this->MilDrawRectangle(smallImage, border, borderColor);
		this->MilDrawCross(smallImage, corss);
		
		this->MilSaveImage(fullFileName, smallImage);

		MbufFree(smallImage);
	}
	else
	{
		//	save by bitmap
	}
}

void ImageData::SaveAlignLocateImage(
	String^ fullFileName,
	String^ type,
	Rectangle zone,
	int sliceIndex,
	Point location,
	Point direction,
	array<bool>^ checkZone,
	String^ borderColor,
	int resize)
{
	if (resize < 1 || resize > 100) resize = 100;

	if (zone.X < 0)
		throw gcnew Exception("zone X < 0");

	if (zone.X > this->imageWidth - 1)
		throw gcnew Exception("zone X > imageWidth - 1");

	if (zone.Y < 0)
		throw gcnew Exception("zone Y < 0");

	if (zone.Y > this->imageHeight - 1)
		throw gcnew Exception("zone Y > imageHeight - 1");

	if (zone.Width < 0)
		throw gcnew Exception("zone Width < 0");

	if ((zone.Width + zone.X) > this->imageWidth)
		throw gcnew Exception("(zone.Width + zone.X) > imageWidth");

	if (zone.Height < 0)
		throw gcnew Exception("zone Height < 0");

	if ((zone.Height + zone.Y) > this->imageHeight)
		throw gcnew Exception("(zone.Height + zone.Y) > imageHeight");

	if (sliceIndex < 0)
		throw gcnew Exception("sliceIndex < 0");

	if (sliceIndex > this->imageCount - 1)
		throw gcnew Exception("sliceIndex > imageCount - 1");

	location.X = location.X * resize / 100;
	location.Y = location.Y * resize / 100;

	if (this->systemData->HalconEnable)
	{
		//	save by halcon
	}
	else if (this->systemData->MilEnable)
	{
		//	save by mil
		MIL_ID smallImage = M_NULL;
		MIL_ID sourceImage = M_NULL;

		MIL_INT scanWidth;
		MIL_INT scanHeight;

		scanWidth = zone.Width * resize / 100;
		scanHeight = zone.Height * resize / 100;

		smallImage = MbufAllocColor(
			this->systemData->MilSystemData->System,
			3,
			scanWidth,
			scanHeight,
			this->imageBit + M_UNSIGNED,
			M_IMAGE + M_PROC,
			M_NULL);
		MbufClear(smallImage, 0);

		for (int band = 0; band < 3; band++)
		{
			int channelIndex = band;

			if (channelIndex >= this->Channels)
				channelIndex = this->Channels - 1;
			sourceImage = this->GetMDataID(type, channelIndex, sliceIndex);
			
			MbufTransfer(
				sourceImage,
				smallImage,
				zone.X,
				zone.Y,
				zone.Width,
				zone.Height,
				0,
				0,
				0,
				scanWidth,
				scanHeight,
				band,
				M_COPY + M_SCALE,
				M_DEFAULT,
				M_NULL,
				M_NULL);
		}

		Rectangle border;
			
		MIL_INT checkZoneWidth = Convert::ToInt32(scanWidth * 0.053);
		MIL_INT checkZoneHeight = Convert::ToInt32(scanHeight * 0.053);

		if (checkZoneWidth < 1) checkZoneWidth = 1;
		if (checkZoneHeight < 1) checkZoneHeight = 1;

		if (checkZone->Length == 24)
		{
			for (int cnt = 0; cnt < 6; cnt++)
			{
				border.X = 0;
				border.Y = (int)checkZoneHeight * cnt * 3;
				border.Width = (int)checkZoneWidth;
				border.Height = (int)checkZoneHeight;

				if (checkZone[cnt])
					this->MilDrawRectangle(smallImage, border, "GREEN");
				else
					this->MilDrawRectangle(smallImage, border, "RED");
			}

			for (int cnt = 0; cnt < 6; cnt++)
			{
				border.X = Convert::ToInt32(checkZoneWidth * cnt * 3);
				border.Y = Convert::ToInt32(scanHeight - checkZoneHeight);
				border.Width = (int)checkZoneWidth;
				border.Height = (int)checkZoneHeight;

				if (checkZone[cnt + 6])
					this->MilDrawRectangle(smallImage, border, "GREEN");
				else
					this->MilDrawRectangle(smallImage, border, "RED");
			}

			for (int cnt = 0; cnt < 6; cnt++)
			{
				border.X = Convert::ToInt32(scanWidth - checkZoneWidth);
				border.Y = Convert::ToInt32(
					scanHeight - checkZoneHeight - (checkZoneHeight * cnt * 3));
				border.Width = (int)checkZoneWidth;
				border.Height = (int)checkZoneHeight;

				if (checkZone[cnt + 12])
					this->MilDrawRectangle(smallImage, border, "GREEN");
				else
					this->MilDrawRectangle(smallImage, border, "RED");
			}

			for (int cnt = 0; cnt < 6; cnt++)
			{
				border.X = Convert::ToInt32(
					scanWidth - checkZoneWidth - (checkZoneWidth * cnt * 3));
				border.Y = 0;
				border.Width = (int)checkZoneWidth;
				border.Height = (int)checkZoneHeight;

				if (checkZone[cnt + 18])
					this->MilDrawRectangle(smallImage, border, "GREEN");
				else
					this->MilDrawRectangle(smallImage, border, "RED");
			}
		}

		Point start;
		Point end;
		if (location.X > 0 && location.Y > 0)
		{
			//	draw x line
			if (direction.Y > 0)
			{
				start.X = location.X;
				start.Y = location.Y;
				end.X = location.X;
				end.Y = (int)scanHeight - 1;
				this->MilDrawLine(smallImage, start, end, "GREEN");

				start.X = location.X + direction.X * 20;
				start.Y = location.Y;
				end.X = location.X + direction.X * 20;
				end.Y = (int)scanHeight - 1;
				this->MilDrawLineDots(smallImage, start, end, 20, "GREEN");
			}
			else if (direction.Y < 0)
			{
				start.X = location.X;
				start.Y = 0;
				end.X = location.X;
				end.Y = location.Y;
				this->MilDrawLine(smallImage, start, end, "GREEN");

				start.X = location.X + direction.X * 20;
				start.Y = 0;
				end.X = location.X + direction.X * 20;
				end.Y = location.Y;
				this->MilDrawLineDots(smallImage, start, end, 20, "GREEN");
			}
			
			if (direction.X > 0)
			{
				//	draw y line
				start.X = location.X;
				start.Y = location.Y;
				end.X = (int)scanWidth - 1;
				end.Y = location.Y;
				this->MilDrawLine(smallImage, start, end, "GREEN");

				start.X = location.X;
				start.Y = location.Y + direction.Y * 20;
				end.X = (int)scanWidth - 1;
				end.Y = location.Y + direction.Y * 20;
				this->MilDrawLineDots(smallImage, start, end, 20, "GREEN");
			}
			else if (direction.X < 0)
			{
				//	draw y line
				start.X = 0;
				start.Y = location.Y;
				end.X = location.X;
				end.Y = location.Y;
				this->MilDrawLine(smallImage, start, end, "GREEN");

				start.X = 0;
				start.Y = location.Y + direction.Y * 20;
				end.X = location.X;
				end.Y = location.Y + direction.Y * 20;
				this->MilDrawLineDots(smallImage, start, end, 20, "GREEN");
			}
			
		}
		else if (location.X > 0)
		{
			//	draw x line
			start.X = location.X;
			start.Y = 0;
			end.X = location.X;
			end.Y = (int)scanHeight - 1;
			this->MilDrawLine(smallImage, start, end, "GREEN");

			start.X = location.X + direction.X * 20;
			start.Y = 0;
			end.X = location.X + direction.X * 20;
			end.Y = (int)scanHeight - 1;
			this->MilDrawLineDots(smallImage, start, end, 20, "GREEN");
		}
		else if (location.Y > 0)
		{
			//	draw y line
			start.X = 0;
			start.Y = location.Y;
			end.X = (int)scanWidth - 1;
			end.Y = location.Y;
			this->MilDrawLine(smallImage, start, end, "GREEN");

			start.X = 0;
			start.Y = location.Y + direction.Y * 20;
			end.X = (int)scanWidth - 1;
			end.Y = location.Y + direction.Y * 20;
			this->MilDrawLineDots(smallImage, start, end, 20, "GREEN");
		}

		border.X = 0;
		border.Y = 0;
		border.Width = (int)scanWidth - 1;
		border.Height = (int)scanHeight - 1;
		this->MilDrawRectangle(smallImage, border, borderColor);
		
		this->MilSaveImage(fullFileName, smallImage);

		MbufFree(smallImage);
	}
	else
	{
		//	save by bitmap
	}
}

void ImageData::AllocateData()
{
	if (this->isAllocated) return;

	//	Pointer Data Allocate
	size_t imageSize = (size_t)this->imageWidth * (size_t)this->imageHeight * (size_t)this->imageCount;
	this->PData = gcnew array<ImagePtrData^>(this->imageChannel);
	for (int cnt = 0; cnt < this->imageChannel; cnt++)
	{
		this->PData[cnt] = gcnew ImagePtrData(
			imageSize, this->imageBit, this->extraCount,
			this->systemData->CudaEnable);
	}

	//	Mil Data Link
	if (this->systemData->MilEnable)
	{
		this->MData = gcnew array<ImageMilData^>(this->imageChannel);

		for (int cnt = 0; cnt < this->imageChannel; cnt++)
		{
			this->MData[cnt] = gcnew ImageMilData(
				this->imageWidth,
				this->imageHeight,
				this->imageCount,
				this->imageBit,
				this->imageDirection,
				this->extraCount);

			this->MData[cnt]->CreateMemory(
				this->systemData->MilSystemData->System,
				this->PData[cnt]);
		}
	}
	else
	{
		this->MData = nullptr;
	}

	//	Halcon Data link
	if (this->systemData->HalconEnable)
	{
	}
	else
	{
	}

	this->isAllocated = true;
}

void ImageData::MilSaveImage(String^ fullFileName, MIL_ID image)
{
	if (!this->systemData->MilEnable)
	{
		throw gcnew Exception("MilSaveImage: Mil not Enable.");
	}

	//	String^ to char *
	char * tmpPath;
	IntPtr tmpString = Marshal::StringToHGlobalAnsi(fullFileName);
	tmpPath = (char *)tmpString.ToPointer();
	String^ extension = System::IO::Path::GetExtension(fullFileName);
	if (extension == ".bmp")
		MbufExport(tmpPath, M_BMP, image);
	else if (extension == ".png")
		MbufExport(tmpPath, M_PNG, image);
	else
		MbufExport(tmpPath, M_TIFF, image);
	Marshal::FreeHGlobal(tmpString);
}

void ImageData::MilDrawCross(MIL_ID image, Point center)
{
	if (!this->systemData->MilEnable)
	{
		throw gcnew Exception("MilSaveImage: Mil not Enable.");
	}

	if (center.X < 0 && center.Y < 0) return;

	MIL_ID context;
				
	try
	{
		context  = MgraAlloc(
			this->systemData->MilSystemData->System,
			M_NULL);

		MgraColor(context, M_COLOR_GREEN);

		MgraArcFill(context, image, center.X, center.Y - 5, 10, 10, 70, 110);
		MgraArcFill(context, image, center.X, center.Y + 5, 10, 10, 250, 290);
		MgraArcFill(context, image, center.X - 5, center.Y, 10, 10, 160, 200);
		MgraArcFill(context, image, center.X + 5, center.Y, 10, 10, 340, 20);
	}
	catch (Exception^ ex)
	{
		throw ex;
	}
	finally
	{
		MgraFree(context);
	}
}

void ImageData::MilDrawLine(MIL_ID image, Point start, Point end, String^ color)
{
	if (!this->systemData->MilEnable)
	{
		throw gcnew Exception("MilSaveImage: Mil not Enable.");
	}

	if (start.X < 0 && start.Y < 0) return;
	if (end.X < 0 && end.Y < 0) return;

	MIL_ID context;

	try
	{
		context = MgraAlloc(
			this->systemData->MilSystemData->System,
			M_NULL);

		if (color == "RED")
		{
			MgraColor(context, M_COLOR_RED);
		}
		else if (color == "GREEN")
		{
			MgraColor(context, M_COLOR_GREEN);
		}
		else if (color == "BLUE")
		{
			MgraColor(context, M_COLOR_BLUE);
		}
		else if (color == "CYAN")
		{
			MgraColor(context, M_COLOR_CYAN);
		}
		else if (color == "MAGENTA")
		{
			MgraColor(context, M_COLOR_MAGENTA);
		}
		else if (color == "YELLOW")
		{
			MgraColor(context, M_COLOR_YELLOW);
		}
		else if (color == "WHITE")
		{
			MgraColor(context, M_COLOR_WHITE);
		}
		else if (color == "BLACK")
		{
			MgraColor(context, M_COLOR_BLACK);
		}
		else
		{
			return;
		}
				
		for (int cnt = -2; cnt < 5; cnt++)
		{
			MgraLine(context, image,
				start.X + cnt, start.Y + cnt,
				end.X + cnt, end.Y + cnt);
		}

	}
	catch (Exception^ ex)
	{
		throw ex;
	}
	finally
	{
		MgraFree(context);
	}
}

void ImageData::MilDrawLineDots(MIL_ID image, Point start, Point end, int count, String^ color)
{
	if (!this->systemData->MilEnable)
	{
		throw gcnew Exception("MilSaveImage: Mil not Enable.");
	}

	if (start.X < 0 && start.Y < 0) return;
	if (end.X < 0 && end.Y < 0) return;

	MIL_ID context;

	try
	{
		context = MgraAlloc(
			this->systemData->MilSystemData->System,
			M_NULL);

		if (color == "RED")
		{
			MgraColor(context, M_COLOR_RED);
		}
		else if (color == "GREEN")
		{
			MgraColor(context, M_COLOR_GREEN);
		}
		else if (color == "BLUE")
		{
			MgraColor(context, M_COLOR_BLUE);
		}
		else if (color == "CYAN")
		{
			MgraColor(context, M_COLOR_CYAN);
		}
		else if (color == "MAGENTA")
		{
			MgraColor(context, M_COLOR_MAGENTA);
		}
		else if (color == "YELLOW")
		{
			MgraColor(context, M_COLOR_YELLOW);
		}
		else if (color == "WHITE")
		{
			MgraColor(context, M_COLOR_WHITE);
		}
		else if (color == "BLACK")
		{
			MgraColor(context, M_COLOR_BLACK);
		}
		else
		{
			return;
		}

		int shiftX = Math::Abs((end.X - start.X) / count);
		int shiftY = Math::Abs((end.Y - start.Y) / count);

		for (int cnt = 0; cnt < count; cnt++)
		{
			MgraRectFill(context, image, 
				start.X - 5 + (shiftX * cnt),
				start.Y - 5 + (shiftY * cnt),
				start.X + 5 + (shiftX * cnt),
				start.Y + 5 + (shiftY * cnt));
		}
	}
	catch (Exception^ ex)
	{
		throw ex;
	}
	finally
	{
		MgraFree(context);
	}
}

void ImageData::MilDrawRectangle(MIL_ID image, Rectangle rectangle, String^ color)
{
	if (!this->systemData->MilEnable)
	{
		throw gcnew Exception("MilSaveImage: Mil not Enable.");
	}

	MIL_ID context;

	try
	{
		context = MgraAlloc(
			this->systemData->MilSystemData->System,
			M_NULL);

		if (color == "RED")
		{
			MgraColor(context, M_COLOR_RED);
		}
		else if (color == "GREEN")
		{
			MgraColor(context, M_COLOR_GREEN);
		}
		else if (color == "BLUE")
		{
			MgraColor(context, M_COLOR_BLUE);
		}
		else if (color == "CYAN")
		{
			MgraColor(context, M_COLOR_CYAN);
		}
		else if (color == "MAGENTA")
		{
			MgraColor(context, M_COLOR_MAGENTA);
		}
		else if (color == "YELLOW")
		{
			MgraColor(context, M_COLOR_YELLOW);
		}
		else if (color == "WHITE")
		{
			MgraColor(context, M_COLOR_WHITE);
		}
		else if (color == "BLACK")
		{
			MgraColor(context, M_COLOR_BLACK);
		}
		else
		{
			return;
		}

		//	Top
		MgraRectFill(context, image,
			rectangle.Left, rectangle.Top,
			rectangle.Right, rectangle.Top + 3);

		//	Bottom
		MgraRectFill(context, image,
			rectangle.Left, rectangle.Bottom - 3,
			rectangle.Right, rectangle.Bottom);

		//	Left
		MgraRectFill(context, image,
			rectangle.Left, rectangle.Top,
			rectangle.Left + 3, rectangle.Bottom);

		//	Right
		MgraRectFill(context, image,
			rectangle.Right - 3, rectangle.Top,
			rectangle.Right, rectangle.Bottom);
	}
	catch (Exception^ ex)
	{
		throw ex;
	}
	finally
	{
		MgraFree(context);
	}	
}