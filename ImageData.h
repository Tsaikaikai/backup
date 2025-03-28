#pragma once
#include "SystemData.h"
#include "ImagePtrData.h"
#include "ImageMilData.h"

using namespace System;
using namespace System::Drawing;
using namespace System::Runtime::InteropServices;

using namespace ImageProcessConfig;

namespace ImageProcessData
{
	public ref class ImageData
	{
	public:
		property String^ Name
		{
			String^ get()
			{
				return this->dataName;
			}
		}

		property int Index
		{
			int get()
			{
				return this->index;
			}
		}

		property int Width
		{
			int get()
			{
				return this->imageWidth;
			}
		}

		property int Height
		{
			int get()
			{
				return this->imageHeight;
			}
		}

		property int Channels
		{
			int get()
			{
				return this->imageChannel;
			}
		}

		property int Bits
		{
			int get()
			{
				return this->imageBit;
			}
		}

		property int Count
		{
			int get()
			{
				return this->imageCount;
			}
		}

		property ImageDirectionType Direction
		{
			ImageDirectionType get()
			{
				return this->imageDirection;
			}
		}

		property int Extra
		{
			int get()
			{
				return this->extraCount;
			}
		}

		property bool Allocated
		{
			bool get()
			{
				return this->isAllocated;
			}
		}

		//	Ptr Data
		array<ImagePtrData^>^ PData;

		//	M-Sys Data
		array<ImageMilData^>^ MData;

		//	H-Sys Data
		
	private:
		SystemData ^ systemData;
		bool isAllocated;

		int index;
		String^ dataName;
		bool fixable;
		int imageWidth;
		int imageHeight;
		int imageBit;
		int imageChannel;
		int imageCount;
		ImageDirectionType imageDirection;
		int extraCount;

	public:
		ImageData(ImageSetting^ setting, SystemData^ data, int index);

	protected:
		~ImageData();

	public:
		void AllocateData(ImageSetting^ setting);
		void ClearData();
		void FreeData();

		void* GetImagePtr(String^ name, int channel, int slice);
		void* GetImagePtr(String^ name, int channel);
		long long GetMDataID(String^ name, int channel, int slice);
		long long GetMDataID(String^ name, int channel);
		String^ GetImageName(String^ name);

		void SaveScanImage(String^ fullFileName, String^ type, int resize);
		void SaveSliceImage(String^ fullFileName, String^ type, int sliceIndex);
		void SaveSmallImage(String^ fullFileName, String^ type, Rectangle zone, int sliceIndex);

		void SaveAlignShiftImage(
			String^ fullFileName, String^ type, Rectangle zone, int sliceIndex,
			Point corss, String^ borderColor, int resize);
		void SaveAlignLocateImage(
			String^ fullFileName, String^ type, Rectangle zone, int sliceIndex,
			Point location, Point direction, array<bool>^ checkZone,
			String^ borderColor, int resize);

	private:
		void AllocateData();

		void MilSaveImage(String^ fullFileName, MIL_ID image);
		void MilDrawCross(MIL_ID image, Point center);
		void MilDrawLine(MIL_ID image, Point start, Point end, String^ color);
		void MilDrawLineDots(MIL_ID image, Point start, Point end, int count, String^ color);
		void MilDrawRectangle(MIL_ID image, Rectangle rectangle, String^ color);
	};
}