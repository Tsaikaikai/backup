#include "Stdafx.h"
#include "ImageProcessorConfig.h"

using namespace ImageProcessConfig;

ImageProcessorConfig::ImageProcessorConfig()
{
	this->Name = "ImageProcessor";

	this->System = gcnew SystemSetting();

	this->ImageList = gcnew array<ImageSetting ^>(0);
	this->RegisterList = gcnew array<RegisterSetting^>(0);
	this->AlignList = gcnew array<AlignSetting ^>(0);
	this->CalibrationList = gcnew array<CalibrationSetting ^>(0);

	this->CaptureList = gcnew array<CaptureSetting ^>(0);
	this->PreProcessList = gcnew array<PreProcessSetting ^>(0);
	this->PostProcessList = gcnew array<PostProcessSetting ^>(0);
	this->OutputList = gcnew array<OutputSetting ^>(0);

	this->LinkList = gcnew array<ProcessorLinkSetting ^>(0);
}

ImageProcessorConfig::~ImageProcessorConfig()
{	
	if (this->ImageList)
	{
		for (int cnt = 0; cnt < this->ImageList->Length; cnt++)
		{
			if (this->ImageList[cnt])
			{
				delete this->ImageList[cnt];
				this->ImageList[cnt] = nullptr;
			}
		}

		delete this->ImageList;
		this->ImageList = nullptr;
	}

	if (this->RegisterList)
	{
		for (int cnt = 0; cnt < this->RegisterList->Length; cnt++)
		{
			if (this->RegisterList[cnt])
			{
				delete this->RegisterList[cnt];
				this->RegisterList[cnt] = nullptr;
			}
		}

		delete this->RegisterList;
		this->RegisterList = nullptr;
	}
	
	if (this->AlignList)
	{
		for (int cnt = 0; cnt < this->AlignList->Length; cnt++)
		{
			if (this->AlignList[cnt])
			{
				delete this->AlignList[cnt];
				this->AlignList[cnt] = nullptr;
			}
		}

		delete this->AlignList;
		this->AlignList = nullptr;
	}

	if (this->CalibrationList)
	{
		for (int cnt = 0; cnt < this->CalibrationList->Length; cnt++)
		{
			if (this->CalibrationList[cnt])
			{
				delete this->CalibrationList[cnt];
				this->CalibrationList[cnt] = nullptr;
			}
		}

		delete this->CalibrationList;
		this->CalibrationList = nullptr;
	}

	if (this->CaptureList)
	{
		for (int cnt = 0; cnt < this->CaptureList->Length; cnt++)
		{
			if (this->CaptureList[cnt])
			{
				delete this->CaptureList[cnt];
				this->CaptureList[cnt] = nullptr;
			}
		}

		delete this->CaptureList;
		this->CaptureList = nullptr;
	}

	if (this->PreProcessList)
	{
		for (int cnt = 0; cnt < this->PreProcessList->Length; cnt++)
		{
			if (this->PreProcessList[cnt])
			{
				delete this->PreProcessList[cnt];
				this->PreProcessList[cnt] = nullptr;
			}
		}

		delete this->PreProcessList;
		this->PreProcessList = nullptr;
	}

	if (this->PostProcessList)
	{
		for (int cnt = 0; cnt < this->PostProcessList->Length; cnt++)
		{
			if (this->PostProcessList[cnt])
			{
				delete this->PostProcessList[cnt];
				this->PostProcessList[cnt] = nullptr;
			}
		}

		delete this->PostProcessList;
		this->PostProcessList = nullptr;
	}

	if (this->OutputList)
	{
		for (int cnt = 0; cnt < this->OutputList->Length; cnt++)
		{
			if (this->OutputList[cnt])
			{
				delete this->OutputList[cnt];
				this->OutputList[cnt] = nullptr;
			}
		}

		delete this->OutputList;
		this->OutputList = nullptr;
	}

	if (this->LinkList)
	{
		for (int cnt = 0; cnt < this->LinkList->Length; cnt++)
		{
			if (this->LinkList[cnt])
			{
				delete this->LinkList[cnt];
				this->LinkList[cnt] = nullptr;
			}
		}

		delete this->LinkList;
		this->LinkList = nullptr;
	}

	if (this->System)
	{
		delete this->System;
		this->System = nullptr;
	}
}

bool ImageProcessorConfig::CheckValue(ImageProcessorConfig ^ tmpConfig)
{
	this->Name = tmpConfig->Name;
	this->System = tmpConfig->System;
	this->CheckFolderExist(this->System->RecipePath);
	this->CheckFolderExist(this->System->ImagePath);
	this->CheckFolderExist(this->System->OutputPath);

	if (tmpConfig->ImageList->Length > 0)
		Array::Resize(this->ImageList, tmpConfig->ImageList->Length);
	for (int cnt = 0; cnt < tmpConfig->ImageList->Length; cnt++)
	{
		if (this->ImageList[cnt] == nullptr)
			this->ImageList[cnt] = gcnew ImageSetting();

		this->ImageList[cnt] = tmpConfig->ImageList[cnt];
	}

	if (tmpConfig->RegisterList->Length > 0)
		Array::Resize(this->RegisterList, tmpConfig->RegisterList->Length);
	for (int cnt = 0; cnt < tmpConfig->RegisterList->Length; cnt++)
	{
		if (this->RegisterList[cnt] == nullptr)
			this->RegisterList[cnt] = gcnew RegisterSetting();

		this->RegisterList[cnt] = tmpConfig->RegisterList[cnt];
	}

	if (tmpConfig->AlignList->Length > 0)
		Array::Resize(this->AlignList, tmpConfig->AlignList->Length);
	for (int cnt = 0; cnt < tmpConfig->AlignList->Length; cnt++)
	{
		if (this->AlignList[cnt] == nullptr)
			this->AlignList[cnt] = gcnew AlignSetting();

		this->AlignList[cnt] = tmpConfig->AlignList[cnt];
	}

	if (tmpConfig->CalibrationList->Length > 0)
		Array::Resize(this->CalibrationList, tmpConfig->CalibrationList->Length);
	for (int cnt = 0; cnt < tmpConfig->CalibrationList->Length; cnt++)
	{
		if (this->CalibrationList[cnt] == nullptr)
			this->CalibrationList[cnt] = gcnew CalibrationSetting();

		this->CalibrationList[cnt] = tmpConfig->CalibrationList[cnt];
	}

	if (tmpConfig->CaptureList->Length > 0)
		Array::Resize(this->CaptureList, tmpConfig->CaptureList->Length);
	for (int cnt = 0; cnt < tmpConfig->CaptureList->Length; cnt++)
	{
		if (this->CaptureList[cnt] == nullptr)
			this->CaptureList[cnt] = gcnew CaptureSetting();

		this->CaptureList[cnt] = tmpConfig->CaptureList[cnt];
	}

	if (tmpConfig->PreProcessList->Length > 0)
		Array::Resize(this->PreProcessList, tmpConfig->PreProcessList->Length);
	for (int cnt = 0; cnt < tmpConfig->PreProcessList->Length; cnt++)
	{
		if (this->PreProcessList[cnt] == nullptr)
			this->PreProcessList[cnt] = gcnew PreProcessSetting();

		this->PreProcessList[cnt] = tmpConfig->PreProcessList[cnt];
	}

	if (tmpConfig->PostProcessList->Length > 0)
		Array::Resize(this->PostProcessList, tmpConfig->PostProcessList->Length);
	for (int cnt = 0; cnt < tmpConfig->PostProcessList->Length; cnt++)
	{
		if (this->PostProcessList[cnt] == nullptr)
			this->PostProcessList[cnt] = gcnew PostProcessSetting();

		this->PostProcessList[cnt] = tmpConfig->PostProcessList[cnt];
	}

	if (tmpConfig->OutputList->Length > 0)
		Array::Resize(this->OutputList, tmpConfig->OutputList->Length);
	for (int cnt = 0; cnt < tmpConfig->OutputList->Length; cnt++)
	{
		if (this->OutputList[cnt] == nullptr)
			this->OutputList[cnt] = gcnew OutputSetting();

		this->OutputList[cnt] = tmpConfig->OutputList[cnt];
	}

	if (tmpConfig->LinkList->Length > 0)
		Array::Resize(this->LinkList, tmpConfig->LinkList->Length);
	for (int cnt = 0; cnt < tmpConfig->LinkList->Length; cnt++)
	{
		if (this->LinkList[cnt] == nullptr)
			this->LinkList[cnt] = gcnew ProcessorLinkSetting();

		this->LinkList[cnt] = tmpConfig->LinkList[cnt];
	}

	//	Read NAME
	this->Name = tmpConfig->Name;

	//	Read UPDATE
	this->Update = tmpConfig->Update;

	//	Read VERSION
	this->Version = gcnew String(this->classVersion);

	if (this->Version != tmpConfig->Version)
		return false;
	else
		return true;
}