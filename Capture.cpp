#include "Stdafx.h"
#include "Capture.h"

using namespace ImageProcessor;
using namespace ImageProcessConfig;

Capture::Capture(
	int id,
	CaptureSetting ^ config,
	SystemData ^ system,
	InfoManager ^ logger,
	TaskManager ^ manager)
{
	//	Check Paramter --------------------------------------------------------
	if (id < 0)
	{
		throw gcnew ArgumentNullException("ID");
	}

	if (config == nullptr)
	{
		throw gcnew ArgumentNullException("Capture Setting");
	}

	if (system == nullptr)
	{
		throw gcnew ArgumentNullException("System Data");
	}

	if (logger == nullptr)
	{
		throw gcnew ArgumentNullException("Logger");
	}
	
	if (manager == nullptr)
	{
		throw gcnew ArgumentNullException("TaskManager");
	}

	//
	this->captureConfig = config;
	this->debugMode = this->captureConfig->DebugMode;

	this->id = id;
	this->name = this->captureConfig->Name;
	this->logger = logger;
	this->taskManager = manager;

	//	
	switch (this->captureConfig->frameGrab->Type)
	{
	case FrameGrabType::MIL_FILELOAD:
		if (!system->MilEnable)
		{
			throw gcnew Exception("Msystem not support.");
		}

		this->frameGrab = gcnew MilLoadImage(
			system->MilSystemData->System,
			this->captureConfig->frameGrab);
		break;

	case FrameGrabType::SEPERA:
		if (!system->MilEnable)
		{
			throw gcnew Exception("Msystem not support.");
		}

		this->frameGrab = gcnew SeperaFrameGrab(
			system->MilSystemData->System,
			this->captureConfig->frameGrab);
		break;

	case FrameGrabType::PYLON:
		if (!system->MilEnable)
		{
			throw gcnew Exception("Msystem not support.");
		}

		this->frameGrab = gcnew PylonFrameGrab(
			system->MilSystemData->System,
			this->captureConfig->frameGrab);
		break;

	case FrameGrabType::MATROX:
		if (!system->MilEnable)
		{
			throw gcnew Exception("Msystem not support.");
		}

		this->frameGrab = gcnew MatroxFrameGrab(
			system->MilSystemData->System,
			this->captureConfig->frameGrab);
		break;

	case FrameGrabType::MATROX_GENICAM:
		if (!system->MilEnable)
		{
			throw gcnew Exception("Msystem not support.");
		}

		this->frameGrab = gcnew MatroxFrameGrabGenICam(
			system->MilSystemData->System,
			this->captureConfig->frameGrab);
		break;

	case FrameGrabType::MATROX_CAMERALINK:
		if (!system->MilEnable)
		{
			throw gcnew Exception("Msystem not support.");
		}

		this->frameGrab = gcnew MatroxFrameGrabCameraLink(
			system->MilSystemData->System,
			this->captureConfig->frameGrab);
		break;
	case FrameGrabType::WEB_CAM:
		if (!system->MilEnable)
		{
			throw gcnew Exception("Msystem not support.");
		}

		this->frameGrab = gcnew WebCamFrameGrab(
			system->MilSystemData->System,
			this->captureConfig->frameGrab);
		break;

	default:
		this->frameGrab = nullptr;
		break;
	}
	this->isContinueCapture = this->captureConfig->IsContinueCapture;
	this->isPauseCapture = false;
	this->isEmergencyStop = false;
	this->isPass = false;

	//	Initial Kernel Thread
	try
	{
		this->isKernelThreadEnd = false;
		this->kernelStartEvent = gcnew AutoResetEvent(false);
		this->KernelReady = gcnew ManualResetEvent(false);
		this->StartGrab = gcnew ManualResetEvent(true);
		this->GrabFinished = gcnew ManualResetEvent(true);

		this->kernelThread = gcnew Thread(
			gcnew ThreadStart(this, &Capture::CaptureKernel));
		this->kernelThread->Name = String::Format("{0}_Kernel", this->name);
		this->kernelThread->SetApartmentState(ApartmentState::STA);
		this->kernelThread->Start();
	}
	catch (Exception^ ex)
	{
		throw ex;
	}

	//
	this->getIndex = -1;
	this->captureData = nullptr;
	this->startData = nullptr;
	this->scanSetting = nullptr;
	this->exposureSetting = nullptr;
	this->functionSetting = nullptr;	
}

Capture::~Capture()
{
	//	End Kernel Thread
	this->isEmergencyStop = true;
	this->isKernelThreadEnd = true;
	this->kernelStartEvent->Set();
	delete this->kernelThread;
	delete this->kernelStartEvent;
	delete this->KernelReady;
	delete this->StartGrab;
	delete this->GrabFinished;

	//
	if (this->frameGrab)
	{
		delete this->frameGrab;
		this->frameGrab = nullptr;
	}

	//
	this->captureData = nullptr;
	this->startData = nullptr;
	this->scanSetting = nullptr;
	this->exposureSetting = nullptr;
	this->functionSetting = nullptr;
}

void Capture::SetImage(ImageData^ data)
{
	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} -----",
			"CAPTURE", this->id,
			"SetImage"));
	}

	try
	{
		//	Check Data --------------------------------------------------------
		if (data == nullptr)
		{
			throw gcnew ArgumentNullException("Capture ImageData.");
		}
		this->captureData = data;

		if (this->frameGrab)
			this->frameGrab->SetImage(this->captureData);
	}
	catch (Exception^ ex)
	{
		throw ex;
	}
	
	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} Finished -----",
			"CAPTURE", this->id,
			"SetImage"));
	}
}

void Capture::SetSetting(
	ImageProcessFunctionConfig ^ function,
	ScanSetting ^ scan,
	ExposureScanSetting ^ exposure)
{
	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} -----",
			"CAPTURE", this->id,
			"SetSetting"));
	}

	try
	{
		if (function == nullptr)
		{
			throw gcnew ArgumentNullException("Capture Function Setting.");
		}
		this->functionSetting = function;

		if (scan == nullptr)
		{
			throw gcnew ArgumentNullException("Capture Scan Setting.");
		}
		this->scanSetting = scan;

		this->exposureSetting = exposure;
	}
	catch (Exception^ ex)
	{
		throw ex;
	}

	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} Finished -----",
			"CAPTURE", this->id,
			"SetSetting"));
	}
}

void Capture::GetImage(StartData ^ data, bool isPass)
{
	//	Start Get Image -------------------------------------------------------
	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} -----",
			"CAPTURE", this->id,
			"Get Image"));
	}

	try
	{
		if (!this->captureData)
		{
			throw gcnew ArgumentNullException("Capture ImageData.");
		}

		this->startData = data;

		this->isPass = isPass;
		this->isPauseCapture = false;
		this->StartGrab->Reset();
		this->GrabFinished->Reset();

		switch (this->captureConfig->Type)
		{
		case CaptureType::SCAN:
		case CaptureType::SLICE:
			this->getIndex = 0;
			break;

		case CaptureType::SLICES:
		default:
			this->getIndex = data->SliceIndex;
			break;
		}
		if (this->getIndex >= this->captureData->Count) this->getIndex = 0;

		this->kernelStartEvent->Set();		
	}
	catch (Exception^ ex)
	{
		throw ex;
	}

	//	Finished --------------------------------------------------------------
	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} Finished -----",
			"CAPTURE", this->id,
			"Get Image"));
	}
}

void Capture::PauseGetImage()
{
	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} -----",
			"CAPTURE", this->id,
			"Pause"));
	}

	try
	{
		this->isPauseCapture = true;
	}
	catch (Exception^ ex)
	{
		throw ex;
	}

	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} Finished -----",
			"CAPTURE", this->id,
			"Pause"));
	}
}

void Capture::Stop()
{
	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} -----",
			"CAPTURE", this->id,
			"Stop"));
	}

	try
	{
		this->isEmergencyStop = true;

		//	Stop frameGrab
		if (this->frameGrab)
			this->frameGrab->StopGrab();
	}
	catch (Exception^ ex)
	{
		throw ex;
	}

	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] {2} Finished -----",
			"CAPTURE", this->id,
			"Stop"));
	}
}

void Capture::CaptureKernel()
{
	while (!this->isKernelThreadEnd)
	{		
		try
		{			
			//	Wait for Start Event ------------------------------------------
			this->KernelReady->Set();
			this->kernelStartEvent->WaitOne();
			if (this->isKernelThreadEnd) break;
			if (this->isEmergencyStop)
			{
				//	Reset emergencyStop
				this->isEmergencyStop = false;
			}
			this->KernelReady->Reset();			

			//	Set Exposure Setting
			if (this->exposureSetting != nullptr)
			{
				ExposureSliceSetting ^ setting;

				if (this->getIndex > this->exposureSetting->SliceList->Length - 1)
				{
					this->logger->Warning(String::Format(
						"[{0,-20} <<{1:000}>>]   Can't find Exposure seeing Slice[{2:000}]",
						"CAPTURE", this->id,
						this->getIndex));
				}
				else
				{
					setting = this->exposureSetting->SliceList[this->getIndex];

					for (int cnt_i = 0; cnt_i < setting->settingList->Length; cnt_i++)
					{
						if (this->debugMode)
						{
							String^ message;

							message = String::Format(
								"[{0,-20} <<{1:000}>>]   Set Exposure seeting {2}",
								"CAPTURE", this->id,
								setting->settingList[cnt_i]->Name);
							for (int cnt_j = 0; cnt_j < setting->settingList[cnt_i]->Parameters->Length; cnt_j++)
							{
								message += String::Format(" {0}",
									setting->settingList[cnt_i]->Parameters[cnt_j]);
							}
							this->logger->Debug(message);
						}

						this->frameGrab->SetExposure(
							setting->settingList[cnt_i]->Name,
							setting->settingList[cnt_i]->Parameters);
					}
				}				
			}
			if (this->isEmergencyStop) continue;

			//	Start grab and raise event ------------------------------------
			if (!this->isPass) this->frameGrab->StartGrab();
			this->StartGrab->Set();	//	Let core to check is start.
			if (this->isEmergencyStop) continue;

			//	Get Image -----------------------------------------------------
			do
			{
				switch (this->captureConfig->Type)
				{
				case CaptureType::SCAN:
					//	new task
					this->taskData = this->taskManager->NewTask();
					this->taskData->isPass = this->isPass;
					this->taskData->ScanRecipe = this->scanSetting;
					this->taskData->FunctionRecipe = this->functionSetting;
					this->taskData->Information = this->startData;
					this->taskData->Image = this->captureData;

					//	capture one scan
					for (int cnt = 0; cnt < this->captureData->Count; cnt++)
					{
						this->CaptureImage(cnt);
						if (this->isEmergencyStop) break;
					}
					break;

				case CaptureType::SLICE:
					//	new task
					this->taskData = this->taskManager->NewTask();
					this->taskData->isPass = this->isPass;
					this->taskData->ScanRecipe = this->scanSetting;
					this->taskData->FunctionRecipe = this->functionSetting;
					this->taskData->Information = this->startData;
					this->taskData->Image = this->captureData;

					//	capture one image
					this->CaptureImage(this->getIndex);

					//	next image index
					this->getIndex += 1;
					this->getIndex %= this->captureData->Count;
					break;

				case CaptureType::SLICES:
					//	new task if need
					if (this->getIndex == 0 || this->taskData == nullptr)
					{
						this->taskData = this->taskManager->NewTask();
						this->taskData->isPass = this->isPass;
						this->taskData->ScanRecipe = this->scanSetting;
						this->taskData->FunctionRecipe = this->functionSetting;
						this->taskData->Information = this->startData;
						this->taskData->Image = this->captureData;
					}

					//	capture one image
					this->CaptureImage(this->getIndex);
					break;

				default:
					break;
				}

			} while (this->isContinueCapture
				&& !this->isPauseCapture
				&& !this->isEmergencyStop);
			if (this->isEmergencyStop) continue;

			//	Stop grab and raise finished event --------------------------------
			if (!this->isPass) this->frameGrab->StopGrab();
			this->GrabFinished->Set();
		}
		catch (Exception^ ex)
		{
			//	log exception
			this->logger->Error(String::Format(
				"[{0,-20} <<{1:000}>>] Kernel Exception: {2}",
				"CAPTURE", this->id, ex->Message));
			this->logger->Debug(String::Format(
				"[{0,-20} <<{1:000}>>] Kernel Exception: {2}",
				"CAPTURE", this->id, ex->ToString()));
		}
	}
}

void Capture::CaptureImage(int index)
{
	if (this->taskData->isCancel)
	{
		if (this->debugMode)
		{
			this->logger->Debug(String::Format(
				"[{0,-20} <<{1:000}>>]   Canceled Capture Image {2:000}",
				"CAPTURE", this->id, index));
		}

		this->logger->Warning(String::Format(
			"[{0,-20} <<{1:000}>>]   Canceled Capture Image {2:000}",
			"CAPTURE", this->id, index));

		//	raise Job finished event
		this->JobFinishEvent(this->taskData, index);
		return;
	}

	if (this->taskData->isPass)
	{
		if (this->debugMode)
		{
			this->logger->Debug(String::Format(
				"[{0,-20} <<{1:000}>>]   Passed Capture Image {2:000}",
				"CAPTURE", this->id, index));
		}

		//	raise Job finished event
		this->JobFinishEvent(this->taskData, index);
		return;
	}

	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>]   Capture Image {2:000}",
			"CAPTURE", this->id, index));
	}

	try
	{
		//	wait to start
		Thread::Sleep(this->captureConfig->WaitTime);

		//	get one image
		if (this->frameGrab)
			this->frameGrab->GetImage(index);

		//	delay to report
		Thread::Sleep(this->captureConfig->DelayTime);
		
	}
	catch (Exception^ ex)
	{
		this->taskData->isCancel = true;
		throw ex;
	}
	finally
	{
		//	rasie to PreProcess
		if (this->isEmergencyStop )
			this->taskData->isCancel = true;

		if (this->debugMode)
		{
			this->logger->Debug(String::Format(
				"[{0,-20} <<{1:000}>>]   Raise job finish event.",
				"CAPTURE", this->id));
		}		
		this->JobFinishEvent(this->taskData, index);
	}
}