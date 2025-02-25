#pragma once

#include "ImageProcessor.h"
#include "TaskManager.h"

using namespace System;
using namespace System::Threading;

using namespace CommonBase::Logger;
using namespace ImageProcessConfig;
using namespace ImageProcessData;
using namespace FrameGrabber;

namespace ImageProcessor
{
	public ref class Capture
	{
	private:
		CaptureSetting ^ captureConfig;
		bool debugMode;

		int id;
		String^ name;
		InfoManager ^ logger;
		TaskManager ^ taskManager;

		IFrameGrab ^ frameGrab;
		bool isContinueCapture;		
		bool isPauseCapture;		
		bool isEmergencyStop;
		bool isPass;

		Thread ^ kernelThread;
		bool isKernelThreadEnd;
		AutoResetEvent ^ kernelStartEvent;		

		int getIndex;
		TaskData ^ taskData;
		ImageData^ captureData;
		StartData ^ startData;
		ScanSetting ^ scanSetting;
		ExposureScanSetting ^ exposureSetting;
		ImageProcessFunctionConfig ^ functionSetting;		

	public:
		event JobFinishEventHandler ^ JobFinishEvent;

		ManualResetEvent^ KernelReady;

		ManualResetEvent ^ StartGrab;
		ManualResetEvent ^ GrabFinished;

	public:
		Capture(
			int id,
			CaptureSetting ^ config,
			SystemData ^ system,
			InfoManager ^ logger,
			TaskManager ^ manager);

	protected:
		~Capture();

	public:
		void SetImage(ImageData^ data);
		void SetSetting(ImageProcessFunctionConfig ^ function, ScanSetting ^ scan, ExposureScanSetting ^ exposure);
		void GetImage(StartData ^ data, bool isPass);
		void PauseGetImage();

		void Stop();

	private:
		void CaptureKernel();
		void CaptureImage(int index);
	};
}