#include "Stdafx.h"
#include "Output.h"

using namespace ImageProcessor;

Output::Output()
{
}

Output::Output(
	int id,
	OutputSetting ^ config,
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
		throw gcnew ArgumentNullException("Output Config");
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
	this->processConfig = config;
	this->debugMode = config->DebugMode;

	this->id = id;
	this->name = config->Name;
	this->systemData = system;
	this->logger = logger;
	this->taskManager = manager;

	this->isSaveScan = config->SaveScanImage;
	this->isSaveResult = config->SaveResultImage;
	this->scanRatio = 10;
	this->resultWidth = 100;
	this->resultHeight = 100;
	this->resultFolder = "Defect";
	this->extension = OutputExtensionMode::NONE;
	this->extensionWidth = 0;
	this->extensionHeight = 0;

	ParameterDetail^ tmpParameter = gcnew ParameterDetail();
	for (int cnt = 0; cnt < config->Parameters->Length; cnt++)
	{
		try
		{
			tmpParameter->ReadString(config->Parameters[cnt]);
			
			if (tmpParameter->Name == "ResultImage")
			{
				this->resultWidth = Convert::ToInt32(tmpParameter->Value[0]);
				this->resultHeight = Convert::ToInt32(tmpParameter->Value[1]);

				if (this->resultWidth < 0)
				{
					throw gcnew Exception("Output: ResultImage Width should > 0");
				}

				if (this->resultHeight < 0)
				{
					throw gcnew Exception("Output: ResultImage Height should > 0");
				}
			}
			else if (tmpParameter->Name == "ScanRatio")
			{
				this->scanRatio = Convert::ToInt32(tmpParameter->Value[0]);

				if (this->scanRatio < 1 || this->scanRatio > 100)
				{
					throw gcnew Exception("Output: ScanRatio should > 0 and < 101");
				}
			}
			else
			{
				this->logger->Warning(String::Format(
					"Output: Unknow Setting. [{0}]",
					tmpParameter->Name));
			}
		}
		catch (Exception^ ex)
		{
			throw gcnew Exception(String::Format(
				"Output: parameter - {0}",
				ex->Message));
		}
	}
}

Output::~Output()
{	
}

void Output::AddJob(TaskData ^ taskData, int jobIndex, int postIndex)
{
	//	Check data ------------------------------------------------------------
	if (taskData == nullptr)
	{
		throw gcnew ArgumentNullException("Output TaskData");
	}
	if (postIndex >= taskData->ResultList->Length || postIndex < 0)
	{
		throw gcnew ArgumentException("Output PostIndex");
	}
	
	//	Add in Job Queue
	try
	{
		taskData->Travel->Output[this->id] = TaskTravelState::ARRIVAL;

		this->DoJob(taskData , jobIndex, postIndex);
	}
	catch (Exception^ ex)
	{
		//	log exception
		this->logger->Error(String::Format(
			"[{0,-20} <<{1:000}>>] Kernel Exception: {2}",
			"OUTPUT", this->id, ex->Message));
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>] Kernel Exception: {2}",
			"OUTPUT", this->id, ex->ToString()));

		//	Cancel Task
		taskData->isCancel = true;
	}
	finally
	{
		this->logger->General(String::Format(
			"Output Finished [{0}]", this->name));

		taskData->Travel->Output[this->id] = TaskTravelState::DEPART;

		if (taskData->Travel->CheckTaskJourney() == TaskJourney::FINISH)
			this->taskManager->FinishTask(taskData);
	}
}

void Output::DoJob(TaskData^ taskData, int jobIndex, int postIndex)
{
	if (taskData->isCancel)
	{
		if (this->debugMode)
		{
			this->logger->Debug(String::Format(
				"[{0,-20} <<{1:000}>>]   Canceled Task[{2}]",
				"OUTPUT", this->id, taskData->Index));
		}

		this->logger->Warning(String::Format(
			"[{0,-20} <<{1:000}>>]   Canceled Task[{2}]",
			"OUTPUT", this->id, taskData->Index));

		return;
	}

	if (taskData->isPass)
	{
		if (this->debugMode)
		{
			this->logger->Debug(String::Format(
				"[{0,-20} <<{1:000}>>]   Passed Task[{2}]",
				"OUTPUT", this->id, taskData->Index));
		}

		return;
	}

	if (this->debugMode)
	{
		this->logger->Debug(String::Format(
			"[{0,-20} <<{1:000}>>]   Task[{2}]",
			"OUTPUT", this->id, taskData->Index));
	}
}

void Output::SaveScanImage(String^ fileName, ImageData^ image)
{
	try
	{
		if (this->debugMode)
		{
			this->logger->Debug(String::Format(
				"[{0,-20} <<{1:000}>>]     SaveScanImage: Save scan image.",
				"OUTPUT", this->id));
		}
		
		image->SaveScanImage(fileName, "SOURCE", this->scanRatio);
	}
	catch (Exception^ ex)
	{
		this->logger->Error(String::Format(
			"[{0,-20} <<{1:000}>>] Save Scan Image Error. {2}",
			"OUTPUT", this->id, ex->Message));
		this->logger->Debug(ex->ToString());
	}
}

void Output::SaveResultImage(String^ fileName, ImageData^ image, Rectangle zone, int index)
{
	try
	{
		image->SaveSmallImage(fileName, "SOURCE", zone, index);
	}
	catch (Exception^ ex)
	{
		this->logger->Error(String::Format(
			"[{0,-20} <<{1:000}>>] Save Result Image Error. {2}",
			"OUTPUT", this->id, ex->Message));
		this->logger->Debug(ex->ToString());
	}
}

void Output::SaveAlignImage(
	String^ fileName,
	ImageData^ image,
	Rectangle zone,
	int index,
	Point cross,
	String^ borderColor,	
	int resize)
{
	try
	{
		image->SaveAlignShiftImage(fileName, "SOURCE", zone, index,
			cross, borderColor, resize);
	}
	catch (Exception^ ex)
	{
		this->logger->Error(String::Format(
			"[{0,-20} <<{1:000}>>] Save Result Image Error. {2}",
			"OUTPUT", this->id, ex->Message));
		this->logger->Debug(ex->ToString());
	}
}

void Output::SaveAlignImage(
	String^ fileName,
	ImageData^ image,
	Rectangle zone,
	int index,	
	Point location,
	Point direction,
	array<bool>^ checkZone,
	String^ borderColor,	
	int resize)
{
	try
	{
		image->SaveAlignLocateImage(fileName, "SOURCE", zone, index,
			location, direction, checkZone, borderColor, resize);
	}
	catch (Exception^ ex)
	{
		this->logger->Error(String::Format(
			"[{0,-20} <<{1:000}>>] Save Result Image Error. {2}",
			"OUTPUT", this->id, ex->Message));
		this->logger->Debug(ex->ToString());
	}
}