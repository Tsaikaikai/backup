#include "Stdafx.h"
#include "TaskManager.h"

using namespace ImageProcessor;

using namespace System::Threading;

TaskManager::TaskManager(int PreProcess, int PostProcess, int Output)
{
	this->preProcessCount = PreProcess;
	this->postProcessCount = PostProcess;
	this->outputProcessCount = Output;

	this->taskList = gcnew ArrayList();

	this->waitForNew = gcnew AutoResetEvent(true);
}

TaskManager::~TaskManager()
{
	this->taskList->Clear();
	this->taskList = nullptr;
}

TaskData ^ TaskManager::NewTask()
{
	this->waitForNew->WaitOne();

	TaskData ^ newTask;
	
	newTask = gcnew TaskData(
		this->preProcessCount,
		this->postProcessCount,
		this->outputProcessCount);

	newTask->Index = this->taskSN++;
	this->taskList->Add(newTask);

	this->waitForNew->Set();
	
	return newTask;
}

void TaskManager::FinishTask(TaskData ^ task)
{
	int index = -1;

	for (int cnt = 0; cnt < this->taskList->Count; cnt++)
	{
		TaskData ^ temp = (TaskData ^) this->taskList[cnt];

		if (temp->Index == task->Index)
		{
			index = cnt;
			break;
		}
	}
	if (index == -1) return;

	this->taskList->RemoveAt(index);
}

void TaskManager::ClearTask()
{
	this->taskList->Clear();
}