#include "Stdafx.h"
#include "SliceSetting.h"

using namespace ImageProcessConfig;

SliceSetting::SliceSetting()
{
	this->Index = -1;
	this->Name = "";
	this->ZoneList = gcnew array<ZoneSetting ^>(0);
}

SliceSetting::~SliceSetting()
{
	if (this->ZoneList)
	{
		for (int cnt = 0; cnt < this->ZoneList->Length; cnt++)
		{
			delete this->ZoneList[cnt];
			this->ZoneList[cnt] = nullptr;
		}

		delete this->ZoneList;
		this->ZoneList = nullptr;
	}
}

void SliceSetting::CheckValue(SliceSetting ^ tmpConfig)
{
	this->Name = tmpConfig->Name;

	if (tmpConfig->ZoneList->Length > 0)
		Array::Resize(this->ZoneList, tmpConfig->ZoneList->Length);
	for (int cnt = 0; cnt < this->ZoneList->Length; cnt++)
	{
		if (!this->ZoneList[cnt])
			this->ZoneList[cnt] = gcnew ZoneSetting();

		this->ZoneList[cnt]->CheckValue(tmpConfig->ZoneList[cnt]);

		this->ZoneList[cnt]->Index = cnt;
	}
}

void SliceSetting::CreateZone(int count)
{
	if (count < 1) return;

	Array::Resize(this->ZoneList, count);
	for (int cnt = 0; cnt < count; cnt++)
	{
		if (this->ZoneList[cnt] == nullptr)
			this->ZoneList[cnt] = gcnew ZoneSetting();
	}
}