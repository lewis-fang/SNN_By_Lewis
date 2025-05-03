#pragma once
#define NULLCOLLUMN 5
#define NULLHEADERROW 1
#define NULLPATITIONROW 3
#define BADPOINT 4

#include"dataDefine.h"
#include<QFileDialog>
#include<QTextStream>

class BioDataLoader :QWidget
{
	Q_OBJECT
public:
	BioDataLoader();
	~BioDataLoader();

	int loadBioData(int TESTNUM, int MNISTDIM1, int MNISTDIM2, float MaxValue);
	int loadBioDataIndex(int TESTNUM,int OUTCLASS);
	float* getImage() { return Data; }
	float* getImageIndex() { return DataIndex; }
	int getImageCnt() { return imagecnt; }
private:

	bool hadData;
	float* Data;
	float* DataIndex;
	int imagecnt;
};