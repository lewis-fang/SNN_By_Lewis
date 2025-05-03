#pragma once
#include"dataDefine.h"
#include<QFileDialog>
#include<QTextStream>

class MNISTLoader:QWidget
{
	Q_OBJECT
public:
	MNISTLoader();
	~MNISTLoader() ;

	int loadMnst(int TESTNUM, int MNISTDIM1, int MNISTDIM2, int OUTCLASS, float MaxValue);
	float* getImage() { return mnistTEST; }
	float* getImageIndex() { return mnistTESTIndex; }
	float* getImageIndexVector() { return mnistTESTIndexVector; }
	int getImageCnt() { return imagecnt; }
private:
	
	void convertOutIndex2Vector(int OUTCLASS);
	bool hadData;
	float* mnistTEST;
	float* mnistTESTIndex;
	float* mnistTESTIndexVector;
	int imagecnt;
};