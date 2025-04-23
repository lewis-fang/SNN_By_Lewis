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

	int loadMnst();
	float* getImage() { return mnistTEST; }
	float* getImageIndex() { return mnistTESTIndex; }
	float* getImageIndexVector() { return mnistTESTIndexVector; }
	int getImageCnt() { return imagecnt; }
private:
	
	void convertOutIndex2Vector();
	bool hadData;
	float* mnistTEST;
	float* mnistTESTIndex;
	float* mnistTESTIndexVector;
	int imagecnt;
};