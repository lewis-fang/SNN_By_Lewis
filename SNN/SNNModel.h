#pragma once
#include"dataDefine.h"
#include"SNNLayer.h"
#include <string.h> 
#include<QtCharts/qchartview.h>

class snnModel
{
public:
	snnModel() ;
	~snnModel() {};

public:
	static bool latency(float* imageInput, size_t length, float* spikeInput, size_t T, float tao, float Vthr);
	void buildMyDefaultSNNModel();

	void fowardRecurrentSpikingSimd(tensor ts,int b);
	void setInput(float* totalImg, float* ideal, int imgNum,int blockLength,int outLength);
	void setBatchsize(int bxs) { batchSize = bxs; };
	void setLearnRate(float lr) { learnRate = lr; };
	void setMinLoss(float ml) { minLoss = ml; };
	void setDiffMinLoss(float dml) { diffLoss = dml; };
	void setWeightSD(float initialsd) { weitInitialsd = initialsd; };
	void setBeta(float bt) { beta = bt; };
	void setUthr(float u) { Uthr = u; };
	void setReset(int rst) { reset = rst; };
	void setMaxEpoch(int maxepo) { maxEpoch = maxepo; };
	
	tensor getOutMem() { return mySNNStructure.back().getOutMem(); };
	tensor getOut() { return mySNNStructure.back().getOut(); };
	SNNLayer getSNNLayer(int i) { return mySNNStructure.at(i); };

	void createTrainThread();
private:
	std::vector<SNNLayer> mySNNStructure;
	std::vector<tensor> InputImageSeries;
	std::vector<float*> idealOut;
	float calculateC(float* actualy, float* idealx, int sz);
	template< typename  T>
	void dcalculateC(T* idealx, T* actualy, T* dyVdx, int sz);
	int lossType;
	int maxEpoch;
	float minLoss;
	float diffLoss;
	float learnRate;
	int batchSize;
	std::vector<float> vloss;

	float weitInitialsd;
	float beta;
	float Uthr;
	int reset;
	void train();
	volatile int parrellelIBatchTrainDone;
	std::condition_variable condC;
	std::mutex myMutexC;
	void updateLossParrallel(tensor bImage, float* vC, int outLen, int realIndex, int b);
	Optimizer mOptimizer;
};