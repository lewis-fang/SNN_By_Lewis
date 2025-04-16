#pragma once
#include"dataDefine.h"
#include"spikeNeuro.h"
class SNNLayer
{
public:
	SNNLayer();
	~SNNLayer();
	bool initSnnLayer(dim inputDim, dim outDim,float beta,float Uthr,int resetMethod,float sd, float mu, int batchSize);

	void setAct(int act) { actFun = act; };
	void setPreLayer(SNNLayer* pre) { preLayer = pre; };
	void setPostLayer(SNNLayer* post) { postLayer = post; };
	void setHiddenNumth(int hdnm) { hiddenNumth = hdnm; };
	bool setIn(tensor ts, int t, int b1, int b2);
	void setSoftmaxOut(bool so) { isSoftmaxOut = so; }
	void layerCalcSimd(int t, int b);
	void setIdealOut(float* tsData, int b);
	void setT(int T) { TIMESTEP = T; }
	
	tensor getW() { return W; };
	tensor getdW() { return dCWTotal; };
	float* getB() { return biasSimd.getData(); };
	float* getddB() { return dBiasSimd.getData(); };
	tensor getOut() {		return outSpike;	};
	tensor getOutMem() { return outputMemTensor; };
	tensor getXinput() { return inputXTensor; };
	tensor getInput() { return inputSpike; };
	int getHiddenNumth() { return hiddenNumth; };
	float* getOutAverage(int b) { return outAverageT + b * AlignVec(outSpike.getDim().dim3, AlignBytes / sizeof(float)); };
	float* getOutAverageEXP(int b) { return outAverageTEXP + b * AlignVec(outSpike.getDim().dim3, AlignBytes / sizeof(float)); };
	void checkSingleNeuro(float* input,float* mem, float* spike);
	//---------------------------------------------------------------------------------q train
	void dLinearMatMultplySimdW(  int b);
	void dLinearMatMultplySimdS(tensor& lastdCI, int b);
	tensor& getDCI() { return dCI; };
	tensor getIdealOut() { return idealOutSpike; };
	void setDCI(tensor xxoo) { dCI = xxoo; };
	void accumulateW(float lr, int batchSize);
	void accumulateWTruncate(float lr, int batchSize);


	void softmaxOutV2(int b);
	void UpdateLayerWBADAMW(float learnrate, Optimizer mOptimizer, int batchSize, int t);
	void ADAMW(float learnrate, Optimizer mOptimizer, tensor w, tensor g, tensor m, tensor v, int t);

	void getG(int batchSize);
	//---------------------------------------------------------------------------------d train
private:
	tensor inputSpike;
	tensor inputXTensor;
	
	tensor W;
	tensor biasSimd;
	tensor outputMemTensor;//mem
	tensor outSpike;
	tensor idealOutSpike;
	spikeNeuro mySpikeNeuro;

	SNNLayer* postLayer;
	SNNLayer* preLayer;
	float* outAverageT;
	float* outAverageTEXP;
	int hiddenNumth;
	bool isSoftmaxOut;
	int actFun;
	int TIMESTEP;
	void linearMatMultplySimd(int t, int b);
	void activateOperateSimd(int t, int b);
	void spikeActivateSimd(int t, int b);
	void softmaxActivate(int t, int b);
	void softmaxOut(int t, int b);
	
	//_____________________q train
	
	void dXdW(tensor dCX, tensor& dCdW, int t, int b);
	void dUdUsub1(tensor& dCU, int t, int b);

	void dSoftmax( int t, int b);
	void dSSurrogate(tensor dCI, tensor& dCU, int t,int b);
	void dUdX(tensor dCU, tensor& dCX, int t, int b);
	void dXdI(tensor dCX, tensor& lastdCI, int t, int b);
	void TransMatrix();

	tensor dBiasSimd;
	tensor dCWTotal,WT;
	tensor dCU, dCX;
	tensor dCI;
	tensor shadowM;
	tensor shadowV;
	tensor shadowBM;
	tensor shadowBV;
	//tensor dydx, lastDydx;
	//_____________________d train
};