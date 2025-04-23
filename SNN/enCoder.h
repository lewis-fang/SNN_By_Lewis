#pragma once
#include"dataDefine.h"
class enCoder
{
public:
	enCoder();
	~enCoder() {};
	bool encodeInput(float* imageInput, size_t length, float* spikeInput);
	void setEncoer(int mth, int T) { encodeMethod = mth; TIMESTEP = T; }
private:
	int encodeMethod;
	int TIMESTEP;
	bool latency(float* imageInput, size_t length, float* spikeInput, float tao, float Vthr);
	bool binaryCode(char* imageInput, size_t length, float* spikeInput);
	bool aveRateCode(float* imageInput, size_t length, float* spikeInput, float Vthr);
};