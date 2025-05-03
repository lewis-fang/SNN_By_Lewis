#pragma once
#include"dataDefine.h"
#define TAO 100
#define NOBACKGGROUNDB
#define VTHR 0.02
class enCoder
{
public:
	enCoder();
	~enCoder() {};
	bool encodeInput(float* imageInput, size_t length, float* spikeInput,float maxValue);
	void setEncoer(int mth, int T) { encodeMethod = mth; TIMESTEP = T; }
private:
	int encodeMethod;
	int TIMESTEP;
	bool latency(float* imageInput, size_t length, float* spikeInput, float tao, float Vthr, float maxValue);
	bool binaryCode(uint32_t* imageInput, size_t length, float* spikeInput, float maxValue);
	bool aveRateCode(float* imageInput, size_t length, float* spikeInput, float Vthr, float maxValue);
	bool normalize(float* imageInput, size_t length, float* normalizeImageInput);

	bool needNormalize;
};