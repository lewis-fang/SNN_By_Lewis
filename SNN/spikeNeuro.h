#pragma once
#include<immintrin.h>
#include"dataDefine.h"
class spikeNeuro
{
public:
	spikeNeuro();
	~spikeNeuro();
	void init(float bt, float Ut, int rM);
	float activate(float xcurrent);
	__m256 activateSimd(__m256 xlastSimd, __m256 xcurrentSimd);
	__m256 heaviside(__m256 memSimd);
	float heavisideNoSimd(float mem);

	float getBeta() { return beta; };
private:
	
	float beta;
	
	int resetMethod;
	float Uthr;
	float lastMem;

	__m256 betaSimd;
	__m256 UthrSimd;
	//__m256 lastMemSimd;
	float reset(float mem);
	__m256 resetSimd(__m256 memSimd);
};
