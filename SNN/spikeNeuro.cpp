#include"spikeNeuro.h"

spikeNeuro::spikeNeuro()
{
	beta = 0.999;// quiesce membrain voltage
	Uthr = 1;
	lastMem = 0;


	//lastMemSimd = _mm256_setzero_ps();
	betaSimd = _mm256_set1_ps(beta);
	UthrSimd= _mm256_set1_ps(Uthr);
	resetMethod = 1;
	/*
	* resetMethod:
	* 0: set to 0
	* 1: -Uthr
	*/
}
spikeNeuro::~spikeNeuro()
{

}
void spikeNeuro::init(float bt, float Ut, int rM)
{
	beta = bt;// quiesce membrain voltage
	Uthr = Ut;
	resetMethod = rM;

	betaSimd = _mm256_set1_ps(beta);
	UthrSimd = _mm256_set1_ps(Uthr);
}

float spikeNeuro::reset(float mem)
{
	return mem > Uthr ? 1.0 : 0;
}
float spikeNeuro::activate(float xcurrent)
{
	if (resetMethod == 0)
	{
		lastMem = 0+xcurrent;
	}
	else
	{
		lastMem = lastMem * beta + xcurrent - Uthr* reset(lastMem);
	}
	
	return lastMem;
}
__m256 spikeNeuro::heaviside(__m256 memSimd)
{
	__m256 ones = _mm256_set1_ps(1.0);
	__m256 cmpRegister = _mm256_cmp_ps(UthrSimd, memSimd, 1);//less than: 0xfffff...
	return _mm256_and_ps(cmpRegister, ones);
}
float spikeNeuro::heavisideNoSimd(float mem)
{
	return mem > Uthr ? 1.0 : 0;
}
__m256 spikeNeuro::resetSimd(__m256 memSimd)
{
	__m256 SMultiUthr = _mm256_mul_ps(UthrSimd, heaviside(memSimd));
	return SMultiUthr;

}
__m256 spikeNeuro::activateSimd(__m256 xlastSimd,__m256 xcurrentSimd)
{
	__m256 lastMemSimd= xcurrentSimd;
	if (resetMethod == 0)
	{
		//lastMemSimd = xcurrentSimd;
	}
	else
	{//1
		__m256 tmpReg = _mm256_fmadd_ps(xlastSimd, betaSimd, xcurrentSimd);	
		lastMemSimd = _mm256_sub_ps(tmpReg, resetSimd(xlastSimd));
	}
	
	return lastMemSimd;
}
