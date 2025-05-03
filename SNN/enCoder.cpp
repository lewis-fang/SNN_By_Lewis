#include"enCoder.h"

enCoder::enCoder() 
{
	encodeMethod=0;
	TIMESTEP=25;
	needNormalize = true;
}
bool enCoder::latency(float* imageInput, size_t length, float* spikeInput, float tao, float Vthr,float maxValue)
{//imageInput 28x28
	//spikeInput 28*28*T
	int offset = AlignBytes / sizeof(float);
	__m256 VthrReg = _mm256_set1_ps(Vthr);
	__m256 taoReg = _mm256_set1_ps(tao);
	int a = 0x7fffffff;
	__m256 tregf2 = _mm256_set1_ps(*(float*)&a);
	__m256 fequal = _mm256_set1_ps(0.5);

	//std::ofstream of("./1.csv", std::ios::app);
	float* spikeTime = (float*)_mm_malloc(length * sizeof(float), AlignBytes);
	for (int i = 0;i < length;i += offset)
	{
		__m256 offsetReg = _mm256_load_ps(imageInput + i);
		//__m256 cmpreg = _mm256_cmp_ps(offsetReg, VthrReg, 1);
		__m256 tmpReg = _mm256_sub_ps(offsetReg, VthrReg);
		tmpReg = _mm256_div_ps(offsetReg, tmpReg);
		tmpReg = _mm256_log_ps(tmpReg);
		tmpReg = _mm256_mul_ps(tmpReg, taoReg);
		//tmpReg= _mm256_or_ps(tmpReg, cmpreg);
		tmpReg = _mm256_and_ps(tmpReg, tregf2);
		_mm256_stream_ps(spikeTime + i, tmpReg);
	}
	memset(spikeInput, 0, TIMESTEP * length * sizeof(float));
	__m256 ones = _mm256_set1_ps(1);

	for (int t = 0;t < TIMESTEP;t++)
	{
		float* currentStrd = spikeInput + t * length;
		__m256 tregf = _mm256_set1_ps(t);

		for (int i = 0;i < length;i += offset)
		{
			__m256 offsetReg = _mm256_load_ps(spikeTime + i);
			offsetReg = _mm256_sub_ps(offsetReg, tregf);
			offsetReg = _mm256_and_ps(offsetReg, tregf2);

			offsetReg = _mm256_cmp_ps(offsetReg, fequal, 1);
			offsetReg = _mm256_and_ps(offsetReg, ones);
			_mm256_stream_ps(currentStrd + i, offsetReg);
		}
	}
	_mm_free(spikeTime);
	return true;
}
bool enCoder::binaryCode(uint32_t* imageInput, size_t length, float* spikeInput, float maxValue)
{
	//spikeInput 28*28*T
	for (int t = 0;t < TIMESTEP;t++)
	{
		float* currentStrd = spikeInput + t * length;

		for (int i = 0;i < length;i += 1)
		{
			uint32_t x = imageInput[i] ;
			currentStrd[i] = float((x >> (TIMESTEP - 1 - t)) & 0x1);
		}
	}
	return true;
}

bool enCoder::aveRateCode(float* imageInput, size_t length, float* spikeInput, float Vthr, float maxValue)
{
	int revolution = 64;
	for (int i = 0;i < length;i += 1)
	{
		int inverval = revolution *(1 - *(imageInput + i))+ 32;
		if (inverval > TIMESTEP / 4) inverval = TIMESTEP + 1;
		if (inverval < TIMESTEP)
		{
			for (int t = inverval;t < TIMESTEP;t += inverval)
			{
				float* currentStrd = spikeInput + t * length;
				currentStrd[i] = 1;
			}
		}
	
	}
	return true;
}

bool enCoder::encodeInput(float* imageInput, size_t length, float* spikeInput, float maxValue)
{
	float* normalizeImageInput = (float*)_mm_malloc(length * sizeof(float), AlignBytes);
	normalize(imageInput, length, normalizeImageInput);
	//memcpy(imageInput, normalizeImageInput, length * sizeof(float));

	switch (encodeMethod)
	{
	case 0:
		latency(normalizeImageInput, length, spikeInput, TAO, VTHR, maxValue);
		break;
	case 1:
	{
		uint32_t* charMnistTEST = (uint32_t*)_mm_malloc(length * sizeof(uint32_t), AlignBytes);
		memset(charMnistTEST, 0, length * sizeof(uint32_t));
		for (int i = 0;i < length;i++)
		{
			charMnistTEST[i] = uint32_t(normalizeImageInput[i]* maxValue);
		}
		binaryCode(charMnistTEST, length, spikeInput,  maxValue);
		_mm_free(charMnistTEST);
		break;
	}
	case 2:
		aveRateCode(normalizeImageInput, length, spikeInput, VTHR , maxValue);
		break;
	default:
		break;
	}
	return true;
}

bool enCoder::normalize(float* imageInput, size_t length, float* normalizeImageInput)
{
	
	switch (needNormalize)
	{
	case false:
		memcpy(normalizeImageInput, imageInput, length * sizeof(float));
		break;
	case 1:
	{
		float mmaxv = imageInput[0];
		float mminv = imageInput[0];
		for (int i = 0;i < length;i++)
		{
			if(mmaxv < imageInput[i]) mmaxv = imageInput[i];
			if (mminv > imageInput[i]) mminv = imageInput[i];
		}
		for (int i = 0;i < length;i++)
		{
			normalizeImageInput[i] = (imageInput[i] - mminv) / (mmaxv - mminv);
		}
		break;
	}
	default:
		break;
	}
	return true;
}