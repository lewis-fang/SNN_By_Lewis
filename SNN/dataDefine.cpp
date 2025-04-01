#include"dataDefine.h"

tensor::tensor() 
{
	data = nullptr; 
	blocksize = 0;
}

tensor::~tensor()
{

}
float* tensor::getDim3Data(int d1, int d2)
{
	size_t offset = AlignBytes / sizeof(float);
	return data + d1 * tensorDim.dim2 * AlignVec(tensorDim.dim3, offset) + d2 * AlignVec(tensorDim.dim3, offset);
}
float* tensor::getDim23Data(int d1)
{
	size_t offset = AlignBytes / sizeof(float);
	return data + d1 * tensorDim.dim2 * AlignVec(tensorDim.dim3, offset);
}
void tensor::initData(dim tsd)
{
	blocksize = tsd.dim1 * tsd.dim2 *AlignVec( tsd.dim3, AlignBytes / sizeof(float));
	SingleBlocksizeDim3 = AlignVec(tsd.dim3, AlignBytes / sizeof(float));
	SingleBlocksizeDim23 = tsd.dim2 * AlignVec(tsd.dim3, AlignBytes / sizeof(float));
	data = (float*)_mm_malloc(blocksize * sizeof(float), AlignBytes);
	memset(data, 0, blocksize * sizeof(float));

	memcpy(&tensorDim, &tsd, sizeof(dim));
}
void tensor::freeData()
{
	if (data != nullptr)
	{
		_mm_free(data);
	}
}

void tensor::randInit(float sd, float mu)
{
	std::normal_distribution<float> nd(mu, sd);
	for (int ch = 0; ch < tensorDim.dim3; ch++)
	{
		for (int r = 0; r < tensorDim.dim1; r++)
		{
			for (int c = 0; c < tensorDim.dim2;c++)
			{
				float randfloat = nd(gen);
				data[r * tensorDim.dim2 * AlignVec(tensorDim.dim3, AlignBytes / sizeof(float)) + c * AlignVec(tensorDim.dim3, AlignBytes / sizeof(float)) + ch] = randfloat;

			}
		}
	}
}

void tensor::divedValue(float ra, int t , int b)
{
	size_t offset = AlignBytes / sizeof(float);
	__m256 raReg = _mm256_set1_ps(ra);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 dataReg = _mm256_load_ps(curdata + i);
			dataReg = _mm256_div_ps(raReg, dataReg);
			_mm256_stream_ps(curdata + i, dataReg);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 dataReg = _mm256_load_ps(curdata + i);
			dataReg = _mm256_div_ps(raReg, dataReg);
			_mm256_stream_ps(curdata + i, dataReg);
		}
	}
}
void tensor::addValue(float ra, int t , int b )
{
	size_t offset = AlignBytes / sizeof(float);
	__m256 raReg = _mm256_set1_ps(ra);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 dataReg = _mm256_load_ps(curdata + i);
			dataReg = _mm256_add_ps(raReg, dataReg);
			_mm256_stream_ps(curdata + i, dataReg);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 dataReg = _mm256_load_ps(curdata + i);
			dataReg = _mm256_add_ps(raReg, dataReg);
			_mm256_stream_ps(curdata + i, dataReg);
		}
	}
}
void tensor::applyRatio(float ra, int t, int b)
{
	size_t offset = AlignBytes / sizeof(float);
	__m256 raReg = _mm256_set1_ps(ra);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 dataReg = _mm256_load_ps(curdata + i);
			dataReg = _mm256_mul_ps(raReg, dataReg);
			_mm256_stream_ps(curdata + i, dataReg);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 dataReg = _mm256_load_ps(curdata + i);
			dataReg = _mm256_mul_ps(raReg, dataReg);
			_mm256_stream_ps(curdata + i, dataReg);
		}
	}
}
void tensor::truncateValue(float ra, int t, int b)
{
	size_t offset = AlignBytes / sizeof(float);
	__m256 raReg = _mm256_set1_ps(ra);
	__m256 invRaReg = _mm256_set1_ps(-ra);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 dataReg = _mm256_load_ps(curdata + i);
			__m256 cmp1 = _mm256_cmp_ps(dataReg, raReg, 2);
			__m256 cmpdataReg = _mm256_and_ps(cmp1,dataReg );
			__m256 cmpraReg = _mm256_andnot_ps(cmp1,raReg);
			__m256 sReg = _mm256_add_ps(cmpdataReg, cmpraReg);

			__m256 cmp3 = _mm256_cmp_ps(invRaReg, sReg, 2);
			__m256 cmpsReg = _mm256_and_ps(cmp3,sReg);
			__m256 cmpinvRaReg = _mm256_andnot_ps(cmp3,invRaReg);
			sReg = _mm256_add_ps(cmpsReg, cmpinvRaReg);
			_mm256_stream_ps(curdata + i, sReg);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 dataReg = _mm256_load_ps(curdata + i);
			__m256 cmp1 = _mm256_cmp_ps(dataReg,raReg,2);
			 cmp1 = _mm256_and_ps(dataReg, cmp1);
			__m256 cmp2 = _mm256_andnot_ps(raReg, cmp1);
			__m256 sReg = _mm256_add_ps(cmp1, cmp2);

			__m256 cmp3 = _mm256_cmp_ps(invRaReg,sReg, 2);
			cmp3 = _mm256_and_ps(sReg, cmp3);
			__m256 cmp4 = _mm256_andnot_ps(invRaReg, cmp3);
			sReg = _mm256_add_ps(cmp3, cmp4);
			_mm256_stream_ps(curdata + i, sReg);
		}
	}
}
void tensor::divTensor(tensor ts, int t , int b)
{
	size_t offset = AlignBytes / sizeof(float);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 dataReg1 = _mm256_load_ps(curdata + i);
			__m256 dataReg2 = _mm256_load_ps(ts.getDim23Data(b) + i);
			dataReg2 = _mm256_div_ps(dataReg1, dataReg2);
			_mm256_stream_ps(curdata + i, dataReg2);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 dataReg1 = _mm256_load_ps(curdata + i);
			__m256 dataReg2 = _mm256_load_ps(ts.getDim3Data(b, t) + i);
			dataReg2 = _mm256_div_ps(dataReg1, dataReg2);
			_mm256_stream_ps(curdata + i, dataReg2);
		}
	}
}
void tensor::mulTensor(tensor ts, int t, int b)
{
	size_t offset = AlignBytes / sizeof(float);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 dataReg1 = _mm256_load_ps(curdata + i);
			__m256 dataReg2 = _mm256_load_ps(ts.getDim23Data(b) + i);
			dataReg2 = _mm256_mul_ps(dataReg1, dataReg2);
			_mm256_stream_ps(curdata + i, dataReg2);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 dataReg1 = _mm256_load_ps(curdata + i);
			__m256 dataReg2 = _mm256_load_ps(ts.getDim3Data(b, t) + i);
			dataReg2 = _mm256_mul_ps(dataReg1, dataReg2);
			_mm256_stream_ps(curdata + i, dataReg2);
		}
	}
}

void tensor::addTensor(tensor ts, int t, int b)
{
	size_t offset = AlignBytes / sizeof(float);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 dataReg1 = _mm256_load_ps(curdata + i);
			__m256 dataReg2 = _mm256_load_ps(ts.getDim23Data(b) + i);
			dataReg2 = _mm256_add_ps(dataReg1, dataReg2);
			_mm256_stream_ps(curdata + i, dataReg2);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 dataReg1 = _mm256_load_ps(curdata + i);
			__m256 dataReg2 = _mm256_load_ps(ts.getDim3Data(b, t) + i);
			dataReg2 = _mm256_add_ps(dataReg1, dataReg2);
			_mm256_stream_ps(curdata + i, dataReg2);
		}
	}
}

void tensor::valueInit(float value, int t, int b)
{
	size_t offset = AlignBytes / sizeof(float);
	__m256 raReg = _mm256_set1_ps(value);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
				_mm256_stream_ps(curdata + i, raReg);
			}
	}
	else
	{
		float* curdata=getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			_mm256_stream_ps(curdata + i, raReg);
		}
	}
}
void tensor::sqrtTensor(int t, int b)
{
	size_t offset = AlignBytes / sizeof(float);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 tmpReg = _mm256_load_ps(curdata + i);
			tmpReg = _mm256_sqrt_ps(tmpReg);
			_mm256_stream_ps(curdata + i, tmpReg);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 tmpReg = _mm256_load_ps(curdata + i);
			tmpReg = _mm256_sqrt_ps(tmpReg);
			_mm256_stream_ps(curdata + i, tmpReg);
		}
	}
}
void tensor::copyTensor(tensor ts, int t, int b)
{
	size_t offset = AlignBytes / sizeof(float);
	if (t == -1)
	{
		float* curdata = getDim23Data(b);
		for (int i = 0;i < SingleBlocksizeDim23;i += offset)
		{
			__m256 tsReg = _mm256_load_ps(ts.getDim23Data(b) + i);
			_mm256_stream_ps(curdata + i, tsReg);
		}
	}
	else
	{
		float* curdata = getDim3Data(b, t);
		for (int i = 0;i < tensorDim.dim3;i += offset)
		{
			__m256 tsReg = _mm256_load_ps(ts.getDim3Data(b, t) + i);
			_mm256_stream_ps(curdata + i, tsReg);
		}
	}
}
void tensor::recordW(int b)
{
	std::ofstream of("./2.csv", std::ios::app);
	float* curData =data+ b * SingleBlocksizeDim23;
	for (int i = 0;i < tensorDim.dim2;i++)
	{
		for (int j = 0;j < tensorDim.dim3;j++)
		{
			of << curData[i * AlignVec(tensorDim.dim3, AlignBytes/sizeof(float)) + j]<<",";
		}
		of << std::endl;
	}
	of.close();
}
void tensor::recordT(int b)
{
	std::ofstream of("./3.csv", std::ios::app);
	float* curData = data + b * SingleBlocksizeDim23;
	for (int i = 0;i < tensorDim.dim2;i++)
	{
		for (int j = 0;j < tensorDim.dim3;j++)
		{
			of << curData[i * AlignVec(tensorDim.dim3, AlignBytes / sizeof(float)) + j];
			of << ",";
		}
		of << std::endl;
	}
	of.close();
}
