#include"SNNLayer.h"

SNNLayer::SNNLayer()
{
	postLayer = nullptr;
	preLayer = postLayer;

	actFun = 1;
	hiddenNumth = -1;
	isSoftmaxOut = false;
	TIMESTEP = 25;
}
SNNLayer::~SNNLayer()
{

}
void SNNLayer::setIdealOut(float* tsData, int b)
{
	for (int t = 0;t < TIMESTEP;t++)
	{
		float* data = idealOutSpike.getDim3Data(b, t);
		memcpy(data, tsData, sizeof(float) * OUTCLASS);
	}

}

bool SNNLayer::initSnnLayer(dim inputDim,dim outDim, float beta, float Uthr, int resetMethod,float sd,float mu, int batchSize)
{
		bool ret = true;
		int offset = AlignBytes / sizeof(float);
		std::cout << "	start initiated" << std::endl;
		inputSpike.initData(inputDim);
		//inputSpike.randInitSpike(1, 0);
		std::cout << "	inputSpike initiated" << std::endl;
		inputXTensor.initData(outDim);
		std::cout << "	inputXTensor initiated" << std::endl;
		outputMemTensor.initData(outDim);
		std::cout << "	outputMemTensor initiated" << std::endl;
		outSpike.initData(outDim);
		std::cout << "	outSpike initiated" << std::endl;
		idealOutSpike.initData(outDim);
		dim dimW(1,outDim.dim3, inputDim.dim3);
		dim dimB(1,1, outDim.dim3);
		W.initData(dimW);
		W.randInit(sd, mu);
		dim dimWT(1, inputDim.dim3, outDim.dim3);
		WT.initData(dimWT);
		biasSimd.initData(dimB);

		std::cout << "	W initiated" << std::endl;
		TransMatrix();
		//--------------------------------------------------------------------------------for traing collection
		int dWDim1 = std::max(batchSize, 3);
		dim ddimW(dWDim1, outDim.dim3, inputDim.dim3);
		dim ddimB(dWDim1, 1,outDim.dim3);
		dCWTotal.initData(ddimW);
		dBiasSimd.initData(ddimB);

		outAverageT= (float*)_mm_malloc(batchSize * AlignVec(outDim.dim3, offset) * sizeof(float), AlignBytes);
		outAverageTEXP = (float*)_mm_malloc(batchSize * AlignVec(outDim.dim3, offset) * sizeof(float), AlignBytes);

		memset(outAverageT, 0, batchSize * sizeof(float) * AlignVec(outDim.dim3, offset));
		memset(outAverageTEXP, 0, batchSize * sizeof(float) * AlignVec(outDim.dim3, offset));

		mySpikeNeuro.init(beta, Uthr, resetMethod);
		std::cout << "	mySpikeNeuro initiated" << std::endl;
		
		shadowM.initData(dimW);
		shadowV.initData(dimW);
		shadowBM.initData(dimB);
		shadowBV.initData(dimB);


		dCU.initData(outDim);
		std::cout << "	dCU initiated" << std::endl;
		dCX.initData(outDim);
		std::cout << "	dCX initiated" << std::endl;
		dCI.initData(outDim);
	return ret;
}

void SNNLayer::linearMatMultplySimd(int t, int b)
{
	float* singleinputSpike = inputSpike.getDim3Data(b, t);
	float* singleXTensor = inputXTensor.getDim3Data(b, t);
	int offset = AlignBytes / sizeof(float);
	for (int j = 0;j < inputXTensor.getDim().dim3;j++)
	{ 
		float* currentWTensor = W.getDim3Data(0,j);
		__m256 sumReg = _mm256_set1_ps(biasSimd.getData()[j]);
		for (int i = 0;i < inputSpike.getDim().dim3;i+=offset)
		{		
			__m256 inputReg = _mm256_load_ps(singleinputSpike+i);
			__m256 wReg = _mm256_load_ps(currentWTensor + i);
			sumReg = _mm256_fmadd_ps(inputReg, wReg, sumReg);
		}
		sumReg = _mm256_hadd_ps(sumReg, sumReg);
		sumReg = _mm256_hadd_ps(sumReg, sumReg);
		singleXTensor[j] = sumReg.m256_f32[0] + sumReg.m256_f32[4];
	}
}
void SNNLayer::spikeActivateSimd(int t, int b)
{
	float* singleOutTensor = outputMemTensor.getDim3Data(b, t);
	float* singleXTensor = inputXTensor.getDim3Data(b, t);
	float* LastSingleoutSpike;
	if (t > 0)
	{
		LastSingleoutSpike = outputMemTensor.getDim3Data( b,t - 1);
	}
	else
	{
		LastSingleoutSpike = outputMemTensor.getDim3Data(b, t);
	}
	
	float* singleoutSpike = outSpike.getDim3Data(b, t);
	int offset = AlignBytes / sizeof(float);
	for (int i = 0;i < inputXTensor.getDim().dim3;i += offset)
	{
		__m256 bzActivateBaseSimdReg = _mm256_load_ps(singleXTensor + i);
		__m256 xLastMemReg = _mm256_setzero_ps();
		if (t > 0)
		{
			xLastMemReg = _mm256_load_ps(LastSingleoutSpike + i);
		}
		__m256 afterActivateSimdReg = mySpikeNeuro.activateSimd(xLastMemReg,bzActivateBaseSimdReg);
		__m256 regSpikeOut = mySpikeNeuro.heaviside(afterActivateSimdReg);
		_mm256_stream_ps(singleOutTensor + i, afterActivateSimdReg);
		_mm256_stream_ps(singleoutSpike + i, regSpikeOut);
	}
}

void SNNLayer::softmaxActivate(int t, int b)
{
	int offset = AlignBytes / sizeof(float);
	float* singleOutTensor = outputMemTensor.getDim3Data(b, t);
	float* singleoutSpike = outSpike.getDim3Data(b, t);
	float* singleXTensor = inputXTensor.getDim3Data(b, t);
	float maxValue = -9999999999.0f;
	for (int i = 0;i < inputXTensor.getDim().dim3;i += 1)
	{//get max value x
		if (maxValue < *(singleXTensor + i))
		{
			maxValue = *(singleXTensor + i);
		}
	}
	float sum = 0.0;
	for (int i = 0;i < inputXTensor.getDim().dim3;i += 1)
	{//get max value x		
		sum += exp(*(singleXTensor + i) - maxValue);
	}
	for (int i = 0;i < inputXTensor.getDim().dim3;i += 1)
	{//get max value x		
		*(singleOutTensor + i) = exp(*(singleXTensor + i) - maxValue) / sum;
		*(singleoutSpike + i) = *(singleOutTensor + i);
	}
}
void SNNLayer::softmaxOut(int t, int b)
{
	int offset = AlignBytes / sizeof(float);
	float* singleOutTensor = outputMemTensor.getDim3Data(b, t);
	float* singleoutSpike = outSpike.getDim3Data(b, t);
	//float* singleXTensor = inputXTensor.getDim3Data(b, t);
	float maxValue = -9999999999.0f;
	for (int i = 0;i < outputMemTensor.getDim().dim3;i += 1)
	{//get max value x
		if (maxValue < *(singleOutTensor + i))
		{
			maxValue = *(singleOutTensor + i);
		}
	}
	float sum = 0.0;
	for (int i = 0;i < outputMemTensor.getDim().dim3;i += 1)
	{//get max value x		
		sum += exp(*(singleOutTensor + i) - maxValue);
	}
	for (int i = 0;i < outputMemTensor.getDim().dim3;i += 1)
	{//get max value x		
		*(singleoutSpike + i) = exp(*(singleOutTensor + i) - maxValue) / sum;
	}
}
void SNNLayer::softmaxOutV2(int b)
{
	//float* singleXTensor = inputXTensor.getDim3Data(b, t);
	float maxValue = -9999999999.0f;
	int offset = AlignBytes / sizeof(float);
	int singleBatichSize = AlignVec(outSpike.getDim().dim3, offset);
	for (int i = 0;i < outSpike.getDim().dim3;i += 1)
	{//get max value x
		float soi = 0;
		for (int t = 0;t < TIMESTEP;t++)
		{
			float* singleoutSpike = outSpike.getDim3Data(b, t);
			soi += singleoutSpike[i];
		}
		//soi /= TIMESTEP;
		outAverageT[singleBatichSize * b + i] = soi;
		if (maxValue < soi)
		{
			maxValue = soi;
		}
	}
	float sum = 0.0;
	for (int i = 0;i < outSpike.getDim().dim3;i += 1)
	{//get max value x		
		sum += exp(outAverageT[singleBatichSize * b + i] - maxValue);
	}
	for (int i = 0;i < outSpike.getDim().dim3;i += 1)
	{//get max value x		
		outAverageTEXP[singleBatichSize * b + i] = exp(outAverageT[singleBatichSize*b+i] - maxValue) / sum;
	}
}
void SNNLayer::activateOperateSimd(int t, int b)
{
	if (actFun == 1)
	{//spike activate
		spikeActivateSimd(t, b);
	}
	else if (actFun == 2)
	{//softmax
		softmaxActivate(t, b);
	}
	else
	{
		std::cout << "what activate function ~" << std::endl;
	}
}
void SNNLayer::layerCalcSimd(int t, int b)
{
	linearMatMultplySimd(t, b);
	activateOperateSimd(t, b);
}

bool SNNLayer::setIn(tensor ts,int t, int b1,int b2)
{
	bool ret = true;
	int offset = AlignBytes / sizeof(float);
	float* lstTensor = ts.getDim3Data(b1,t);
	float* thsTensor = inputSpike.getDim3Data(b2,t);
	if (ts.getDim().dim3 == inputSpike.getDim().dim3)
	{
		memcpy(thsTensor, lstTensor, ts.getDim().dim3 * sizeof(float));
	}
	else
	{
		ret = false;
		std::cout << "snn structure wrong, size not equal to the next layer~" << std::endl;
		std::cout << ts.getDim().dim3<<","<< inputSpike.getDim().dim3 << std::endl;
	}
	return ret;
}

void SNNLayer::checkSingleNeuro(float* input,float* mem, float*spike)
{
	for (int i = 0;i < TIMESTEP;i++)
	{
		 mem[i] = mySpikeNeuro.activate(input[i]);
		 spike[i] = mySpikeNeuro.heavisideNoSimd(mem[i]);
	}
}
//_____________________q
void SNNLayer::dLinearMatMultplySimdW( int b)
{

	dCWTotal.valueInit(0.0,-1,b);
	//std::cout << "valueInit ok" << std::endl;
	if (hiddenNumth == 1)
	{		
		dCU.valueInit(0.0, -1, b);
		dCX.valueInit(0.0, -1, b);
		//
		for (int t = TIMESTEP - 1;t >= 0;t--)
		{
			if (isSoftmaxOut == true)
			{
				dSoftmax(t, b);			
			}
			dSSurrogate(dCI, dCU, t, b);
			if (t > 0)
			{
				dUdUsub1(dCU, t, b);
			}
		}
		for (int t = TIMESTEP - 1;t >= 0;t--)
		{
			dUdX(dCU, dCX, t, b);
		}
	}
	for (int t = TIMESTEP - 1;t >= 0;t--)
	{
		dXdW(dCX, dCWTotal, t, b);
	}
}
void SNNLayer::dUdUsub1(tensor& dCU, int t, int b)
{
	//dCU.copyTensor(dCU, t, b);
	float* dataLast = dCU.getDim3Data(b, t - 1);
	float* dataCur= dCU.getDim3Data(b, t);
	__m256 betaReg = _mm256_set1_ps(mySpikeNeuro.getBeta());
	for (int i = 0;i < dCU.getDim().dim3;i += AlignBytes / sizeof(float))
	{
		__m256 lastReg=_mm256_load_ps(dataLast + i);
		__m256 curReg = _mm256_load_ps(dataCur + i);
		curReg =_mm256_mul_ps(curReg, betaReg);
		//curReg = _mm256_add_ps(curReg, lastReg);
		_mm256_stream_ps(dataLast + i, curReg);
	}
}
void SNNLayer::dXdW(tensor dCX, tensor& dCW, int t, int b)
{
	float* curDydx = dCX.getDim3Data(b, t);
	int offset = AlignBytes / sizeof(float);
	float* curdBias = dBiasSimd.getDim23Data(b);
	for (int i = 0;i < outSpike.getDim().dim3;i += offset)
	{
		__m256 dbReg = _mm256_load_ps(curdBias + i);
		__m256 dCXReg = _mm256_load_ps(curDydx + i);
		dbReg = _mm256_add_ps(dbReg, dCXReg);
		_mm256_stream_ps(curdBias + i, dbReg);
	}
	float* singleinputSpike = inputSpike.getDim3Data(b, t);
	for (int j = 0;j < dCX.getDim().dim3;j++)
	{
		float* currentWTensor = dCW.getDim3Data(b,j);
		__m256 dydxReg = _mm256_set1_ps(curDydx[j]);
		for (int i = 0;i < inputSpike.getDim().dim3;i += offset)
		{
			__m256 inputReg = _mm256_load_ps(singleinputSpike + i);
			__m256 dWReg = _mm256_load_ps(currentWTensor + i);
			dWReg =_mm256_fmadd_ps(inputReg, dydxReg, dWReg);
			_mm256_stream_ps(currentWTensor + i, dWReg);
		}
	}
}
void SNNLayer::dLinearMatMultplySimdS(tensor& lastdCI, int b)
{
	//dSpike/dSpike=dSpike/dU*dU/dX*dX/dSpike
	dCU.valueInit(0.0,-1,b);
	dCX.valueInit(0.0, -1, b);
	lastdCI.valueInit(0.0, -1, b);
	
	for (int t = TIMESTEP - 1;t >= 0;t--)
	{
		if (isSoftmaxOut == true)
		{
			dSoftmax(t, b);
		}
		dSSurrogate(dCI, dCU, t, b);
		if (t > 0)
		{
			dUdUsub1(dCU, t, b);
		}
	}
	for (int t = TIMESTEP - 1;t >= 0;t--)
	{
		dUdX(dCU, dCX, t, b);
		dXdI(dCX, lastdCI, t, b);
	}
//	std::cout << "dLinearMatMultplySimdS: dUdUsub1 ok" << std::endl;
}
void SNNLayer::dSSurrogate(tensor dCI, tensor& dCU, int t, int b)
{//1/pi*(1/(1+pow(U*pi,2)))
	float* dCIPointer = dCI.getDim3Data(b, t);
	float* dCUPointer = dCU.getDim3Data(b, t);
	float* memPointer = outputMemTensor.getDim3Data(b, t);
	for (int i = 0;i < dCI.getDim().dim3;i += AlignBytes / sizeof(float))
	{
		__m256 memReg = _mm256_load_ps(memPointer + i);
		__m256 dCIReg = _mm256_load_ps(dCIPointer + i);
		__m256 dCUReg = _mm256_load_ps(dCUPointer + i);
		memReg = _mm256_mul_ps(memReg, _mm256_set1_ps(PI));
		memReg = _mm256_mul_ps(memReg, memReg);
		memReg = _mm256_add_ps(memReg, _mm256_set1_ps(1.0f));
		memReg = _mm256_div_ps(_mm256_set1_ps(1.0/PI), memReg);
		memReg = _mm256_mul_ps(memReg, dCIReg);
		dCUReg =_mm256_add_ps(dCUReg, memReg);
		_mm256_stream_ps(dCUPointer + i, dCUReg);
	}
}
void SNNLayer::dSoftmax( int t, int b)
{
	size_t offset = AlignBytes / sizeof(float);
	float* curIdealData = idealOutSpike.getDim3Data(b, t);
	float* outATEXP = outAverageTEXP+b*AlignVec(outSpike.getDim().dim3, offset);
	float* dCIData = dCI.getDim3Data(b, t);
	for (int i = 0;i < dCI.getDim().dim3;i += offset)
	{
		__m256 dataReg1 = _mm256_load_ps(curIdealData + i);
		__m256 dataReg2 = _mm256_load_ps(outATEXP + i);
		dataReg2 = _mm256_sub_ps(dataReg2, dataReg1);
		_mm256_stream_ps(dCIData + i, dataReg2);
	}
}
void SNNLayer::dUdX(tensor dCU, tensor& dCX, int t, int b)
{
	dCX.copyTensor(dCU, t, b);
}
void SNNLayer::dXdI(tensor dCX, tensor& lastdCI, int t, int b)
{
	float* lastDydx = lastdCI.getDim3Data(b, t);
	float* dXPointer = dCX.getDim3Data(b, t);
	int offset = AlignBytes / sizeof(float);
	for (int j = 0;j < lastdCI.getDim().dim3;j++)
	{
		float* currentWTensor = WT.getDim3Data(0, j);
		__m256 sumReg = _mm256_setzero_ps();
		for (int i = 0;i < dCX.getDim().dim3;i += offset)//out length
		{
			__m256 dxReg = _mm256_load_ps(dXPointer + i);
			__m256 WTReg = _mm256_load_ps(currentWTensor + i);
			sumReg = _mm256_fmadd_ps(dxReg, WTReg, sumReg);
		}
		sumReg = _mm256_hadd_ps(sumReg, sumReg);
		sumReg = _mm256_hadd_ps(sumReg, sumReg);
		lastDydx[j] = sumReg.m256_f32[0] + sumReg.m256_f32[4];
	}
}
void SNNLayer::TransMatrix()
{
	float* data = W.getData();
	float* dataT = WT.getData();
	int WiLineLength = AlignVec(W.getDim().dim3, AlignBytes / sizeof(float));
	int WoLineLength=AlignVec(WT.getDim().dim3, AlignBytes / sizeof(float));
	for (int d2 = 0;d2 < W.getDim().dim2;d2++)
	{//output not aligned to 8 floats		
		for (int d3 = 0;d3 < W.getDim().dim3;d3++)
		{//input aligned to 8 floats		
			dataT[d3 * WoLineLength + d2] = data[d2 * WiLineLength + d3];
		}
	}
}

void SNNLayer::accumulateWTruncate(float lr,int batchSize)
{
	__m256 raReg = _mm256_set1_ps(TRUCATE);
	__m256 invRaReg = _mm256_set1_ps(-TRUCATE);
	dCWTotal.truncateValue(TRUCATE,-1);
	int offset = AlignBytes / sizeof(float);
	int singleBlockSizeBias = AlignVec(outSpike.getDim().dim3, offset);
	int singleBlockSizeW = AlignVec(dCWTotal.getDim().dim3, offset) * dCWTotal.getDim().dim2;
	dCWTotal.applyRatio(-lr,-1);
	W.addTensor(dCWTotal,-1);
	for (int b = 0;b < batchSize;b++)
	{
		float* curdBias = dBiasSimd.getDim23Data(b);
		for (int i = 0;i < outSpike.getDim().dim3;i += offset)
		{
			__m256 dbReg = _mm256_load_ps(curdBias + i);
			__m256 bReg = _mm256_load_ps(biasSimd.getData() + i);

			__m256 cmp1 = _mm256_cmp_ps(dbReg, raReg, 2);
			__m256 cmpdataReg = _mm256_and_ps(cmp1, dbReg);
			__m256 cmpraReg = _mm256_andnot_ps(cmp1, raReg);
			__m256 sReg = _mm256_add_ps(cmpdataReg, cmpraReg);

			__m256 cmp3 = _mm256_cmp_ps(invRaReg, sReg, 2);
			__m256 cmpsReg = _mm256_and_ps(cmp3, sReg);
			__m256 cmpinvRaReg = _mm256_andnot_ps(cmp3, invRaReg);
			sReg = _mm256_add_ps(cmpsReg, cmpinvRaReg);

			sReg = _mm256_mul_ps(sReg, _mm256_set1_ps(-lr));
			bReg = _mm256_add_ps(sReg, bReg);
			_mm256_stream_ps(biasSimd.getData() + i, bReg);
		}
	}

	memset(dBiasSimd.getData(), 0, sizeof(float) * singleBlockSizeBias * batchSize);

	TransMatrix();
}

void SNNLayer::accumulateW(float lr, int batchSize)
{
	__m256 lrReg = _mm256_set1_ps(-lr);
	int offset = AlignBytes / sizeof(float);
	int singleBlockSizeBias = AlignVec(outSpike.getDim().dim3, offset);
	int singleBlockSizeW= AlignVec(dCWTotal.getDim().dim3, offset)* dCWTotal.getDim().dim2;
	for (int b = 0;b < batchSize;b++)
	{
		float* curdBias = dBiasSimd.getDim23Data(b);
		float* curdW = dCWTotal.getDim23Data(b);
		for (int i = 0;i < singleBlockSizeBias;i += offset)
		{
			__m256 dbReg = _mm256_load_ps(curdBias + i);
			__m256 bReg = _mm256_load_ps(biasSimd.getData() + i);
			bReg = _mm256_fmadd_ps(lrReg,dbReg, bReg);
			_mm256_stream_ps(biasSimd.getData() + i, bReg);
		}
		for (int i = 0;i < singleBlockSizeW;i += offset)
		{
			__m256 dwReg = _mm256_load_ps(curdW + i);
			__m256 wReg = _mm256_load_ps(W.getData() + i);
			wReg = _mm256_fmadd_ps(lrReg, dwReg, wReg);
			_mm256_stream_ps(W.getData() + i, wReg);
		}
	}

	memset(dBiasSimd.getData(), 0, sizeof(float) * singleBlockSizeBias * batchSize);

	TransMatrix();
}
void SNNLayer::UpdateLayerWBADAMW(float learnrate, Optimizer mOpti,int batchSize, int t)
{
	getG(batchSize);
	ADAMW(learnrate, mOpti, W, dCWTotal, shadowM, shadowV, t);
	ADAMW(learnrate, mOpti, biasSimd, dBiasSimd, shadowBM, shadowBV, t);
	memset(dBiasSimd.getData(), 0, sizeof(float) * dBiasSimd.getBlockSize());
	TransMatrix();
}
void SNNLayer::ADAMW(float learnrate, Optimizer mOpti, tensor w,tensor g, tensor moment, tensor velocity, int t)
{

	moment.applyRatio(mOpti.beta1);
	memcpy(g.getDim23Data(1), g.getDim23Data(0), sizeof(float) * g.getDim23BlockSize());
	g.applyRatio((1 - mOpti.beta1) / (1 - std::powf(mOpti.beta1, t)), -1, 0);
	moment.addTensor(g);

	velocity.applyRatio(mOpti.beta2);
	g.mulTensor(g, -1, 1);
	g.applyRatio((1 - mOpti.beta2) / (1 - std::powf(mOpti.beta2, t)), -1, 1);
	memcpy(g.getDim23Data(0), g.getDim23Data(1), sizeof(float) * g.getDim23BlockSize());
	velocity.addTensor(g);

	g.copyTensor(velocity, -1, 0);
	g.sqrtTensor(-1, 0);
	g.addValue(mOpti.sigma, -1, 0);
	memcpy(g.getDim23Data(1), moment.getDim23Data(0), sizeof(float) * g.getDim23BlockSize());
	moment.divTensor(g);
	moment.applyRatio(-learnrate);
	w.addTensor(moment);
	memcpy(moment.getDim23Data(0), g.getDim23Data(1), sizeof(float) * g.getDim23BlockSize());
}
void SNNLayer::getG( int batchSize)
{
	int offset = AlignBytes / sizeof(float);
	int singleBlockSizeBias = AlignVec(outSpike.getDim().dim3, offset);
	int singleBlockSizeW = AlignVec(dCWTotal.getDim().dim3, offset) * dCWTotal.getDim().dim2;
	float* curdBias0 = dBiasSimd.getData();
	float* curdW0= dCWTotal.getData();
	for (int b = 1;b < batchSize;b++)
	{
		float* curdBias = dBiasSimd.getDim23Data(b);
		float* curdW = dCWTotal.getDim23Data(b);

		for (int i = 0;i < singleBlockSizeBias;i += offset)
		{
			__m256 dbReg = _mm256_load_ps(curdBias + i);
			__m256 bReg0 = _mm256_load_ps(curdBias0 + i);
			bReg0 = _mm256_add_ps( dbReg, bReg0);
			_mm256_stream_ps(curdBias0 + i, bReg0);
		}
		for (int i = 0;i < singleBlockSizeW;i += offset)
		{
			__m256 dwReg = _mm256_load_ps(curdW + i);
			__m256 wReg0 = _mm256_load_ps(curdW0 + i);
			wReg0= _mm256_add_ps( dwReg, wReg0);
			_mm256_stream_ps(curdW0 + i, wReg0);
		}
	}
}
//_____________________d