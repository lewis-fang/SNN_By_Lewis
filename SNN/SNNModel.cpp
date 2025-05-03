#include"SNNModel.h"


snnModel::snnModel()
{
	lossType = 1;
	learnRate = 0.00025f;
	minLoss =0.001f;
	diffLoss =1e-6f;
	maxEpoch = 500;
	batchSize = 1;
	weitInitialsd = 0.1f;
	beta = 0.9f;
	Uthr = 1.0f;
	reset = 1;
	TIMESTEP = 25;
	encodeMethod = 0;
	modelBuilt = false;
	/*
	* 0 first fired
	* 1 binary rete encode
	* 2 uniform rate encode
	* 3 population rate encode
	* 4 diff fired
	*/
	isTrning = false;
}
void snnModel::createTrainThread()
{
	std::thread thr(&snnModel::BPTT, this);
	thr.detach();
}

void snnModel::buildMyDefaultSNNModel(int LAYER1, int MNISTDIM1, int MNISTDIM2,int OUTCLASS)
{

	float sd = weitInitialsd;
	float mu = 0.0;
	int act = 1;
	mySNNStructure.clear();
	dim iDim1(batchSize, TIMESTEP,MNISTDIM1 * MNISTDIM2);
	dim oDim1(batchSize,TIMESTEP, LAYER1);

	SNNLayer aSNNLayer1;
	aSNNLayer1.initSnnLayer(iDim1, oDim1, beta, Uthr, reset, sd, mu, batchSize);
	aSNNLayer1.setAct(act);
	aSNNLayer1.setHiddenNumth(1);
	aSNNLayer1.setT(TIMESTEP);
	mySNNStructure.push_back( aSNNLayer1);
	std::cout << "layer 1 added~" << std::endl;
	dim iDim2(oDim1);
	dim oDim2(batchSize,TIMESTEP, OUTCLASS);

	SNNLayer aSNNLayer2;
	aSNNLayer2.initSnnLayer(iDim2, oDim2, beta, Uthr, reset, sd, mu, batchSize);
	aSNNLayer2.setAct(act);//set softmax
	aSNNLayer2.setSoftmaxOut(true);
	aSNNLayer2.setHiddenNumth(2);
	aSNNLayer2.setT(TIMESTEP);
	mySNNStructure.push_back(aSNNLayer2);
	std::cout << "layer 2 added~" << std::endl;
	//bool SNNLayer::initSnnLayer(dim inputDim,dim outDim, float beta, float Uthr, int resetMethod,float sd,float mu)
	lossType = 2;

	modelBuilt = true;
}
void snnModel::fowardRecurrentSpikingSimd(tensor ts, int b)
{
	size_t depth = mySNNStructure.size();
	
	for (int t = 0;t < TIMESTEP;t++)
	{
		mySNNStructure.at(0).setIn(ts, t, 0,b);
		for (int ly=0;ly<depth;ly++)
		{
			SNNLayer slayer = mySNNStructure.at(ly);
			slayer.layerCalcSimd(t,b);	
			tensor tmpTensor = slayer.getOut();
			if (ly < depth-1)
			{		
				mySNNStructure.at(ly+1).setIn(tmpTensor,t,b,b);
			}
		}
		SNNLayer sendlayer=mySNNStructure.back();
		sendlayer.softmaxOutV2(b);
	}
}

void snnModel::BPTT()
{
	if (isTrning == true)
	{
		printf("a thread is in traing\n");
		return;
	}
	bool ret = true;
	isTrning = true;
	if (idealOut.size() == 0 || InputImageSeries.size() == 0)
	{
		std::cout << "empty input tensor" << std::endl;
		ret=false;
	}
	if (idealOut.size() != InputImageSeries.size())
	{
		std::cout << "size of out and tesnor is not equal" << std::endl;
		ret = false;
	}
	if (mySNNStructure.size() == 0)
	{
		std::cout << "empty model" << std::endl;
		ret = false;
	}
	size_t NIMG = InputImageSeries.size();
	int epoch = 0;

	std::vector<int> randInt;
	for (int i = 0;i < InputImageSeries.size();i++)
	{
		randInt.push_back(i);
	}
	float C = 0.0, lastC = 0.0;
	size_t depth = mySNNStructure.size();
	int outLen = mySNNStructure.back().getOut().getDim().dim3;
	std::thread* thrTrain = new std::thread[batchSize];
	float* vC = (float*)_mm_malloc(sizeof(float) * batchSize, AlignBytes);
	float* outValue = (float*)_mm_malloc(sizeof(float) * outLen * batchSize, AlignBytes);
	int iterK = 1;
	while ((epoch < maxEpoch) && ret)
	{
		std::cout << "------------------SIMD Epoch T " << epoch << "------------------" << std::endl;
		std::random_shuffle(randInt.begin(), randInt.end());
		clock_t st = clock();
		C = 0;
		for (int im = 0; im < NIMG; im += batchSize)
		{
			parrellelIBatchTrainDone = 0;
			for (int b = 0; b < batchSize; b++)
			{
				int realIndex = randInt.at(im + b);
				thrTrain[b] = std::thread(&snnModel::updateLossParrallel, this, InputImageSeries.at(realIndex), vC, outLen, realIndex, b);
				thrTrain[b].detach();
			}
			std::unique_lock<std::mutex> uniquelock(myMutexC);
			while (parrellelIBatchTrainDone < batchSize)
			{
				condC.wait(uniquelock);
			}
			for (int b = 0;b < batchSize;b++)
			{
				C += vC[b];
			}
			
			for (SNNLayer layer : mySNNStructure)
			{
				//layer.accumulateW(learnRate, batchSize);
				layer.UpdateLayerWBADAMW(learnRate, mOptimizer, batchSize, iterK);
			}
			iterK++;
		}
		if (lossType == 1)
		{
			C = sqrt(C / NIMG);
		}
		else
		{
			C /= (NIMG);
		}
		vloss.push_back(C);

		if ((C < minLoss) || (std::abs(C - lastC) < diffLoss))
		{
			break;
		}
		else
		{

			lastC = C;
		}
		printf("Used Time : %d\nC:%f\n", (int)(clock() - st), C);
		epoch++;
	}
	printf("end epoc--: C:%f\n",  C);
	saveToFile();
	isTrning = false;
}

float snnModel::calculateC(float* actualy, float* idealx, int sz)
{//a:label b:out

	float sum = 0.0;
	if (lossType == 1)
	{
		for (int i = 0; i < sz; i++)
		{
			sum += 0.5 * (actualy[i] - idealx[i]) * (actualy[i] - idealx[i]);
		}
		return sum / sz;
	}
	else
	{
		for (int i = 0; i < sz; i++)
		{//cross entropy
			sum += -idealx[i] * std::log(0.00001 + actualy[i]);
		}
		return sum;
	}
}
void snnModel::updateLossParrallel(tensor bImage, float* vC, int outLen, int realIndex, int b)
{

	fowardRecurrentSpikingSimd(bImage,b);
	SNNLayer sEndLayer = mySNNStructure.back();
	int OUTCLASS = sEndLayer.getOut().getDim().dim3;
	sEndLayer.setIdealOut(idealOut.at(realIndex),b);
	vC[b] = calculateC(sEndLayer.getOutAverageEXP(b), idealOut.at(realIndex), OUTCLASS);

	for (int t = 0;t < TIMESTEP;t++)
	{
		dcalculateC(idealOut.at(realIndex), sEndLayer.getOutAverageEXP(b), sEndLayer.getDCI().getDim3Data(b, t), OUTCLASS);
	}

	size_t depth = mySNNStructure.size();
	for (int sl = depth - 1;sl > 0;sl--)
	{
		SNNLayer slayer = mySNNStructure.at(sl);
		SNNLayer lastslayer = mySNNStructure.at(sl - 1);
		slayer.dLossPropagateSimdS(lastslayer.getDCI(),b);
	}

	for (SNNLayer slayer : mySNNStructure)
	{
		slayer.dWeightPropagateSimdW(b);
	}
	std::unique_lock<std::mutex> unilock(myMutexC);
	parrellelIBatchTrainDone++;
	if (parrellelIBatchTrainDone == batchSize)
	{
		condC.notify_one();
	}
	unilock.unlock();
}
template< typename  T>
void snnModel::dcalculateC(T* idealx, T* actualy, T* dyVdx, int sz)
{

	for (int i = 0; i < sz; i++)
	{
		if (lossType == 1)
		{//square
			dyVdx[i] = actualy[i] - idealx[i];
		}
		else
		{//cross entropy
			//dyVdx[i] = -x[i]/y[i] ;
			dyVdx[i] = 1;
		}
	}

}

void snnModel::setInput(float* totalImg, float* ideal, int imgNum, int blockLength, int outLength,int MNISTBLOCK,float maxValue)
{
	InputImageSeries.clear();
	idealOut.clear();
	dim tsdim(1,TIMESTEP, MNISTBLOCK);
	int offset = AlignBytes / sizeof(float);
	//float maxValue=my
	for (int imagecnt  = 0;imagecnt < imgNum;imagecnt++)
	{
		float* currentBase = totalImg + imagecnt * MNISTBLOCK;
		tensor ts;
		ts.initData(tsdim);
		float* dt = ts.getData();
		//latency(currentBase, MNISTBLOCK, dt, TAO, VTHR);
		myEnCoder.encodeInput(currentBase, MNISTBLOCK, dt, maxValue);
		InputImageSeries.push_back(ts);

		idealOut.push_back(ideal+ imagecnt*AlignVec(outLength, offset));
	}
}

void snnModel::saveToFile()
{
	std::ofstream fo("./model.net", std::ios::trunc);
	fo << "SNN model" << std::endl;
	fo << "[Uthr]" << Uthr << std::endl;
	fo << "[beta]" << beta << std::endl;
	fo << "[weitInitialsd]" << weitInitialsd << std::endl;
	for (int i = 0;i < mySNNStructure.size();i++)
	{
		
		SNNLayer slayer=mySNNStructure.at(i);
		tensor W = slayer.getW();
		float* bData = slayer.getB();
		for (int j = 0;j < slayer.getInput().getDim().dim3;j++)
		{
			fo << "[Layer_" << i << "-out_"<<j << "-weights]" ;
			float* wData = W.getDim3Data(0, j);
			for (int k = 0;k < slayer.getOut().getDim().dim3;k++)
			{
				fo << wData[k] << ",";
			}
			fo << std::endl;
		}
		fo << "[Layer_" << i  << "-bias]";
		for (int j = 0;j < slayer.getOut().getDim().dim3;j++)
		{		
			fo << bData[j] << ",";
		}
		fo << std::endl;
	}
	fo.close();
}

tensor snnModel::getHiddenOutMem(int i) //{ return mySNNStructure.at(i).getOutMem(); };
{
	if (i< mySNNStructure.size() && i>=0)
	{
		return mySNNStructure.at(i).getOutMem();
	}
	else
	{
		std::cout << "get hidden mem wrong: i out of range" << std::endl;
		return mySNNStructure.back().getOutMem();
	}
}
tensor snnModel::getHiddenOut(int i) //{ return mySNNStructure.at(i).getOut(); };
{
	if (i < mySNNStructure.size() && i >= 0)
	{
		return mySNNStructure.at(i).getOut();
	}
	else
	{
		std::cout << "get hidden spike wrong: i out of range" << std::endl;
		return mySNNStructure.back().getOut();
	}
}

bool snnModel::encodeInput(float* imageInput, size_t length, float* spikeInput, float maxValue)
{
	return myEnCoder.encodeInput(imageInput, length, spikeInput,maxValue);

}