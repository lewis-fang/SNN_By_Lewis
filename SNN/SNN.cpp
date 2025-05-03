#include "SNN.h"
SNN::SNN(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
	usedDataSet = -1;
	connect(ui.pushButton_buildamodel, SIGNAL(clicked()), this, SLOT(buildDefaultModel1()), Qt::AutoConnection);
	connect(ui.pushButton_calc, SIGNAL(clicked()), this, SLOT(lauchModelCalcSimd()), Qt::AutoConnection);
	connect(ui.pushButton_train, SIGNAL(clicked()), this, SLOT(train()), Qt::AutoConnection);
	connect(ui.comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeNeuroConfig(int)), Qt::AutoConnection);
	connect(myModelConfig.uiModelConfig.spinBox_Layer0NeuronNumber, SIGNAL(editingFinished()), this, SLOT(changeNeuroConfig()), Qt::AutoConnection);
	connect(ui.pushButton_loadDataset, SIGNAL(clicked()), this, SLOT(loadMnst()), Qt::AutoConnection);

	ui.pushButton_calc->setEnabled(false);
	ui.pushButton_train->setEnabled(false);

	TIMESTEP = 25;
	loadMnst();
	initPlotBoard();

	connect(ui.actionView_Hidden_Spike, SIGNAL(triggered()), myView3, SLOT(show()), Qt::AutoConnection);
	connect(ui.actionModel_More_Config, SIGNAL(triggered()), &myModelConfig, SLOT(show()), Qt::AutoConnection);
	connect(ui.actionOptimizer, SIGNAL(triggered()),&myTrainConfig, SLOT(show()), Qt::AutoConnection);
}

SNN::~SNN()
{}
void SNN::buildDefaultModel1()
{
	mySNNModel.setBeta(ui.lineEdit_beta->text().toFloat());
	mySNNModel.setUthr(ui.lineEdit_Uther->text().toFloat());
	mySNNModel.setReset(ui.lineEdit_Reset->text().toFloat());

	mySNNModel.setBatchsize(ui.lineEdit_batchsize->text().toInt());
	mySNNModel.setWeightSD(myModelConfig.uiModelConfig.lineEdit_sd->text().toFloat());

	

	TIMESTEP=ui.lineEdit_T->text().toInt();
	mySNNModel.setT(TIMESTEP);
	mySNNModel.setTEncodeMethod(ui.comboBox->currentIndex(), TIMESTEP);
	int LAYER1 = myModelConfig.uiModelConfig.spinBox_Layer0NeuronNumber->text().toInt();
	int MNISTDIM1 = myModelConfig.uiModelConfig.spinBox_InputH->value();
	int MNISTDIM2 = myModelConfig.uiModelConfig.spinBox_InputW->value();
	int MNISTBLOCK = AlignVec(MNISTDIM1 * MNISTDIM2, AlignBytes / sizeof(float));
	int OUTCLASS= myModelConfig.uiModelConfig.spinBox_outSize->value();
	mySNNModel.buildMyDefaultSNNModel(LAYER1, MNISTDIM1,MNISTDIM2, OUTCLASS);
	ui.pushButton_calc->setEnabled(true);
	ui.pushButton_train->setEnabled(true);

	updatePlotView();
}
void SNN::loadMnst()
{
	int MNISTDIM1 = myModelConfig.uiModelConfig.spinBox_InputH->value();
	int MNISTDIM2 = myModelConfig.uiModelConfig.spinBox_InputW->value();
	int TESTNUM = myModelConfig.uiModelConfig.lineEdit_totalImageNumber->text().toInt();
	int OUTCLASS = myModelConfig.uiModelConfig.spinBox_outSize->value();
	int dataset = myModelConfig.uiModelConfig.comboBox_dataset->currentIndex();
	float maxValue = myModelConfig.uiModelConfig.lineEdit_maxValue->text().toFloat();
	int imagecnt = 0;
	switch (dataset)
	{
	case 0:
		imagecnt=myMNIST.loadMnst(TESTNUM, MNISTDIM1, MNISTDIM2, OUTCLASS, maxValue);
		usedDataSet = 0;
		break;
	case 1:
		std::cout << "it's not realized: cifar10 dataset" << std::endl;
		break;
	case 2:
		{
			int imagecnt1 = myBioDataLoader.loadBioData(TESTNUM, MNISTDIM1, MNISTDIM2, maxValue);
			int imagecnt2 = myBioDataLoader.loadBioDataIndex(TESTNUM, OUTCLASS);
			if (imagecnt1 == imagecnt2 && imagecnt1 == TESTNUM)
			{
				imagecnt = imagecnt1;
				usedDataSet = 2;
			}
			break;
		}

	default:
		std::cout << "it's not realized: unkown" << std::endl;
		break;

	}
	if (imagecnt > 0)
	{
		ui.spinBox_CALCIMAGE->setMaximum(imagecnt - 1);
	}
	
}

void SNN::lauchModelCalcSimd()
{
	int imagecnt = ui.spinBox_CALCIMAGE->value();
	if (usedDataSet != 0 && usedDataSet != 2)
	{
		std::cout << "no images imported" << std::endl;
		return;
	}
	if (mySNNModel.getModelBuilt() == false)
	{
		std::cout << "no model built" << std::endl;
		return;
	}
	int MNISTDIM1 = myModelConfig.uiModelConfig.spinBox_InputH->value();
	int MNISTDIM2 = myModelConfig.uiModelConfig.spinBox_InputW->value();
	int MNISTBLOCK=AlignVec(MNISTDIM1 * MNISTDIM2, AlignBytes / sizeof(float));
	
	int OUTCLASS= myModelConfig.uiModelConfig.spinBox_outSize->value();
	float* currentBase = NULL;
	if (usedDataSet == 0)
	{
		currentBase = myMNIST.getImage() + imagecnt * MNISTBLOCK;
	}
	else if(usedDataSet == 2)
	{
		currentBase = myBioDataLoader.getImage() + imagecnt * MNISTBLOCK;
	}
	else
	{

	}
	float maxValue = myModelConfig.uiModelConfig.lineEdit_maxValue->text().toFloat();
	dim tsdim(1,TIMESTEP, MNISTBLOCK);
	tensor ts;
	ts.initData(tsdim);
	float* dt = ts.getData();
	mySNNModel.encodeInput(currentBase, MNISTBLOCK, dt, maxValue);
	printf("latency successfully\n");

	originalImage.clear();
	SpikeImage.clear();
	
	for (int i = 0;i < MNISTDIM1;i++)
	{
		for (int j = 0;j < MNISTDIM2;j++)
		{
			float oi = *(currentBase + i * MNISTDIM2 + j);
			if (oi > 20.0/256)
			{
				QScatterDataItem dit1 = QScatterDataItem(QVector3D(j, i, -1));
				originalImage.push_back(dit1);
			}
			for (int t = 0;t < TIMESTEP;t++)
			{
				unsigned int spkie = *(dt + t * MNISTBLOCK + i* MNISTDIM2+j);
				if (spkie == 1)
				{
					QScatterDataItem dit2 = QScatterDataItem(QVector3D(j, i, t));
					SpikeImage.push_back(dit2);
					
				}
			}
		}
	}
	mySNNModel.fowardRecurrentSpikingSimd(ts,0);
	tensor outs=mySNNModel.getOut();
	tensor mem = mySNNModel.getOutMem();
	printf("--------------------------------------membrain\n");
	for (int i = 0;i < TIMESTEP;i++)
	{
		float* dt = mem.getDim3Data(0,i);
		for (int j = 0;j < OUTCLASS;j++)
		{
			printf("%f,", *(dt + j));
		}
		printf("\n");
	}
	printf("--------------------------------------spike\n");
	for (int i = 0;i < TIMESTEP;i++)
	{
		float* dt = outs.getDim3Data(0, i);
		for (int j = 0;j < OUTCLASS;j++)
		{
			printf("%f,", *(dt+j));
		}
		printf("\n");
	}
	printf("--------------------------------------OUTAVERAEGEXP\n");

	float* dto = mySNNModel.getSNNLayer(1).getOutAverageEXP(0);
	float maxclass = 0.0;
	int maxIndex = -1;
	for (int j = 0;j < OUTCLASS;j++)
	{
		printf("%f,", *(dto + j));
		if (maxclass < *(dto + j))
		{
			maxclass = *(dto + j);
			maxIndex = j;
		}
	}
	printf("\n");

	for (int i = 1;i < MNISTDIM1 * MNISTDIM2 + 1;i++)
	{
		int a = int(currentBase[i - 1]) / 128;
		printf("%c%c", '+' * a + ' ' * (1 - a), '\n' * (i % MNISTDIM2== 0));
	}
	SpikeOut.clear();
	int outLength = AlignVec(mySNNModel.getOut().getDim().dim3, AlignBytes / sizeof(float));
	for (int c = 0;c < OUTCLASS;c++)
	{
		for (int t = 0;t < TIMESTEP;t++)
		{
			if ((int)outs.getData()[t * outLength + c] == 1)
			{
				QScatterDataItem dit3 = QScatterDataItem(QVector3D(c, 1, t));
				SpikeOut.push_back(dit3);
			}
		
		}
	}
	plot3D();
	plotOutSpike(maxIndex);
	printf("----------------------\nI guess the number: %d, with Confidence: %f\n-------------------------------\n", maxIndex, maxclass);
}

void SNN::train()
{
	int imageNumber = ui.lineEdit_imagenum->text().toInt();
	int totalImage = 0;
	float maxValue = myModelConfig.uiModelConfig.lineEdit_maxValue->text().toFloat();
	if ((usedDataSet==0|| usedDataSet==2)&& mySNNModel.getModelBuilt() == true)
	{

		int MNISTDIM1 = myModelConfig.uiModelConfig.spinBox_InputH->value();
		int MNISTDIM2 = myModelConfig.uiModelConfig.spinBox_InputW->value();
		int MNISTBLOCK = AlignVec(MNISTDIM1 * MNISTDIM2, AlignBytes / sizeof(float));
		int OUTCLASS = myModelConfig.uiModelConfig.spinBox_outSize->value();
		

		mySNNModel.setLearnRate(ui.lineEdit_learnrate->text().toFloat());
		mySNNModel.setMinLoss(ui.lineEdit_losswindow->text().toFloat());
		mySNNModel.setDiffMinLoss(myTrainConfig.uiTrainConfig.lineEdit_difflosswindow->text().toFloat());
		mySNNModel.setMaxEpoch(ui.lineEdit_maxepoch->text().toInt());

		if (usedDataSet == 0)
		{
			mySNNModel.setInput(myMNIST.getImage(), myMNIST.getImageIndexVector(), imageNumber, MNISTBLOCK, OUTCLASS, MNISTBLOCK, maxValue);
		}
		else if(usedDataSet==2)
		{
			mySNNModel.setInput(myBioDataLoader.getImage(), myBioDataLoader.getImageIndex(), imageNumber, MNISTBLOCK, OUTCLASS, MNISTBLOCK, maxValue);

		}
		mySNNModel.createTrainThread();
	}
	else
	{
		std::cout << "no images imported" << std::endl;
	}
}


void SNN::plot3D() 
{
	scatterSeries1->dataProxy()->removeItems(0, scatterSeries1->dataProxy()->itemCount());
	scatterSeries2->dataProxy()->removeItems(0, scatterSeries2->dataProxy()->itemCount());
	for (int i = 0; i < originalImage.size(); ++i)
	{
		scatterSeries1->dataProxy()->addItem(originalImage.at(i));
	}

	
	for (int i = 0; i < SpikeImage.size(); ++i)
	{
		scatterSeries2->dataProxy()->addItem(SpikeImage.at(i));
	}


	
}
void SNN::plotOutSpike(int number)
{
	int OUTCLASS = myModelConfig.uiModelConfig.spinBox_outSize->value();

	memSeries->clear();
	outSpikeScatters->clear();
	memSeries->setColor(Qt::black);
	
	float* spike = mySNNModel.getOut().getData();
	float* mem= mySNNModel.getOutMem().getData();
	int outLength = AlignVec(mySNNModel.getOut().getDim().dim3,AlignBytes/sizeof(float));
	float maxm = mem[number], minm = mem[number];
	for (int i = 0;i < TIMESTEP;i++)
	{
		memSeries->append(QPointF(i, mem[i* outLength +number]));
		if (maxm < mem[i * outLength + number]) maxm = mem[i * outLength + number];
		if (minm > mem[i * outLength + number]) minm = mem[i * outLength + number];
	}
	myView2->chart()->axisY()->setMax(maxm);
	myView2->chart()->axisY()->setMin(minm);
	for (int c = 0;c < OUTCLASS;c++)
	{
		for (int t = 0;t < TIMESTEP;t++)
		{
			if ((int)spike[t * outLength + c] == 1)
			{
				outSpikeScatters->append(QPointF(t,c));
			}

		}
	}

	float* hiddenSpike = mySNNModel.getHiddenOut(0).getData();
	int outHiddenLength = AlignVec(mySNNModel.getHiddenOut(0).getDim().dim3, AlignBytes / sizeof(float));
	hiddenSpikeScatters->clear();
	for (int c = 0;c < mySNNModel.getHiddenOut(0).getDim().dim3;c++)
	{
		for (int t = 0;t < TIMESTEP;t++)
		{
			if ((int)hiddenSpike[t * outHiddenLength + c] == 1)
			{
				hiddenSpikeScatters->append(QPointF(t, c));
			}
		}
	}
}

void SNN::initPlotBoard()
{
	scatter = new Q3DScatter();
	scatterContainer = QWidget::createWindowContainer(scatter);
	ui.verticalLayout_7->addWidget(scatterContainer);
	scatterSeries1 = new QScatter3DSeries();
	scatterSeries2 = new QScatter3DSeries();
	scatter->addSeries(scatterSeries1);
	scatter->addSeries(scatterSeries2);

	scatterSeries1->setItemSize(0.1f);
	scatterSeries1->setBaseColor(Qt::red);
	scatterSeries2->setItemSize(0.1f);
	scatterSeries2->setBaseColor(Qt::gray);

	QValue3DAxis* scatterAxisX = new QValue3DAxis();
	scatterAxisX->setTitle("X");
	scatterAxisX->setRange(0, 30);
	scatterAxisX->setTitleVisible(true);
	scatter->setAxisX(scatterAxisX);

	QValue3DAxis* scatterAxisY = new QValue3DAxis();
	scatterAxisY->setTitle("Y");
	scatterAxisY->setRange(0, 30);
	scatterAxisY->setTitleVisible(true);
	scatterAxisY->setReversed(true);
	scatter->setAxisY(scatterAxisY);

	QValue3DAxis* scatterAxisZ = new QValue3DAxis();
	scatterAxisZ->setTitle("T");
	scatterAxisZ->setRange(-2, TIMESTEP+1);
	scatterAxisZ->setTitleVisible(true);
	scatter->setAxisZ(scatterAxisZ);
	scatter->setOrthoProjection(true);
	scatter->setHorizontalAspectRatio(0.5);
	scatter->setAspectRatio(2);

	myView2 = new QChartView;
	myView4 = new QChartView;

	myView2->setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
	myView4->setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
	ui.verticalLayout->addWidget(myView2);
	ui.verticalLayout_3->addWidget(myView4);
	// ui.verticalLayout_7->addWidget(myView4);
	memSeries = new QLineSeries;
	outSpikeScatters = new QScatterSeries;
	outSpikeScatters->setUseOpenGL(true);
	outSpikeScatters->setMarkerSize(7);
	outSpikeScatters->setColor(Qt::red);

	myView2->chart()->addSeries(memSeries);
	myView4->chart()->addSeries(outSpikeScatters);

	myView2->chart()->setTitle("Model Out Mem");
	myView4->chart()->setTitle("Out Spike Pattern");

	myView2->chart()->legend()->setVisible(false);
	myView4->chart()->legend()->setVisible(false);
	myView2->chart()->createDefaultAxes();
	myView4->chart()->createDefaultAxes();

	QValueAxis* axisY1 = new QValueAxis;
	axisY1->setMin(-1);
	axisY1->setMax(10);
	axisY1->setTickCount(12);
	axisY1->setLabelFormat("%d");
	QValueAxis* axisY2 = new QValueAxis;
	axisY2->setMin(-1);
	axisY2->setMax(10);
	axisY2->setTickCount(12);
	axisY2->setLabelFormat("%d");
	myView4->chart()->removeAxis(myView4->chart()->axisY());
	myView4->chart()->addAxis(axisY1,  Qt::AlignLeft);
	myView4->chart()->addAxis(axisY2, Qt::AlignRight);
	outSpikeScatters->attachAxis(axisY1);

	myView2->chart()->axisX()->setMax(TIMESTEP + 1);
	myView4->chart()->axisX()->setMax(TIMESTEP + 1);
	myView2->chart()->axisX()->setMin(-1);
	myView4->chart()->axisX()->setMin(-1);

	myView3 = new QChartView;//hidden spike
	hiddenSpikeScatters = new QScatterSeries;
	hiddenSpikeScatters->setUseOpenGL();
	hiddenSpikeScatters->setMarkerSize(5.0);
	hiddenSpikeScatters->setColor(Qt::black);
	myView3->chart()->addSeries(hiddenSpikeScatters);
	myView3->chart()->setTitle("Hidden Spike Pattern");

	myView3->chart()->legend()->setVisible(false);
	myView3->chart()->createDefaultAxes();

	QValueAxis* axisY3 = new QValueAxis;
	int LAYER1 = myModelConfig.uiModelConfig.spinBox_Layer0NeuronNumber->text().toInt();
	axisY3->setMin(-1);
	axisY3->setMax(LAYER1);
	axisY3->setTickCount(12);
	axisY3->setLabelFormat("%d");
	myView3->chart()->removeAxis(myView3->chart()->axisY());
	myView3->chart()->addAxis(axisY3, Qt::AlignLeft);
	hiddenSpikeScatters->attachAxis(axisY3);
	myView3->chart()->axisX()->setMax(TIMESTEP + 1);
	myView3->chart()->axisX()->setMin(-1);
	myView3->setWindowFlags(Qt::WindowStaysOnTopHint);
	myView3->setMinimumSize(QSize(800  , 400));
	//myView3->showMaximized();
}
void SNN::showHiddenSpikeOut()
{

}

void SNN::changeNeuroConfig(int)
{
	if (ui.comboBox->currentIndex() == 1)
	{
		TIMESTEP = 8;
	}
	else if (ui.comboBox->currentIndex() == 0)
	{
		TIMESTEP = 25;
	}
	else if (ui.comboBox->currentIndex() == 2)
	{
		TIMESTEP = 64;
	}
	ui.lineEdit_T->setText(QString::number(TIMESTEP));
	myView2->chart()->axisX()->setMax(TIMESTEP + 1);
	myView3->chart()->axisX()->setMax(TIMESTEP + 1);
	int LAYER1 = myModelConfig.uiModelConfig.spinBox_Layer0NeuronNumber->text().toInt();

	myView3->chart()->axisY()->setMax(LAYER1 + 1);
	myView4->chart()->axisX()->setMax(TIMESTEP + 1);
	scatter->axisZ()->setMax(TIMESTEP + 1);
}

void SNN::updatePlotView()
{
	//ui.lineEdit_T->setText(QString::number(TIMESTEP));
	myView2->chart()->axisX()->setMax(TIMESTEP + 1);
	myView3->chart()->axisX()->setMax(TIMESTEP + 1);	
	myView4->chart()->axisX()->setMax(TIMESTEP + 1);
	scatter->axisZ()->setMax(TIMESTEP + 1);
	scatter->axisX()->setMax(myModelConfig.uiModelConfig.spinBox_InputW->value());
	scatter->axisY()->setMax(myModelConfig.uiModelConfig.spinBox_InputH->value());
}

