#include "SNN.h"


SNN::SNN(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
	connect(ui.pushButton_buildamodel, SIGNAL(clicked()), this, SLOT(buildDefaultModel1()), Qt::AutoConnection);
	connect(ui.pushButton_calc, SIGNAL(clicked()), this, SLOT(lauchModelCalcSimd()), Qt::AutoConnection);
//	connect(ui.pushButton_singleNeuro, SIGNAL(clicked()), this, SLOT(checkSingleNeuro()), Qt::AutoConnection);
	connect(ui.pushButton_train, SIGNAL(clicked()), this, SLOT(train()), Qt::AutoConnection);
	connect(ui.comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeNeuroConfig(int)), Qt::AutoConnection);
	ui.pushButton_calc->setEnabled(false);
	ui.pushButton_train->setEnabled(false);
	mnistTEST=(float*)_mm_malloc(TESTNUM * MNISTBLOCK*sizeof(float), AlignBytes);
	mnistTESTIndex = (float*)_mm_malloc(TESTNUM  * sizeof(float), AlignBytes);
	charMnistTEST = (char*)_mm_malloc(TESTNUM * MNISTBLOCK * sizeof(char), AlignBytes);
	memset(mnistTESTIndex, 0, TESTNUM * sizeof(float));
	memset(mnistTEST, 0, TESTNUM * MNISTBLOCK * sizeof(float));
	memset(charMnistTEST, 0, TESTNUM * MNISTBLOCK * sizeof(char));
	TIMESTEP = 25;
	loadMnst();
	initPlotBoard();

	connect(ui.actionView_Hidden_Spike, SIGNAL(triggered()), myView3, SLOT(show()), Qt::AutoConnection);

}

SNN::~SNN()
{}
void SNN::buildDefaultModel1()
{
	mySNNModel.setBeta(ui.lineEdit_beta->text().toFloat());
	mySNNModel.setUthr(ui.lineEdit_Uther->text().toFloat());
	mySNNModel.setReset(ui.lineEdit_Reset->text().toFloat());

	mySNNModel.setBatchsize(ui.lineEdit_batchsize->text().toInt());
	mySNNModel.setWeightSD(ui.lineEdit_sd->text().toFloat());

	mySNNModel.setTEncodeMethod(ui.comboBox->currentIndex());
	if (ui.comboBox->currentIndex() == 1)
	{
		TIMESTEP = 8;
	}
	else if(ui.comboBox->currentIndex() == 0)
	{
		TIMESTEP = 25;
	}
	else if (ui.comboBox->currentIndex() ==2)
	{
		TIMESTEP = 64;
	}
	ui.lineEdit_T->setText(QString::number(TIMESTEP));
	mySNNModel.setT(TIMESTEP);
	
	mySNNModel.buildMyDefaultSNNModel();
	ui.pushButton_calc->setEnabled(true);
	ui.pushButton_train->setEnabled(true);
}
void SNN::loadMnst()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("import mnist train"), "", tr("CSV(*.csv)")); //Ñ¡ÔñÂ·¾¶
	std::cout << fileName.toLocal8Bit().data() << std::endl;
	QFile Wts(fileName);
	bool GoodLine = true;
	int imagecnt = 0;
	if (Wts.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		QString qLine;
		QTextStream qstream(&Wts);
		while (!qstream.atEnd() && GoodLine)
		{
			qLine = qstream.readLine();
			QStringList qstrlist = qLine.split(',', QString::SkipEmptyParts);
			if (qstrlist.size() == MNISTDIM1 * MNISTDIM2 + 1)
			{
				float* currentBase = mnistTEST + imagecnt * MNISTBLOCK;
				char* currentCharBase = charMnistTEST + imagecnt * MNISTBLOCK;

				mnistTESTIndex[imagecnt] = qstrlist.at(0).toFloat();
				for (int i = 1;i < MNISTDIM1 * MNISTDIM2 + 1;i++)
				{
					currentBase[i - 1] = qstrlist.at(i).toFloat();
					currentCharBase[i - 1] = qstrlist.at(i).toUShort();
				}
				imagecnt++;
			}
			else
			{
				GoodLine = false;
				break;
			}
			if (imagecnt == TESTNUM)
			{
				break;
			}
		}
		if (GoodLine)
		{
			std::cout << "MNST images are imported successfully!-->"<< imagecnt << std::endl;
			ui.spinBox_CALCIMAGE->setMaximum(imagecnt - 1);
		}
	}
}

void SNN::lauchModelCalcSimd()
{
	int imagecnt = ui.spinBox_CALCIMAGE->value();
	float* currentBase = mnistTEST + imagecnt * MNISTBLOCK;
	dim tsdim(1,TIMESTEP, MNISTBLOCK);
	tensor ts;
	ts.initData(tsdim);
	float* dt = ts.getData();
	mySNNModel.encodeInput(currentBase, MNISTBLOCK, dt);
	printf("latency successfully\n");
	//for (int i = 0;i < MNISTBLOCK;i++)
	//{
	//	for (int t = 0;t < TIMESTEP;t++)
	//	{
	//		unsigned int spkie = *(dt + t * MNISTBLOCK + i);
	//		printf("%c,", spkie * '-');
	//	}
	//	printf("\n");
	//}
	originalImage.clear();
	SpikeImage.clear();
	
	for (int i = 0;i < MNISTDIM1;i++)
	{
		for (int j = 0;j < MNISTDIM2;j++)
		{
			float oi = *(currentBase + i * MNISTDIM2 + j);
			if (oi > 20)
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

	mySNNModel.setLearnRate(ui.lineEdit_learnrate->text().toFloat());
	mySNNModel.setMinLoss(ui.lineEdit_losswindow->text().toFloat());
	mySNNModel.setDiffMinLoss(ui.lineEdit_difflosswindow->text().toFloat());
	mySNNModel.setMaxEpoch(ui.lineEdit_maxepoch->text().toInt());

	mySNNModel.setInput(mnistTEST,mnistTESTIndex, imageNumber, MNISTBLOCK,OUTCLASS);
	mySNNModel.createTrainThread();

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
	axisY3->setMin(-1);
	axisY3->setMax(LAYER1);
	axisY3->setTickCount(12);
	axisY3->setLabelFormat("%d");
	QValueAxis* axisY4 = new QValueAxis;
	axisY4->setMin(-1);
	axisY4->setMax(LAYER1);
	axisY4->setTickCount(12);
	axisY4->setLabelFormat("%d");
	myView3->chart()->removeAxis(myView3->chart()->axisY());
	myView3->chart()->addAxis(axisY3, Qt::AlignLeft);
	myView3->chart()->addAxis(axisY4, Qt::AlignRight);
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
	myView4->chart()->axisX()->setMax(TIMESTEP + 1);
	scatter->axisZ()->setMax(TIMESTEP + 1);
}