#include "SNN.h"


SNN::SNN(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
	connect(ui.pushButton_buildamodel, SIGNAL(clicked()), this, SLOT(buildDefaultModel1()), Qt::AutoConnection);
	connect(ui.pushButton_calc, SIGNAL(clicked()), this, SLOT(lauchModelCalcSimd()), Qt::AutoConnection);
	connect(ui.pushButton_singleNeuro, SIGNAL(clicked()), this, SLOT(checkSingleNeuro()), Qt::AutoConnection);
	connect(ui.pushButton_train, SIGNAL(clicked()), this, SLOT(train()), Qt::AutoConnection);
	ui.pushButton_calc->setEnabled(false);
	ui.pushButton_train->setEnabled(false);
	mnistTEST=(float*)_mm_malloc(TESTNUM * MNISTBLOCK*sizeof(float), AlignBytes);
	mnistTESTIndex = (float*)_mm_malloc(TESTNUM  * sizeof(float), AlignBytes);
	loadMnst();

	 scatter = new Q3DScatter();
	 scatterContainer = QWidget::createWindowContainer(scatter);
	ui.verticalLayout_7->addWidget(scatterContainer);
	 scatterSeries1 = new QScatter3DSeries();
	 scatterSeries2 = new QScatter3DSeries();
	 scatter->addSeries(scatterSeries1);
	 scatter->addSeries(scatterSeries2);
	 scatter2= new Q3DScatter();
	 scatterContainer2 = QWidget::createWindowContainer(scatter2);
	 ui.verticalLayout_7->addWidget(scatterContainer2);
	 scatterSeries3 = new QScatter3DSeries();
	 scatter2->addSeries(scatterSeries3);

	 myView1 = new QChartView;
	 myView2 = new QChartView;
	 myView3 = new QChartView;
	 myView1->setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
	 myView2->setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
	 myView3->setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
	 ui.verticalLayout->addWidget(myView1);
	 ui.verticalLayout_2->addWidget(myView2);
	 ui.verticalLayout_3->addWidget(myView3);
	
	 inSeries = new QLineSeries;
	 memSeries = new QLineSeries;
	  spikeSeries = new QLineSeries;

	myView1->chart()->addSeries(inSeries);
	 myView2->chart()->addSeries(memSeries);
	myView3->chart()->addSeries(spikeSeries);

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
			if (qstrlist.size() == MNISTDIM * MNISTDIM + 1)
			{
				float* currentBase = mnistTEST + imagecnt * MNISTBLOCK;
				mnistTESTIndex[imagecnt] = qstrlist.at(0).toFloat();
				for (int i = 1;i < MNISTDIM * MNISTDIM + 1;i++)
				{
					currentBase[i - 1] = qstrlist.at(i).toFloat();
					//if (imagecnt > TESTNUM - 20)
					//{
					//	int a = int(currentBase[i - 1]) / 128;
					//	printf("%c%c", '+' * a + ' ' * (1 - a), '\n' * (i % MNISTDIM == 0));
					//}			
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
	snnModel::latency(currentBase, MNISTBLOCK, dt, TIMESTEP, TAO, 5);
	printf("latency successfully\n");
	for (int i = 0;i < MNISTBLOCK;i++)
	{
		for (int t = 0;t < TIMESTEP;t++)
		{
			unsigned int spkie = *(dt + t * MNISTBLOCK + i);
			printf("%c,", spkie * '-');
		}
		printf("\n");
	}
	originalImage.clear();
	SpikeImage.clear();
	for (int i = 0;i < MNISTDIM;i++)
	{
		for (int j = 0;j < MNISTDIM;j++)
		{
			float oi = *(currentBase + i * MNISTDIM + j);
			if (oi > 20)
			{
				QScatterDataItem dit1 = QScatterDataItem(QVector3D(j, i, 0));
				originalImage.push_back(dit1);
			}
			for (int t = 0;t < TIMESTEP;t++)
			{
				unsigned int spkie = *(dt + t * MNISTBLOCK + i* MNISTDIM+j);
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

	for (int i = 1;i < MNISTDIM * MNISTDIM + 1;i++)
	{
		int a = int(currentBase[i - 1]) / 128;
		printf("%c%c", '+' * a + ' ' * (1 - a), '\n' * (i % MNISTDIM == 0));
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
void SNN::checkSingleNeuro()
{
	float mem[TIMESTEP];
	float spike[TIMESTEP];
	SNNLayer aSNNlayer;
	float input[TIMESTEP];
	memset(input, 0, sizeof(float) * TIMESTEP);
	for (int i = 1;i < TIMESTEP;i++)
	{
		input[i] =0.21;
	}
	aSNNlayer.checkSingleNeuro(input,mem, spike);
	inSeries->clear();
	memSeries->clear();
	spikeSeries->clear();

	inSeries->setColor(Qt::green);
	memSeries->setColor(Qt::black);
	spikeSeries->setColor(Qt::red);

	for (int i = 0;i < TIMESTEP;i++)
	{
		inSeries->append(QPointF(i, input[i]));
		memSeries->append( QPointF(i, mem[i]));
		spikeSeries->append(QPointF(i, spike[i]));
		printf("%d\t%f\t%f\t%f\n", i, input[i], mem[i], spike[i]);
	//	std::cout << i << "\t" << input[i] <<"\t" << mem[i] << "\t" << spike[i] << std::endl;
	}
	//myView1->chart()->addSeries(inSeries);
	//myView2->chart()->addSeries(memSeries);
//	myView3->chart()->addSeries(spikeSeries);

	myView1->chart()->legend()->setVisible(false);
	myView2->chart()->legend()->setVisible(false);
	myView3->chart()->legend()->setVisible(false);
	myView1->chart()->createDefaultAxes();
	myView2->chart()->createDefaultAxes();
	myView3->chart()->createDefaultAxes();
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
	scatterSeries3->dataProxy()->removeItems(0, scatterSeries3->dataProxy()->itemCount());
	for (int i = 0; i < originalImage.size(); ++i)
	{
		scatterSeries1->dataProxy()->addItem(originalImage.at(i));
	}
	scatterSeries1->setItemSize(0.1);
	scatterSeries1->setBaseColor(Qt::gray);
	
	//scatter->addSeries(scatterSeries1);

	for (int i = 0; i < SpikeImage.size(); ++i)
	{
		scatterSeries2->dataProxy()->addItem(SpikeImage.at(i));
	}

	scatterSeries2->setItemSize(0.1);
	scatterSeries2->setBaseColor(Qt::red);
	//scatter->addSeries(scatterSeries2);

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
	scatterAxisZ->setRange(0, 28);
	scatterAxisZ->setTitleVisible(true);
	scatter->setAxisZ(scatterAxisZ);
	scatter->setOrthoProjection(true);
	scatter->setHorizontalAspectRatio(0.5);
	scatter->setAspectRatio(2);
	
	for (int i = 0; i < SpikeOut.size(); ++i)
	{
		scatterSeries3->dataProxy()->addItem(SpikeOut.at(i));
	}
	QValue3DAxis* scatterAxisX2 = new QValue3DAxis();
	scatterAxisX2->setTitle("NUMBER");
	scatterAxisX2->setRange(0, 12);
	scatterAxisX2->setTitleVisible(true);
	scatter2->setAxisX(scatterAxisX2);

	QValue3DAxis* scatterAxisY2 = new QValue3DAxis();
	scatterAxisY2->setTitle("SPIKE");
	scatterAxisY2->setTitleVisible(true);
	scatterAxisY2->setRange(0, 1.5);
	scatter2->setAxisY(scatterAxisY2);

	scatter2->setAxisZ(scatterAxisZ);
	scatter2->setOrthoProjection(true);
}
void SNN::plotOutSpike(int number)
{
	memSeries->clear();
	spikeSeries->clear();
	memSeries->setColor(Qt::black);
	spikeSeries->setColor(Qt::red);

	float* spike = mySNNModel.getOut().getData();
	float* mem= mySNNModel.getOutMem().getData();
	int outLength = AlignVec(mySNNModel.getOut().getDim().dim3,AlignBytes/sizeof(float));
	float maxm = mem[number], minm = mem[number];
	for (int i = 0;i < TIMESTEP;i++)
	{
		memSeries->append(QPointF(i, mem[i* outLength +number]));
		spikeSeries->append(QPointF(i, spike[i * outLength + number]));
		if (maxm < mem[i * outLength + number]) maxm = mem[i * outLength + number];
		if (minm > mem[i * outLength + number]) minm = mem[i * outLength + number];
	}


	myView2->chart()->legend()->setVisible(false);
	myView3->chart()->legend()->setVisible(false);

	myView2->chart()->createDefaultAxes();
	myView3->chart()->createDefaultAxes();

	myView2->chart()->axisX()->setMax(TIMESTEP + 1);
	myView3->chart()->axisX()->setMax(TIMESTEP + 1);
	myView2->chart()->axisY()->setMax(maxm);
	myView2->chart()->axisY()->setMin(minm);
	myView3->chart()->axisY()->setMax(1.1);
}