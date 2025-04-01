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
		//float* currentBase = mnistTEST + (imagecnt-1) * MNISTBLOCK;
		//float* spikeInput = (float*)_mm_malloc(MNISTBLOCK * TIMESTEP*sizeof(float), AlignBytes);
		//snnModel::latency(currentBase, MNISTBLOCK, spikeInput, TIMESTEP,100, 3);
		//printf("latency successfully\n");
	}
}

void SNN::lauchModelCalcSimd()
{
	int imagecnt = ui.spinBox_CALCIMAGE->value();
	float* currentBase = mnistTEST + imagecnt * MNISTBLOCK;
	//float* spikeInput = (float*)_mm_malloc(MNISTBLOCK * TIMESTEP * sizeof(float), AlignBytes);
	

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
	QChartView* myView1=new QChartView;
	QChartView* myView2 = new QChartView;
	QChartView* myView3 = new QChartView;
	myView1->setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
	myView2->setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
	myView3->setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
	ui.verticalLayout->addWidget(myView1);
	ui.verticalLayout_2->addWidget(myView2);
	ui.verticalLayout_3->addWidget(myView3);

	QLineSeries* inSeries = new QLineSeries;
	QLineSeries* memSeries = new QLineSeries;
	QLineSeries* spikeSeries = new QLineSeries;

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
	myView1->chart()->addSeries(inSeries);
	myView2->chart()->addSeries(memSeries);
	myView3->chart()->addSeries(spikeSeries);

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