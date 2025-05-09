#include"loadMNIST.h"


MNISTLoader::~MNISTLoader()
{
	if (mnistTEST != NULL)
	{
		_mm_free(mnistTEST);
	}
	if (mnistTESTIndex != NULL)
	{
		_mm_free(mnistTESTIndex);
	}
	if (mnistTESTIndexVector != NULL)
	{
		_mm_free(mnistTESTIndexVector);
	}	
}
MNISTLoader::MNISTLoader()
{
	mnistTEST = NULL;
	mnistTESTIndex = NULL;
	mnistTESTIndexVector = NULL;
	imagecnt = 0;
	hadData = false;
};
int MNISTLoader::loadMnst(int TESTNUM,int MNISTDIM1,int MNISTDIM2,int OUTCLASS,float MaxValue)
{
	int MNISTBLOCK = AlignVec(MNISTDIM1 * MNISTDIM2, AlignBytes / sizeof(float));
	if (mnistTEST != NULL)
	{
		_mm_free(mnistTEST);
	}
	if (mnistTESTIndex != NULL)
	{
		_mm_free(mnistTESTIndex);
	}
	if (mnistTESTIndexVector != NULL)
	{
		_mm_free(mnistTESTIndexVector);
	}
	mnistTEST = (float*)_mm_malloc(TESTNUM * MNISTBLOCK * sizeof(float), AlignBytes);
	mnistTESTIndex = (float*)_mm_malloc(TESTNUM * sizeof(float), AlignBytes);
	mnistTESTIndexVector = (float*)_mm_malloc(TESTNUM * AlignVec(OUTCLASS, AlignBytes / sizeof(float)) * sizeof(float), AlignBytes);

	memset(mnistTESTIndex, 0, TESTNUM * sizeof(float));
	memset(mnistTEST, 0, TESTNUM * MNISTBLOCK * sizeof(float));
	memset(mnistTESTIndexVector, 0, TESTNUM * AlignVec(OUTCLASS, AlignBytes / sizeof(float)) * sizeof(float));

	QString fileName = QFileDialog::getOpenFileName(this, tr("import mnist train"), "", tr("CSV(*.csv)")); //ѡ��·��
	std::cout << fileName.toLocal8Bit().data() << std::endl;
	QFile Wts(fileName);
	bool GoodLine = true;
	imagecnt = 0;
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
				mnistTESTIndex[imagecnt] = qstrlist.at(0).toFloat();
				for (int i = 1;i < MNISTDIM1 * MNISTDIM2 + 1;i++)
				{
					currentBase[i - 1] = qstrlist.at(i).toFloat() / MaxValue;
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
			std::cout << "MNST images are imported successfully!-->" << imagecnt << std::endl;
			hadData = true;
			convertOutIndex2Vector(OUTCLASS);
		}
		else
		{
			imagecnt = 0;
		}
	}
	return imagecnt;
}

void MNISTLoader::convertOutIndex2Vector(int OUTCLASS)
{
	int offset = AlignBytes / sizeof(float);
	for (int i = 0;i < imagecnt;i++)
	{
		float* out = mnistTESTIndexVector + i * AlignVec(OUTCLASS, offset);
		int outIndex = mnistTESTIndex[i];
		if (outIndex < OUTCLASS)
		{
			out[outIndex] = 1.0;
		}
		else
		{
			std::cout << "unexpected out index" << std::endl;
		}
	}
}