#include"loadBioData1.h"

BioDataLoader::~BioDataLoader()
{
	if (Data != NULL)
	{
		_mm_free(Data);
	}
	if (DataIndex != NULL)
	{
		_mm_free(DataIndex);
	}
	
}
BioDataLoader::BioDataLoader()
{
	Data = NULL;
	DataIndex = NULL;
	imagecnt = 0;
	hadData = false;
	
};
int BioDataLoader::loadBioData(int TESTNUM, int MNISTDIM1, int MNISTDIM2, float MaxValue)
{
	if (Data != NULL)
	{
		_mm_free(Data);
	}
	int nc = NULLCOLLUMN;
	int nhr = NULLHEADERROW;

	int npr = NULLPATITIONROW;
	int MNISTBLOCK = AlignVec(MNISTDIM1 * MNISTDIM2, AlignBytes / sizeof(float));

	Data = (float*)_mm_malloc(TESTNUM * MNISTBLOCK * sizeof(float), AlignBytes);

	memset(Data, 0, TESTNUM * MNISTBLOCK * sizeof(float));

	QString fileName = QFileDialog::getOpenFileName(this, tr("import mnist train"), "", tr("CSV(*.csv)")); //选择路径
	std::cout << fileName.toLocal8Bit().data() << std::endl;
	QFile Wts(fileName);
	bool GoodLine = false;
	imagecnt = 0;
	int rowcnt = 0;
	if (Wts.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		QString qLine;
		QTextStream qstream(&Wts);
		while (nhr--)
		{
			qLine = qstream.readLine();
		}
		while (!qstream.atEnd())
		{
			qLine = qstream.readLine();
			QStringList qstrlist = qLine.split(',', QString::SkipEmptyParts);
			if (qstrlist.size() == MNISTDIM2 + nc+1)
			{
				float* currentBase = Data + imagecnt * MNISTBLOCK;
				for (int i = 0;i < MNISTDIM2;i++)
				{
					if (i!= (BADPOINT-1))
					{
						currentBase[i+(rowcnt % MNISTDIM1)* MNISTDIM2] = qstrlist.at(i + nc).toFloat()/ MaxValue;
					}
				
				}
				rowcnt++;
			}
			else
			{
				continue;
			}
		
			if (rowcnt % MNISTDIM1== 0)
			{
				imagecnt++;
				int kaka = npr;
				while (kaka--)
				{
					qLine = qstream.readLine();
				}
			}
			//std::cout << rowcnt << "," << imagecnt << imagecnt << std::endl;
			if (imagecnt == TESTNUM)
			{
				GoodLine = true;
				break;
			}
		}
		if (GoodLine || imagecnt > 0)
		{
			std::cout << "bioData images are imported successfully!-->" << imagecnt << std::endl;
			hadData = true;
		}
		else
		{
			std::cout << "bioData images are imported FAIL!-->" << imagecnt << std::endl;
			imagecnt = 0;
		}
	}
	return imagecnt;
}

int BioDataLoader::loadBioDataIndex(int TESTNUM, int OUTCLASS)
{
	if (DataIndex != NULL)
	{
		_mm_free(DataIndex);
	}
	DataIndex = (float*)_mm_malloc(TESTNUM * sizeof(float), AlignBytes);
	memset(DataIndex, 0, TESTNUM * sizeof(float));

	QString fileName = QFileDialog::getOpenFileName(this, tr("import bio train"), "", tr("CSV(*.csv)")); //选择路径
	std::cout << fileName.toLocal8Bit().data() << std::endl;
	QFile Wts(fileName);
	bool GoodLine = true;
    int	indexImagecnt = 0;
	if (Wts.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		QString qLine;
		QTextStream qstream(&Wts);
		qLine = qstream.readLine();
		while (!qstream.atEnd() && GoodLine)
		{
			qLine = qstream.readLine();
			QStringList qstrlist = qLine.split(',', QString::SkipEmptyParts);
			if (qstrlist.size() == OUTCLASS )
			{
				float* currentBase = DataIndex + indexImagecnt * AlignVec(OUTCLASS,AlignBytes/sizeof(float));
				for (int i = 0;i < OUTCLASS;i++)
				{

					currentBase[i] = qstrlist.at(i).toFloat();
				}
				indexImagecnt++;
			}
			else
			{
				GoodLine = false;
				break;
			}
			if (indexImagecnt == TESTNUM)
			{
				break;
			}
		}
		if (GoodLine)
		{
			std::cout << "bioData Indexs are imported successfully!-->" << indexImagecnt << std::endl;
			hadData &= true;
			//ui.spinBox_CALCIMAGE->setMaximum(imagecnt - 1);
		}
		else
		{
			std::cout << "bioData Indexs are imported FAIL!-->" << imagecnt << std::endl;
			indexImagecnt = 0;
		}
	}
	return indexImagecnt;
}