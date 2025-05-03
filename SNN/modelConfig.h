#pragma once
#include "ui_uModelConfig.h"
#include <QtWidgets/QMainWindow>
class ModelConfig :public QWidget
{
	Q_OBJECT
public:
	Ui::ModelConfig uiModelConfig;
	ModelConfig()
	{
		uiModelConfig.setupUi(this);
		setWindowFlags(Qt::WindowStaysOnTopHint);
		connect(uiModelConfig.comboBox_dataset, SIGNAL(currentIndexChanged(int)), this, SLOT(selectDataset(int)), Qt::AutoConnection);
	};
public slots:
	void selectDataset(int d)
	{
		switch (d)
		{
		case 0:
			uiModelConfig.spinBox_InputH->setValue(28);
			uiModelConfig.spinBox_InputW->setValue(28);
			uiModelConfig.lineEdit_totalImageNumber->setText(QString::number(10000));
			uiModelConfig.spinBox_outSize->setValue(10);
			uiModelConfig.lineEdit_maxValue->setText(QString::number(256));
			break;
		case 1:
			break;
		case 2:
			uiModelConfig.spinBox_InputH->setValue(16);
			uiModelConfig.spinBox_InputW->setValue(32);
			uiModelConfig.lineEdit_totalImageNumber->setText(QString::number(120));
			uiModelConfig.spinBox_outSize->setValue(5);
			uiModelConfig.lineEdit_maxValue->setText(QString::number(36000));
			break;
		default:
			break;
		}
	};
};



