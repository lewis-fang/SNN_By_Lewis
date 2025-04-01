#pragma once

#include"dataDefine.h"
#include <QtWidgets/QMainWindow>
#include "ui_SNN.h"
#include<iostream>
#include<immintrin.h>
#include"SNNModel.h"
#include"SNNLayer.h"
#include<QtCharts/qchartview.h>
#include<QtCharts/qscatterseries.h>
#include<QtCharts/qlineseries.h>
#include <QtCharts/qvalueaxis.h>
#include<QFileDialog>
#include<QTextStream>
QT_CHARTS_USE_NAMESPACE
class SNN : public QMainWindow
{
    Q_OBJECT

public:
    SNN(QWidget *parent = nullptr);
    ~SNN();

private:
    Ui::SNNClass ui;

    float* mnistTEST;
    float* mnistTESTIndex;

    std::vector<tensor> vMnistTEST;
    snnModel mySNNModel;
public slots:
    void loadMnst();
    void buildDefaultModel1();
    void lauchModelCalcSimd();
    void checkSingleNeuro();
    void train();
};
