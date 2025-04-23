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
#include"loadMNIST.h"
#include <QtDataVisualization/Q3DScatter>
#include <QtDataVisualization/QScatter3DSeries>
#include <QtDataVisualization/QValue3DAxis>

QT_CHARTS_USE_NAMESPACE
using namespace QtDataVisualization;
class SNN : public QMainWindow
{
    Q_OBJECT

public:
    SNN(QWidget *parent = nullptr);
    ~SNN();

private:
    Ui::SNNClass ui;
    snnModel mySNNModel;
    MNISTLoader myMNIST;

    inline  float randomFloat(float min, float max) {
        return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    };
    std::vector<QScatterDataItem> originalImage;
    std::vector<QScatterDataItem> SpikeImage;
    std::vector<QScatterDataItem> SpikeOut;
    void plotOutSpike(int number);
    Q3DScatter* scatter;
    QWidget* scatterContainer ;
    QScatter3DSeries* scatterSeries1;
    QScatter3DSeries* scatterSeries2;

    QChartView* myView2 ;
    QChartView* myView3;
    QChartView* myView4;

    QLineSeries* memSeries ;
    QScatterSeries* outSpikeScatters;
    QScatterSeries* hiddenSpikeScatters;

    int TIMESTEP;
    void initPlotBoard();

public slots:
    void loadMnst();
    void buildDefaultModel1();
    void lauchModelCalcSimd();
    void train();
    void plot3D();
    void showHiddenSpikeOut();
    void changeNeuroConfig(int);
};
