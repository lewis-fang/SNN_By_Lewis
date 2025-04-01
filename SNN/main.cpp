#include "stdafx.h"
#include "SNN.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    SNN w;
    w.show();
    return a.exec();
}
