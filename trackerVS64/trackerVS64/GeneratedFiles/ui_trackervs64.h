/********************************************************************************
** Form generated from reading UI file 'trackervs64.ui'
**
** Created by: Qt User Interface Compiler version 5.4.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TRACKERVS64_H
#define UI_TRACKERVS64_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_trackerVS64Class
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *trackerVS64Class)
    {
        if (trackerVS64Class->objectName().isEmpty())
            trackerVS64Class->setObjectName(QStringLiteral("trackerVS64Class"));
        trackerVS64Class->resize(600, 400);
        menuBar = new QMenuBar(trackerVS64Class);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        trackerVS64Class->setMenuBar(menuBar);
        mainToolBar = new QToolBar(trackerVS64Class);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        trackerVS64Class->addToolBar(mainToolBar);
        centralWidget = new QWidget(trackerVS64Class);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        trackerVS64Class->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(trackerVS64Class);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        trackerVS64Class->setStatusBar(statusBar);

        retranslateUi(trackerVS64Class);

        QMetaObject::connectSlotsByName(trackerVS64Class);
    } // setupUi

    void retranslateUi(QMainWindow *trackerVS64Class)
    {
        trackerVS64Class->setWindowTitle(QApplication::translate("trackerVS64Class", "trackerVS64", 0));
    } // retranslateUi

};

namespace Ui {
    class trackerVS64Class: public Ui_trackerVS64Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRACKERVS64_H
