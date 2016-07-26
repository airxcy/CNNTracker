#-------------------------------------------------
#
# Project created by QtCreator 2014-12-22T23:23:16
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = groupMask
TEMPLATE = app

HEADERS  += \
    includes/Qts/mainwindow.h \
    includes/Qts/streamthread.h \
    includes/Qts/viewqt.h \
    includes/Qts/modelsqt.h

SOURCES += \
    src/Qts/main.cpp \
    src/Qts/mainwindow.cpp \
    src/Qts/streamthread.cpp \
    src/Qts/viewqt.cpp \
    src/Qts/modelsqt.cpp

# install
target.path = build/
INSTALLS += target

INCLUDEPATH += $$PWD/includes

# ITF Path
ITFLIBDIR += /home/cyxia/GitHub/ITF_Test/lib/.build_release/lib
ITFINCLUDEDIR += ../include
unix:LIBS +=  -L$${ITFLIBDIR} -litf
INCLUDEPATH += $${ITFINCLUDEDIR}

# OPENCV Path
unix:INCLUDEPATH += /home/cyxia/opencv2410/src/include
unix:DEPENDPATH += /home/cyxia/opencv2410/src/include
unix:LIBS += -L/home/cyxia/opencv2410/buildGPU/lib  -lopencv_core\
-lopencv_gpu\
-lopencv_highgui\
-lopencv_imgproc\
-lopencv_legacy\

# Path to cuda toolkit install
CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcudart -lcuda -lcublas
