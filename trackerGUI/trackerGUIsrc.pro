#-------------------------------------------------
#
# Project created by QtCreator 2014-12-22T23:23:16
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = groupMask
TEMPLATE = app

ITFDIR = /home/cyxia/GitHub/CNNTracker
ITFSRCDIR = $${ITFDIR}/src
ITFLIBDIR = $${ITFDIR}/.build_release/lib
ITFINCLUDEDIR = $${ITFDIR}/include

CONFIG += c++11

HEADERS  += \
    includes/Qts/mainwindow.h \
    includes/Qts/streamthread.h \
    includes/Qts/viewqt.h \
    includes/Qts/modelsqt.h \
    $${ITFINCLUDEDIR}/itf/trackers/utils.h \
    $${ITFINCLUDEDIR}/itf/trackers/buffers.h \
    $${ITFINCLUDEDIR}/itf/trackers/trackertype.h \
    $${ITFINCLUDEDIR}/itf/trackers/trackers.h \
    $${ITFINCLUDEDIR}/itf/trackers/gpucommon.hpp \
    $${ITFINCLUDEDIR}/itf/trackers/buffgpu.h
SOURCES += \
    src/Qts/main.cpp \
    src/Qts/mainwindow.cpp \
    src/Qts/streamthread.cpp \
    src/Qts/viewqt.cpp \
    src/Qts/modelsqt.cpp\
    $${ITFSRCDIR}/itf/trackers/utils.cpp \
    $${ITFSRCDIR}/itf/trackers/buffers.cpp \
    $${ITFSRCDIR}/itf/trackers/targetfinder.cpp \
    $${ITFSRCDIR}/itf/trackers/conn_precomp.hpp \
    $${ITFSRCDIR}/itf/trackers/conncomp.cpp \
    $${ITFSRCDIR}/itf/trackers/pointTracker.cpp

CUDA_SOURCES += $${ITFSRCDIR}/itf/trackers/pointTracker.cu
CUDA_SOURCES += $${ITFSRCDIR}/itf/trackers/targetfinder.cu
CUDA_SOURCES += $${ITFSRCDIR}/itf/trackers/buffgpu.cu
CUDA_SOURCES += $${ITFSRCDIR}/itf/trackers/tracks.cu

# install
target.path = build/
INSTALLS += target

INCLUDEPATH += $$PWD/includes
INCLUDEPATH += ITFINCLUDEDIR

# ITF Path
LIBS +=  -L$${ITFLIBDIR} -litf
INCLUDEPATH += $${ITFINCLUDEDIR}

# OPENCV Path
unix:INCLUDEPATH += /home/cyxia/opencv2410/src/include
unix:DEPENDPATH += /home/cyxia/opencv2410/src/include
unix:LIBS += -L/home/cyxia/opencv2410/buildGPU2/lib  -lopencv_core\
-lopencv_gpu\
-lopencv_highgui\
-lopencv_imgproc\
-lopencv_legacy\

### CUDA Set Up #####
# Path to cuda toolkit install
CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
# libs used in your code
LIBS += -lcudart -lcuda -lcublas
# GPU architecture
CUDA_ARCH = sm_30
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v


# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
