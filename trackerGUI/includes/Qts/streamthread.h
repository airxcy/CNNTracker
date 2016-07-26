#ifndef STREAMTHREAD
#define STREAMTHREAD

#include "itf/trackers/buffers.h"
#include "itf/trackers/trackers.h"

#include <QThread>
#include <QMutex>
#include <QImage>
#include <QWaitCondition>
#include <opencv2/core/core.hpp>

class TrkScene;
class StreamThread : public QThread
{
    Q_OBJECT
public:
    //StreamThread(){};
    StreamThread(QObject *parent = 0);
    ~StreamThread();
    void streaming();
	void parsefname();
    void setUpPersMap();
    void setUpROI();
public:
    std::string vidfname;
    QString qdirstr,baseDirname,vidid,gtdir;
    unsigned char * frameptr;
    int framewidth,frameheight,frameidx,frameByteSize;
    double fps;
    float* persMap;
    QMutex mutex;
    QWaitCondition cv0;
    std::vector<float> roivec;
//    NoTracker* tracker;
    TrkScene* trkscene;
    int delay,bufflen;
    FrameBuff* framebuff;

    CrowdTracker* tracker;
public slots:
    void streamStart(std::string &filename);
    bool init();
    void writeVid();
signals:
    void initSig();
    void aframedone();
protected:
    void run();
public:
    bool restart, abort, pause,paused, inited, firsttime=false,persDone=false,outVid,applyseg=false;
};

#endif // STREAMTHREAD

