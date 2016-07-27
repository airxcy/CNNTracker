#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "Qts/streamthread.h"

#include <iostream>
#include <stdio.h>

#include <QPainter>
#include <QBrush>
#include <QPixmap>
#include <cmath>
#include <QGraphicsSceneEvent>
#include <QMimeData>
#include <QByteArray>
#include <QFont>
char viewstrbuff[200];
QPointF points[100];

void DefaultScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit clicked(event);
}
void DefaultScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    QPen pen;
    QFont txtfont("Roman",40);
    txtfont.setBold(true);
    pen.setColor(QColor(255,255,255));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(10);
    painter->setPen(QColor(243,134,48,150));
    painter->setFont(txtfont);
    painter->drawText(rect, Qt::AlignCenter,"打开文件\nOpen File");
}
TrkScene::TrkScene(const QRectF & sceneRect, QObject * parent):QGraphicsScene(sceneRect, parent)
{
    streamThd=NULL;
}
TrkScene::TrkScene(qreal x, qreal y, qreal width, qreal height, QObject * parent):QGraphicsScene( x, y, width, height, parent)
{
    streamThd=NULL;
}
void TrkScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    //debuggingFile<<streamThd->inited<<std::endl;
     if(streamThd!=NULL&&streamThd->inited)
     {
         updateFptr(streamThd->frameptr, streamThd->frameidx);
     }
     painter->setBrush(bgBrush);
     painter->drawRect(rect);
 //    painter->setBrush(QColor(0,0,0,100));
 //    painter->drawRect(rect);
     painter->setBrush(Qt::NoBrush);
     if(streamThd!=NULL&&streamThd->inited)
     {
         int* neighbor = streamThd->tracker->getNbCount();
         int nFeatures= streamThd->tracker->getNFeatures();
         int nSearch=streamThd->tracker->getNFeatures();
         float* distmat=streamThd->tracker->getDistCo();
         float* cosine=streamThd->tracker->getCosCo();
         float* velo=streamThd->tracker->getVeloCo();
         int* labelVec=streamThd->tracker->getLabel();
         unsigned char* clrvec=streamThd->tracker->getClrvec();
         int2* corners = streamThd->tracker->getCorners();
        Groups& groups = streamThd->tracker->getGroups();
         linepen.setColor(QColor(255,200,200));
         linepen.setWidth(3);
         painter->setPen(linepen);
         painter->setFont(QFont("System",20,2));
         QString infoString="fps:"+QString::number(streamThd->fps)+"\n"
                 +"Prop Idx:"+QString::number(showModeIdx)+"\n"
                 +"numGroups:"+QString::number(groups.numGroups)+"\n";
         painter->drawText(rect, Qt::AlignLeft|Qt::AlignTop,infoString);
         painter->setFont(QFont("System",20,2));
         float* persMap = streamThd->tracker->persMap->cpu_ptr();
         Tracks* tracks = streamThd->tracker->getTracks();
         linepen.setWidth(1);

        for(int i =0;i<tracks->nQue;i++)
        {
            FeatPts p1 = *tracks->getPtr(i);

            int l1 = labelVec[i];
            float r=255,g=255,b=255;
            if(l1)
                r=clrvec[l1*3],g=clrvec[l1*3+1],b=clrvec[l1*3+2];
            //if(l1)
            {
                linepen.setColor(QColor(r,g,b));
                painter->setPen(linepen);
                painter->drawPoint(p1.x,p1.y);
            }
            /*
            for(int j=i+1;j<tracks->nQue;j++)
            {

                FeatPts p2=*tracks->getPtr(j);
                float nbVal=neighbor[i*nFeatures+j];
                int l2=labelVec[j];
                r+=clrvec[l2*3],g+=clrvec[l2*3+1],b+=clrvec[l2*3+2];
                r/=2,g/=2,b/=2;
                linepen.setColor(QColor(r,g,b,nbVal*0.5));
                painter->setPen(linepen);
                painter->drawLine(p1.x,p1.y,p2.x,p2.y);

            }
            */
        }

         GroupTracks& groupsTrk = streamThd->tracker->getGroupsTrk();
         float x0,y0,x1,y1;
         linepen.setWidth(2);

         CrowdTracker& tracker = *(streamThd->tracker);

         int* groupSize = groups.ptsNum->cpu_ptr();
         int* groupVec= groups.trkPtsIdx->cpu_ptr();

         float2* groupVelo=groups.velo->cpu_ptr();
         BBox* groupbBox=groups.bBox->cpu_ptr();
         float2* groupCom=groups.com->cpu_ptr();
         int* gHeadCount=tracker.gHeadCount->cpu_ptr();
         int* gHeadidx=tracker.gHeadIdxMat->cpu_ptr();
        MemBuff<float>& nbSumVec=*tracker.groups->ptsCorr;
        painter->setFont(QFont("Terminal",10,1));
        linepen.setStyle(Qt::SolidLine);
        /*
         for(int i=1;i<groups.kltGroupNum;i++)
         {
            uchar gType=(*tracker.gType)[i];
            if(gType!=DETECT_INVALID)
            {
             BBox& bb = groupbBox[i];
             float bbh = bb.bottom - bb.top, bbw = bb.right - bb.left;
                 linepen.setWidth(1);
                 linepen.setColor(QColor(0,0,255));
//                 if(tracker.mergeTable[i]>1)
//                    linepen.setColor(QColor(255,0,0));

                 painter->setPen(linepen);
                 painter->drawRect(bb.left, bb.top, bbw, bbh);
                linepen.setWidth(1);
                linepen.setColor(QColor(0,255,255));
                painter->setPen(linepen);
                //if(gType==DETECT_1V1)
                //if(i<groups.kltGroupNum)
                {
                painter->drawText(bb.left,bb.top,QString(DetectTypeString[gType])+","+QString::number((*tracker.rankCountNew)[i])
                                  +","+QString::number((*tracker.shapevec)[i]));
                   // painter->drawText(bb.left,bb.top,QString::number((*groups.ptsCorr)[i]));
                }
            }
         }
        */
        std::vector< std::vector<int> > & gTrkCorr=tracker.gTrkCorr;
        uchar* updateType =tracker.updateType->cpu_ptr();
        BBoxF* udpateBoxVec=tracker.kltUpdateBoxVec->cpu_ptr();
        painter->setFont(QFont("Roman",15,2));
         for(int i=0;i<groupsTrk.numGroup;i++)
         {
             std::vector<int>& corrvec = gTrkCorr[i];
             float2 v=(*tracker.kltUpdateVec)[i];
             float dist=v.x*v.x+v.y*v.y;
             GroupTrack* trkptr=groupsTrk.getPtr(i);
             if((*groupsTrk.vacancy)[i]
                     //&&trkptr->updateCount<0.05&&trkptr->len>10
                 &&trkptr->updateCount/trkptr->len<0.05
                     &&!(trkptr->updateCount>0.1&&(*tracker.trkptscount)[i]<3)
//                     &&trkptr->updateCount<0.5&&dist>0.1&&
                     &&((trkptr->trkType==KLT_TRK&&trkptr->len>11&&gTrkCorr[i].size()<1)
                                          ||(trkptr->trkType==HEAD_TRK&&tracker.isLinked[i]))
                     )
             {
                 linepen.setStyle(Qt::DashLine);
                 int* headstats =trkptr->getCur_(trkptr->headStats->cpu_ptr(),CC_STAT_TOTAL);
                 int hasHead = headstats[CC_STAT_AREA]>0;
                 int alhpa = std::min(trkptr->len*10,255);
                 linepen.setWidth(2);
                 //linepen.setColor(QColor(clrvec[i*3],clrvec[i*3+1],clrvec[i*3+2]));
                 linepen.setColor(QColor(trkptr->trkType*255,255,(trkptr->updateCount>0)*255));
                 painter->setPen(linepen);
                 BBox bb =*groupsTrk.getCurBBox(i);
                 float bbh = bb.bottom - bb.top, bbw = bb.right - bb.left;
                 painter->drawRect(bb.left, bb.top, bbw , bbh);
                 linepen.setWidth(1);
                 linepen.setStyle(Qt::SolidLine);
                 painter->setPen(linepen);
                 painter->drawText(bb.left,bb.bottom,QString::number(i));
                //painter->drawText(bb.left,bb.bottom,QString::number(trkptr->updateCount)+"*"+QString::number(trkptr->len));
                 for(int j=1;j<std::min(trkptr->len,TRK_BUFF_LEN)-1;j++)
                 {
                    BBox b1 = *trkptr->getPtr_(trkptr->bBox->cpu_ptr(),j-1);
                    BBox b2 = *trkptr->getPtr_(trkptr->bBox->cpu_ptr(),j);
//                    float2 com1 = {(b1.left+b1.right)/2.0,(b1.top+b1.bottom)/2.0};
//                    float2 com2 = {(b2.left+b2.right)/2.0,(b2.top+b2.bottom)/2.0};
                    float2 com1 = *trkptr->getPtr_(trkptr->com->cpu_ptr(),j-1);
                    float2 com2 = *trkptr->getPtr_(trkptr->com->cpu_ptr(),j);
                    painter->drawLine(com1.x,com1.y,com2.x,com2.y);
                 }
                 linepen.setColor(Qt::red);
                 painter->setPen(linepen);
                 float2 com = *trkptr->getCurCom();
                 painter->drawLine(com.x,com.y,com.x+(*tracker.kltUpdateVec)[i].x*10,com.y+(*tracker.kltUpdateVec)[i].y*10);
//                 linepen.setWidth(2);
//                 linepen.setColor(QColor(255,255,255));
//                 painter->setFont(QFont("System",10,2));
//                 painter->setPen(linepen);
                 //painter->drawText(bb.left, bb.bottom,QString::number(i)+","+QString::number(trkptr->updateCount));
//                 BBoxF upbox =udpateBoxVec[i];
//                 painter->drawLine(upbox.left,upbox.bottom,bb.left,bb.bottom);
//                 linepen.setWidth(1);
//                 linepen.setColor(Qt::black);

//                 painter->setPen(linepen);
//                 painter->drawRect(upbox.left, upbox.top, upbox.right - upbox.left , upbox.bottom - upbox.top);
                 if(hasHead)
                 {
                    linepen.setWidth(1);
                    painter->setPen(linepen);
                    painter->drawRect(headstats[CC_STAT_LEFT],headstats[CC_STAT_TOP],headstats[CC_STAT_WIDTH],headstats[CC_STAT_HEIGHT]);
                    //painter->drawText(headstats[CC_STAT_LEFT],headstats[CC_STAT_TOP],QString::number(headstats[CC_STAT_WIDTH]));
                 }
//                 if(i>100)
//                 {
//                     linepen.setWidth(2);
//                     linepen.setColor(Qt::red);
//                     painter->setPen(linepen);
//                     painter->drawLine(width()/2,height()/2,bb.left,bb.top);
//                     painter->drawText((bb.left+width()/2)/2,(bb.top+height()/2)/2,QString::number(i));
//                 }

//                 if(trkptr->trkType==KLT_TRK)
//                     for(int j=0;j<corrvec.size();j++)
//                     {
//                        BBox jBox = *groupsTrk.getCurBBox(corrvec[j]);
//                        linepen.setWidth(2);
//                        linepen.setColor(Qt::red);
//                        painter->setPen(linepen);

//                        painter->drawRect(jBox.left,jBox.top,jBox.right-jBox.left,jBox.bottom-jBox.top);
//                     }
//                for(int j=0;j<groupsTrk.numGroup;j++)
//                {
//                    float val = tracker.gTrkNb[i*tracker.maxGroupTrk+j];
//                    if(val>1)
//                    {
//                        int color=val*255;
//                        UperLowerBound(color,0,255);
//                        linepen.setWidth(1);
//                        linepen.setColor(QColor(255,255,255,color));
//                        painter->setPen(linepen);
//                        BBox jBox=*groupsTrk.getCurBBox(j);
//                        painter->drawLine(bb.left,bb.top,jBox.left,jBox.top);
//                    }
//                }

             }
         }
//        painter->setFont(QFont("Roman",30,2));
//        for(int i=0;i<tracker.gTrkGroupN;i++)
//        {
//          BBox box = tracker.mergeBox[i];
//          linepen.setColor(Qt::red);
//          painter->setPen(linepen);
//          painter->drawRect(box.left,box.top,box.right-box.left,box.bottom-box.top);
//          QString dispstr="";
//          for(int j=0;j<tracker.gTrkbbNum[i];j++)
//          {
//                dispstr=dispstr+"|"+QString::number(tracker.mergeIdx[i][j]);
//          }
//          painter->drawText(box.left,box.top,dispstr);
//        }

    //painter->setBrush(QImage(streamThd->tracker->gaussKernel->cpu_ptr(), KernelW, KernelH, QImage::Format_RGB888));
    //painter->drawRect(QRect(0,0,KernelW,KernelH));

     }

     //update();
     //views().at(0)->update();
}
void TrkScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	/*
    if(event->button()==Qt::RightButton)
    {
        int x = event->scenePos().x(),y=event->scenePos().y();
        DragBBox* newbb = new DragBBox(x-10,y-10,x+10,y+10);
        int pid = dragbbvec.size();
        newbb->bbid=pid;
        newbb->setClr(255,255,255);
        sprintf(newbb->txt,"%c\0",pid+'A');
        dragbbvec.push_back(newbb);
        addItem(newbb);
    }
    QGraphicsScene::mousePressEvent(event);
	*/
}
void TrkScene::updateFptr(unsigned char * fptr,int fidx)
{
    bgBrush.setTextureImage(QImage(fptr,streamThd->framewidth,streamThd->frameheight,QImage::Format_RGB888));
    frameidx=fidx;
    //debuggingFile<<frameidx<<std::endl;
}
void TrkScene::clear()
{
    bgBrush.setStyle(Qt::NoBrush);
}
