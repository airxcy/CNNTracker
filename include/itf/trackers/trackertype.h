#ifndef TRACKERTYPE_H
#define TRACKERTYPE_H

typedef struct //corelation pair i0 i1 are idx
{
    int i0, i1;
    float correlation;
}ppair, p_ppair;

typedef unsigned char BYTE;
typedef float REAL;
typedef int PntT;

typedef struct//Int Track Points
{
    PntT x;
    PntT y;
    int t;
} TrkPts, *TrkPts_p;

typedef struct//Float Track Points
{
    REAL x;
    REAL y;
    int t;
}FeatPts,*FeatPts_p;

typedef struct//optical flow vector (x0,y0)->(x1,y1) len is frame Time Span, idx is the id
{
    REAL x0, y0, x1, y1;
    int  len, idx;
} ofv, *ofv_p;

typedef struct//Bounding Box
{
    int left;
    int top;
    int right;
    int bottom;
}BBox, *BBox_p;
/*
BBox operator +(const BBox &a, const float2 &b) {
	BBox tmpBox;
	tmpBox.left = a.left + b.x;
	tmp
}
*/
typedef struct//Bounding Box
{
    float left;
    float top;
    float right;
    float bottom;
}BBoxF, *BBoxF_p;

struct cvxPnt {
    float x, y;

    bool operator <(const cvxPnt &p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
};

enum ConnectedComponentsTypes {
    CC_STAT_LEFT   = 0, //!< The leftmost (x) coordinate which is the inclusive start of the bounding
                        //!< box in the horizontal direction.
    CC_STAT_TOP    = 1, //!< The topmost (y) coordinate which is the inclusive start of the bounding
                        //!< box in the vertical direction.
    CC_STAT_WIDTH  = 2, //!< The horizontal size of the bounding box
    CC_STAT_HEIGHT = 3, //!< The vertical size of the bounding box
    CC_STAT_AREA   = 4, //!< The total area (in pixels) of the connected component
    CC_STAT_MAX    = 5,
};
enum ConnectedComponentsTypes2
{
    CC_STAT_HEAD_X = 5,
    CC_STAT_HEAD_Y = 6,
    CC_STAT_TOTAL = 7
};

enum GroupDetectType
{
    DETECT_GOOD = 0,
    DETECT_INVALID,
    DETECT_DUPLICATED,
    DETECT_SPLITED,
    DETECT_USED,
    DETECT_DELAY_MERGE,
    DETECT_1V1,
    DETECT_MERGE,
    DETECT_FIRM,
    NUM_DETECT_TYPE
};
enum GroupTrkType
{
    KLT_TRK=0,
    HEAD_TRK=1,
    NUM_GTRK_TYPE=2
};
extern char DetectTypeString[NUM_DETECT_TYPE][5];
#define UperLowerBound(val,minv,maxv) {int ind=val>(minv);val=ind*val+(!ind)*(minv);ind=val<(maxv);val=ind*val+(!ind)*(maxv);}

#endif // TRACKERTYPE_H

