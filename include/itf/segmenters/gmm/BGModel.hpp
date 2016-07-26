//
//  BGModel.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef BGMODEL_H
#define BGMODEL_H

#include "itf/common.hpp"
#include "itf/segmenters/gmm/Types.hpp"

namespace itf {

  class BGModel
  {
  public:

    BGModel(int width, int height);
    virtual ~BGModel();

    void InitModel(IplImage* image);
    void UpdateModel(IplImage* image);
  
    virtual void setBGModelParameter(int id, int value) {};

    virtual IplImage* GetSrc();
    virtual IplImage* GetFG();
    virtual IplImage* GetBG();

  protected:
  
    IplImage* m_SrcImage;
    IplImage* m_BGImage;
    IplImage* m_FGImage;

    const int m_width;
    const int m_height;
  
    virtual void Init() = 0;
    virtual void Update() = 0;
  };



} // namespace itf

#endif
