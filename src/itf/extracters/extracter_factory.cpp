//
//  extracter_factory.cpp
//  ITF_wk
//
//  Created by wking on 11/13/14.
//  Copyright (c) 2014 Kun Wang. All rights reserved.
//

#include "itf/extracters/extracter_factory.hpp"
#include "itf/extracters/density_feature/CDensityExtracter.hpp"



namespace itf {

IExtracter* CExtracterFactory::SpawnExtracter(const ExtracterType& type) {
    IExtracter *iextracter = NULL;

    // Prawn a specific extracter
    switch(type) {
        case Density:
            iextracter = new CDensityExtracter();
            break;
        default:
            LOG(ERROR) << "error extracter type";
    }

    return iextracter;
}

}
