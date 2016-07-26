//
//  extracter_factory.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef EXTRACTER_FACTORY_HPP_
#define EXTRACTER_FACTORY_HPP_

#include "itf/extracters/iextracter.hpp"

namespace itf {

/**
 * @brief Extracter factory used to instantiate one of concrete extracters
 *        such as DensityExtracter or ForegroundExtracter.
 *
 */
class CExtracterFactory {
 public:
    /// The enum ExtracterType defines the available feature-extracter types
    enum ExtracterType {
        Density  //!< based on deep learning
    };
    /**
     *
     * @brief Instantiate one of concrete extracters
     *
     * @param type 
     *    Note: only CExtracterFactory::ExtracterType::Density is available for now
     *
     */
    IExtracter* SpawnExtracter(const ExtracterType& type);
};


} // namespace itf

#endif  // EXTRACTER_FACTORY_HPP_
