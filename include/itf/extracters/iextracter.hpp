//
//  iextracter.hpp
//  ITF_Inegrated
//
//  Created by Xin Zhu on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef IEXTRACTER_HPP_
#define IEXTRACTER_HPP_

#include "itf/common.hpp"
#include "itf/proto/itf.pb.h"

#include <fstream>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/core/data/load.hpp>



/*
* Thus, if a subclass of the class IExtracter needs to be instantiated, it has to implement each of the virtual
* functions, which means that it supports the interface declared by the class IExtracter. Failure to override a
* pure virtual function in a derived class, then attempting to instantiate objects of that class,
* is a compilation error.
*
*/

namespace itf {
/**
 *
 * @brief The class IExtracter is an abstract class attempting to describe an interface of class.
 * Hence, it can not be instantiated directly and attempting to instantiate an object of an abstract class
 * will cause a compilation error.
 */
    // The abstract interface class for impletementing function of extracter
    // Nopte: no member variables or constructor, but it can
    class IExtracter {
    public:
        virtual ~IExtracter(){};

        /**
         * @brief Returns a vector that contains a list of features.
         *
         * @param src The frame to be extract features
         *
         * This method is a concrete implement of interface of IExtracter.
         *
         */
        virtual std::vector<float> ExtractFeatures(const cv::Mat &src) = 0;

        /**
         *
         * @brief Load Region of Interest.
         *
         * @param file file name specifying the location of ROI file.
         *
         */
        virtual void LoadROI(const std::string &file) = 0;

        /**
         *
         * @brief Load perspective map.
         *
         * @param file file name specifying the location of perspective file.
         *
         */
        virtual void LoadPerspectiveMap(const std::string &file) = 0;

        /**
         *
         * @brief Load parameters related to itf model.
         *
         * @param ep file name specifying the location of parameter file.
         *
         */
        virtual void SetExtracterParameters(const itf::ExtracterParameter &ep) = 0;

        /**
         *
         * @brief Set the dimensionality of frame or image.
         *
         * @param rows 
         *    number of rows.
         * @param cols 
         *    number of columns.
         */
        virtual void SetImagesDim(int rows, int cols) = 0;        
    };


} // namespace itf

#endif  // IEXTRACTER_HPP_
