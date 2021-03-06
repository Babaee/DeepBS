
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "litiv/utils/parallel.hpp"
#include "litiv/utils/opencv.hpp"
#include <opencv2/video/background_segm.hpp>

/// super-interface for background subtraction algos which exposes common interface functions
struct IIBackgroundSubtractor : public cv::BackgroundSubtractor {

    // @@@ add refresh model as virtual pure func here?

    /// (re)initiaization method; needs to be called before starting background subtraction (assumes no specific ROI)
    void initialize(const cv::Mat& oInitImg);
    /// (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) = 0;
    /// returns the default learning rate value used in 'apply'
    virtual double getDefaultLearningRate() const = 0;
    /// segments the input image into fg/bg based on the so-far-learned background model, simultanously updating the latter based on 'dLearningRate'
    virtual void apply(cv::InputArray oImage, cv::OutputArray oFGMask, double dLearningRate=-1) = 0;
    /// computes the current empty background image based on model data
    virtual void getBackgroundImage(cv::OutputArray oBackgroundImage) const = 0;
    /// turns automatic model reset on or off
    virtual void setAutomaticModelReset(bool);
    /// modifies the given ROI so it will not cause lookup errors near borders when used in the processing step
    virtual void validateROI(cv::Mat& oROI) const;
    /// sets the ROI to be used for input analysis (note: this function will reinit the model and return the validated ROI)
    virtual void setROI(cv::Mat& oROI);
    /// returns a copy of the ROI used for input analysis
    virtual cv::Mat getROICopy() const;
    /// required for derived class destruction from this interface
    virtual ~IIBackgroundSubtractor() = default;

protected:
    /// default impl constructor (for common parameters only -- none must be const to avoid constructor hell when deriving)
    IIBackgroundSubtractor();
    /// common (re)initiaization method for all impl types (should be called in impl-specific initialize func)
    virtual void initialize_common(const cv::Mat& oInitImg, const cv::Mat& oROI);

    /// basic info struct used in px model LUTs
    struct PxInfoBase {
        int nImgCoord_Y;
        int nImgCoord_X;
        size_t nModelIdx;
    };
    /// background model ROI used for input analysis (specific to the input image size)
    cv::Mat m_oROI;
    /// input image size
    cv::Size m_oImgSize;
    /// ROI border size to be ignored, useful for descriptor-based methods
    size_t m_nROIBorderSize;
    /// input image channel size
    size_t m_nImgChannels;
    /// input image type
    int m_nImgType;
    /// total number of pixels (depends on the input frame size) & total number of relevant pixels
    size_t m_nTotPxCount, m_nTotRelevantPxCount;
    /// total number of ROI pixels before & after border cleanup
    size_t m_nOrigROIPxCount, m_nFinalROIPxCount;
    /// current frame index, frame count since last model reset & model reset cooldown counters
    size_t m_nFrameIdx, m_nFramesSinceLastReset, m_nModelResetCooldown;
    /// internal pixel index LUT for all relevant analysis regions (based on the provided ROI)
    std::vector<size_t> m_vnPxIdxLUT;
    /// internal pixel info LUT for all possible pixel indexes
    std::vector<PxInfoBase> m_voPxInfoLUT;
    /// specifies whether the algorithm parameters are fully initialized or not (must be handled by derived class)
    bool m_bInitialized;
    /// specifies whether the model has been fully initialized or not (must be handled by derived class)
    bool m_bModelInitialized;
    /// specifies whether automatic model resets are enabled or not
    bool m_bAutoModelResetEnabled;
    /// specifies whether the camera is considered moving or not
    bool m_bUsingMovingCamera;
    /// the foreground mask generated by the method at [t-1]
    cv::Mat m_oLastFGMask;
    /// copy of latest pixel intensities (used when refreshing model)
    cv::Mat m_oLastColorFrame;

private:
    IIBackgroundSubtractor& operator=(const IIBackgroundSubtractor&) = delete;
    IIBackgroundSubtractor(const IIBackgroundSubtractor&) = delete;
};

template<lv::ParallelAlgoType eImpl>
struct IBackgroundSubtractor_;

#if HAVE_GLSL
template<>
struct IBackgroundSubtractor_<lv::GLSL> :
        public lv::IParallelAlgo_GLSL,
        public IIBackgroundSubtractor {
    /// required for derived class destruction from this interface
    virtual ~IBackgroundSubtractor_() {}
    /// returns a copy of the latest foreground mask
    void getLatestForegroundMask(cv::OutputArray oLastFGMask);
    /// (re)initiaization method (asynchronous version w/ gl interface); needs to be called before starting background subtraction
    virtual void initialize_gl(const cv::Mat& oInitImg, const cv::Mat& oROI) override;
    /// overloads 'initialize' from IIBackgroundSubtractor and redirects it to 'initialize_gl'
    virtual void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) override final;
    /// model update/segmentation function (asynchronous version w/ gl interface); the learning param is used to override the internal learning speed
    void apply_gl(cv::InputArray oNextImage, bool bRebindAll=false, double dLearningRate=-1);
    /// model update/segmentation function (asynchronous version w/ gl interface); the learning param is used to override the internal learning speed
    void apply_gl(cv::InputArray oNextImage, cv::OutputArray oLastFGMask, bool bRebindAll=false, double dLearningRate=-1);
    /// overloads 'apply' from IIBackgroundSubtractor and redirects it to 'apply_gl'
    virtual void apply(cv::InputArray oNextImage, cv::OutputArray oLastFGMask, double dLearningRate=-1) override final;

protected:
    /// glsl impl constructor
    IBackgroundSubtractor_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                           size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                           bool bUseTimers, bool bUseIntegralFormat);
    /// used to pass learning rate parameter to overridden dispatch call, if needed
    double m_dCurrLearningRate;
    /// eliminates 'hides overloaded virtual func' warning on some platforms
    using lv::IParallelAlgo_GLSL::apply_gl;
};

using IBackgroundSubtractor_GLSL = IBackgroundSubtractor_<lv::GLSL>;
#endif //HAVE_GLSL

#if HAVE_CUDA
// IBackgroundSubtractor_<lv::CUDA> will not compile here, missing impl
#endif //HAVE_CUDA

#if HAVE_OPENCL
// IBackgroundSubtractor_<lv::OpenCL> will not compile here, missing impl
#endif //HAVE_OPENCL

template<>
struct IBackgroundSubtractor_<lv::NonParallel> :
        public lv::NonParallelAlgo,
        public IIBackgroundSubtractor {
    /// required for derived class destruction from this interface
    virtual ~IBackgroundSubtractor_() {}
};

using IBackgroundSubtractor = IBackgroundSubtractor_<lv::NonParallel>;
