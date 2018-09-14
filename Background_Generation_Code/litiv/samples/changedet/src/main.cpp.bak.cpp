
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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
//
/////////////////////////////////////////////////////////////////////////////
//
// This sample demonstrates how to extract foreground masks using a change
// detection via background subtraction algo, using either a webcam or a
// user-provided video as input. It will also display the processing speed.
//
/////////////////////////////////////////////////////////////////////////////

#include <litiv/video/BackgroundSubtractorViBe.hpp>
#include "litiv/video.hpp" // includes all background subtraction algos, along with most core utility & opencv headers
#include "FluxTensorMethod.hpp"

#define USE_WEBCAM 0 // defines whether OpenCV's first detected camera should be used, or a given video should be read
#if !USE_WEBCAM
// #define VIDEO_FILE_PATH "/Users/Capricorn/Downloads/bgslibrary-master/dataset/video.avi" // if you want to use your own file, put it here!
#endif //(!USE_WEBCAM)
std::string FILE_PATH = "/Users/Capricorn/Desktop/SuBSENSE_Test/sofa/input"; // if you want to use your own file, put it here!
//std::string FILE_PATH = "/Users/Capricorn/Desktop/SuBSENSE_Test/PETS2006/input"; // if you want to use your own file, put it here!
//std::string FILE_PATH = "/Users/Capricorn/Desktop/SuBSENSE_Test/tramstop/input"; // if you want to use your own file, put it here!
//std::string FILE_PATH = "/Users/Capricorn/Desktop/SuBSENSE_Test/office"; // if you want to use your own file, put it here!
//std::string FILE_PATH = "/Users/Capricorn/Desktop/SuBSENSE_Test/winterDriveway/input"; // if you want to use your own file, put it here!
//std::string FILE_PATH1 = "/Users/Capricorn/Desktop/SuBSENSE_Test/highway";
long flux_metrics[] = {0,0,0,0};
void CalcBackgroundImage(cv::Mat &InputImage, cv::Mat &MaskImage, cv::Mat &BGImg);

int backLib_index[240][320];
long backLib_sum[240][320][3];
int backLib_num[240][320];

int main(int, char**) { // this sample uses no command line argument
    FluxTensorMethod flux (5,5,5,5,30);
    try {
        cv::Mat oInput; // this matrix will be used to fetch input frames (no need to preallocate)
//        cv::Mat oInput1;
        //oCap >> oInput; // minimalistic way to fetch a frame from a video capture object
        cv::Mat flux_output_frame;
        int frameNumber = 1;
        std::string fileindex = "";

        int x = frameNumber;
        int div = 100000;
        for (int i = 1; i <= 6; i++) {
            fileindex = fileindex + (char)(48 + (int)(x / div));
            x = x % div;
            div = div / 10;
        }

        std::string fileName = FILE_PATH + "/in" + fileindex + ".jpg";
//        std::string fileName1 = FILE_PATH1 + "/in" + fileindex + ".jpg";
        // std::cout << "processing" << fileName << std::endl;

        oInput = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
//        oInput1 = cv::imread(fileName1, CV_LOAD_IMAGE_COLOR);

        if(oInput.empty()) // check if the fetched frame is empty (i.e. if the video failed to load/seek)
            CV_Error(-1,"Could not fetch video frame from video capture object");
        cv::Mat oForegroundMask; // no need to preallocate the output matrix (the algo will make sure it is allocated at some point)
//        cv::Mat oForegroundMask1;
        cv::Mat oBackgroundImage;
        oInput.copyTo(oBackgroundImage);
//        cv::Mat oBackgroundImage1;
        //std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorViBe>(); // instantiate a background subtractor algo with default parameters
//        std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorSuBSENSE>(); // instantiate a background subtractor algo with default parameters
//        std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorSuBSENSE>(); // instantiate a background subtractor algo with default parameters
//        BackgroundSubtractorSuBSENSE* pAlgo = new BackgroundSubtractorSuBSENSE();
         std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorPAWCS>(); // instantiate a background subtractor algo with default parameters
        //BackgroundSubtractorSuBSENSE pAlgo;
        const double dDefaultLearningRate = pAlgo->getDefaultLearningRate(); // gets the suggested learning rate to use post-initialization (algo-dependent, some will totally ignore it)
//        const double dDefaultLearningRate1 = pAlgo1->getDefaultLearningRate(); // gets the suggested learning rate to use post-initialization (algo-dependent, some will totally ignore it)
        cv::Mat oROI; // specifies the segmentation region of interest (when the matrix is left unallocated, the full frame is used by default)
        cv::resize(oInput,oInput,cv::Size(320,240));
//        cv::resize(oInput1,oInput1,cv::Size(320,240));
        pAlgo->initialize(oInput,oROI); // initialize the background model using the video's first frame (it may already contain foreground; the algo should adapt to any case)
//        pAlgo1->initialize(oInput1,oROI);
        size_t nCurrInputIdx = 0; // use a frame counter to report average processing speed
        lv::StopWatch oStopWatch;
        while(true) { // loop, as long as we can still fetch frames
            const double dCurrLearningRate = nCurrInputIdx<=50?1:dDefaultLearningRate; // boost the learning rate in the first ~50 frames for initialization, and switch to default after
//            const double dCurrLearningRate1 = nCurrInputIdx<=50?1:dDefaultLearningRate1; // boost the learning rate in the first ~50 frames for initialization, and switch to default after
//            flux.update(oInput,flux_output_frame);
//            flux_output_frame.copyTo(oForegroundMask);
            pAlgo->apply(oInput,oForegroundMask,dCurrLearningRate); // apply background subtraction on a new frame (oInput), and fetch the result simultaneously (oForegroundMask)
//            pAlgo->getBackgroundImage(oBackgroundImage);
            CalcBackgroundImage(oInput,oForegroundMask,oBackgroundImage);
//            pAlgo1->apply(oInput1,oForegroundMask1,dCurrLearningRate1);
//            pAlgo1->getBackgroundImage(oBackgroundImage1);
            cv::imshow("Video input",oInput); // display the input video frame
            cv::imshow("Segmentation output",oForegroundMask); // display the output segmentation mask (white = foreground)
            cv::imshow("Background Model",oBackgroundImage);
//            cv::imshow("Flux outputL",flux_output_frame);

//            cv::imshow("Video input1",oInput1); // display the input video frame
//            cv::imshow("Segmentation output1",oForegroundMask1); // display the output segmentation mask (white = foreground)
//            cv::imshow("Background Model1",oBackgroundImage1);
            //cv::imshow("Fluxtensor output",flux_output_frame);
            if(cv::waitKey(1)==(int)27) // immediately refresh the display with a 1ms pause, and continue processing (or quit if the escape key is pressed)
                break;
            
            frameNumber++;
            fileindex = "";

            x = frameNumber;
            div = 100000;
            for (int i = 1; i <= 6; i++) {
                fileindex = fileindex + (char)(48 + (int)(x / div));
                x = x % div;
                div = div / 10;
            }

            fileName = FILE_PATH + "/in" + fileindex + ".jpg";
//            fileName1 = FILE_PATH1 + "/in" + fileindex + ".jpg";
            // std::cout << "processing" << fileName << std::endl;

            oInput = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
//            oInput1 = cv::imread(fileName1, CV_LOAD_IMAGE_COLOR);
            cv::resize(oInput,oInput,cv::Size(320,240));
//            cv::resize(oInput1,oInput1,cv::Size(320,240));

            if(oInput.empty()) // if the frame is empty, we hit the end of the video, or the webcam was shut off
                break;
            if((++nCurrInputIdx%30)==0) // every 30 frames, display the total average processing speed
                std::cout << " avgFPS = " << nCurrInputIdx/oStopWatch.elapsed() << std::endl;
        }
    }
    // delete pAlgo;
    catch(const cv::Exception& e) {std::cout << "\nmain caught cv::Exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\nmain caught std::exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(...) {std::cout << "\nmain caught unhandled exception\n" << std::endl; return -1;}
    return 0;
}

cv::Mat backgroundLib[60];

void CalcBackgroundImage(cv::Mat &InputImage, cv::Mat &MaskImage, cv::Mat &BGImg)
{
    MaskImage = MaskImage;
    for(int i=0;i<240;i++) {
        uchar *pBG = BGImg.ptr<uchar>(i);
        uchar *pIN = InputImage.ptr<uchar>(i);
        uchar *pMI = MaskImage.ptr<uchar>(i);
        for (int j = 0; j < 320; j++) {
            if(*pMI==0){

            }
            if (j != 310) {
                *pBG++ = *pIN++;
                *pBG++ = *pIN++;
                *pBG++ = *pIN++;
            } else {
                *pBG++ = 250;
                *pBG++ = 0;
                *pBG++ = 0;
            }
        }
    }
}
