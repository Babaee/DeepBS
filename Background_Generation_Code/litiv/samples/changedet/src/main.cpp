
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

#include "litiv/video.hpp" // includes all background subtraction algos, along with most core utility & opencv headers
#include "FluxTensorMethod.hpp"

std::string IN_FILE_PATH = "/usr/home/rez/ZM/input/";           // Video input directory
std::string OUT_FILE_PATH = "/usr/home/rez/ZM/Output/";         // Background output diretory

std::string categories[12] = {"","badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"};
std::string video_seq[12][7];
int video_num[12];
void CalcBackgroundImage(cv::Mat &InputImage, cv::Mat &MaskImage, cv::Mat &BGImg, cv::Mat &FluxImage);
std::string FetchFileindex(int i);

int backLib_memory = 5;

int IM_height = 240;
int IM_width = 320;

#define FluxSum_LearningRate_Up 0.2
#define FluxSum_LearningRate_Down 0.01
#define BackLib_Memory_High 90
#define BackLib_Memory_Low 3
long FluxSum_320_240_Upper = 20000; //25%
long FluxSum_320_240_Lower = 2000; //2,6%
long FluxSum_UpperLimit = 0;
long FluxSum_LowerLimit = 0;
#define backLib_size 95

int backLib_index[900][900];
long backLib_sum[900][900][3];
int backLib_num[900][900];
int backLib[backLib_size+2][900][900][3];

double Global_FluxSum = 0;
long InputFrame_idx = 0;

int windowStartX = 100;
int windowStartY = 100;
int foreground_padding_size = 5;

void Set_Video() // Set directory structure for CDNet2014
{
    video_num[1]=4;
    video_seq[1][1]="blizzard"; video_seq[1][2]="skating"; video_seq[1][3]="snowFall";
    video_seq[1][4]="wetSnow";

    video_num[2]=4;
    video_seq[2][1]="highway"; video_seq[2][2]="office"; video_seq[2][3]="pedestrians";
    video_seq[2][4]="PETS2006";

    video_num[3]=4;
    video_seq[3][1]="badminton"; video_seq[3][2]="boulevard"; video_seq[3][3]="sidewalk";
    video_seq[3][4]="traffic";

    video_num[4]=6;
    video_seq[4][1]="boats"; video_seq[4][2]="canoe"; video_seq[4][3]="fall";
    video_seq[4][4]="fountain01"; video_seq[4][5]="fountain02"; video_seq[4][6]="overpass";

    video_num[5]=6;
    video_seq[5][1]="abandonedBox"; video_seq[5][2]="parking"; video_seq[5][3]="sofa";
    video_seq[5][4]="streetLight"; video_seq[5][5]="tramstop"; video_seq[5][6]="winterDriveway";

    video_num[6]=4;
    video_seq[6][1]="port_0_17fps"; video_seq[6][2]="tramCrossroad_1fps"; video_seq[6][3]="tunnelExit_0_35fps";
    video_seq[6][4]="turnpike_0_5fps";

    video_num[7]=6;
    video_seq[7][1]="bridgeEntry"; video_seq[7][2]="busyBoulvard"; video_seq[7][3]="fluidHighway";
    video_seq[7][4]="streetCornerAtNight"; video_seq[7][5]="tramStation"; video_seq[7][6]="winterStreet";

    video_num[8]=4;
    video_seq[8][1]="continuousPan"; video_seq[8][2]="intermittentPan"; video_seq[8][3]="twoPositionPTZCam";
    video_seq[8][4]="zoomInZoomOut";

    video_num[9]=6;
    video_seq[9][1]="backdoor"; video_seq[9][2]="bungalows"; video_seq[9][3]="busStation";
    video_seq[9][4]="copyMachine"; video_seq[9][5]="cubicle"; video_seq[9][6]="peopleInShade";

    video_num[10]=5;
    video_seq[10][1]="corridor"; video_seq[10][2]="diningRoom"; video_seq[10][3]="lakeSide";
    video_seq[10][4]="library"; video_seq[10][5]="park";

    video_num[11]=4;
    video_seq[11][1]="turbulence0"; video_seq[11][2]="turbulence1"; video_seq[11][3]="turbulence2";
    video_seq[11][4]="turbulence3";

}

int main(int, char**) {
    Set_Video();
    cv::Mat oForegroundMask;
    cv::Mat oBackgroundImage;
    cv::Mat flux_output_frame;
    cv::Mat oInput;
    std::string fileName_prefix;
    std::string outputfileName_prefix;
    std::string fileName;
    std::string outputfileName;

    int cat=1,vid=1;
    //std::cout<<"Please Input cat and vid and start: ";
    //std::cin>>cat>>vid>>startf;
    try {

        //for (int cat=3;cat<=11;cat++) {
        //    for (int vid = 1; vid <= video_num[cat]; vid++) {
                FluxTensorMethod flux(5, 5, 5, 5, 30);                              // Initializing FluxTensor

                memset(backLib_index, 0, sizeof(int) * 900 * 900);
                memset(backLib_sum, 0, sizeof(long) * 900 * 900 * 3);
                memset(backLib_num, 0, sizeof(int) * 900 * 900);
                memset(backLib, 0, sizeof(int) * backLib_memory * 900 * 900 * 3);



                int frameNumber = 1;
                InputFrame_idx = 1;


                fileName_prefix = IN_FILE_PATH + categories[cat] + "/" + video_seq[cat][vid] + "/input/in";
                outputfileName_prefix = OUT_FILE_PATH + categories[cat] + "/" + video_seq[cat][vid] + "/in";

                fileName = fileName_prefix + FetchFileindex(frameNumber) + ".jpg";
                outputfileName = outputfileName_prefix + FetchFileindex(frameNumber) + ".jpg";
                oInput = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
                cv::Size s = oInput.size();
                IM_height = s.height;
                IM_width = s.width;
                if (oInput.empty())
                    CV_Error(-1, "Could not fetch video frame from video capture object");

                cv::Mat oBackgroundImage1;

                std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorSuBSENSE>(); // instantiate a background subtractor algo with default parameters

                const double dDefaultLearningRate = pAlgo->getDefaultLearningRate();
                cv::Mat oROI;
                //cv::resize(oInput,oInput,cv::Size(IM_width,IM_height));
                oInput.copyTo(oBackgroundImage);
                pAlgo->initialize(oInput, oROI);

                size_t nCurrInputIdx = 0;
                lv::StopWatch oStopWatch;

                FluxSum_UpperLimit = FluxSum_320_240_Upper*(IM_height*IM_width)/(320*240);
                FluxSum_LowerLimit = FluxSum_320_240_Lower*(IM_height*IM_width)/(320*240);
                while (true) {
                    const double dCurrLearningRate = nCurrInputIdx <= 50 ? 1 : dDefaultLearningRate;
                    pAlgo->apply(oInput, oForegroundMask, dCurrLearningRate);       //Get Foreground segmentation mask
                    //pAlgo->getBackgroundImage(oBackgroundImage1);
                    flux.update(oInput, flux_output_frame);                         //Get motion information from flux tensor
                    CalcBackgroundImage(oInput, oForegroundMask, oBackgroundImage, flux_output_frame);  // Calculate background image

                    cv::imwrite(outputfileName, oBackgroundImage);                  //Store background image to disk
                    //std::string prints = "BM: " + std::to_string(backLib_memory) + "pxl";
                    //cv::putText(oBackgroundImage, prints, cv::Point(220, 20),
                     //           cv::FONT_HERSHEY_COMPLEX_SMALL,
                     //           0.8, cv::Scalar(0, 255, 0),
                      //          1);

                    //cv::putText(oBackgroundImage1,"Old Model",cv::Point(200,20),
                    //            cv::FONT_HERSHEY_COMPLEX_SMALL,
                    //            0.8,cv::Scalar(0,0,255),
                    //            2);

                    cv::imshow("Video input", oInput);
                    cv::imshow("Segmentation output", oForegroundMask);
                    cv::imshow("New Background Model", oBackgroundImage);
                    //cv::imshow("SuBSENSE Background Model",oBackgroundImage1);
                    //cv::imshow("FluxTensor output",flux_output_frame);

                    //cv::moveWindow("Video input", windowStartX, windowStartY);
                    //cv::moveWindow("Segmentation output", windowStartX + 650, windowStartY);
                    //cv::moveWindow("FluxTensor output",windowStartX+650,windowStartY);
                    //cv::moveWindow("SuBSENSE Background Model",windowStartX,windowStartY+280);
                    //cv::moveWindow("New Background Model", windowStartX + 325, windowStartY);

                    if (cv::waitKey(1) == (int) 27)
                        break;

                    frameNumber++;
                    InputFrame_idx++;
                    fileName = fileName_prefix + FetchFileindex(frameNumber) + ".jpg";
                    outputfileName = outputfileName_prefix + FetchFileindex(frameNumber) + ".jpg";

                    oInput = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
                    //cv::resize(oInput,oInput,cv::Size(IM_width,IM_height));

                    if (oInput.empty())
                        break;
                    if ((++nCurrInputIdx % 30) == 0) // every 30 frames, display the total average processing speed
                        std::cout << " avgFPS = " << nCurrInputIdx / oStopWatch.elapsed()<<endl; //<< "  video: " <<categories[cat]<<"-"<< video_seq[cat][vid]<<" - Frame: "<<frameNumber<< std::endl;
                }
                //flux.~FluxTensorMethod();
            //}
        //}
    }

    catch(const cv::Exception& e) {std::cout << "\nmain caught cv::Exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\nmain caught std::exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(...) {std::cout << "\nmain caught unhandled exception\n" << std::endl; return -1;}
    return 0;
}

std::string FetchFileindex(int frameNumber)     //fetch file name e.g. input:12    output:"000012"
{
    std::string fileindex = "";
    int x = frameNumber;
    int div = 100000;
    for (int i = 1; i <= 6; i++) {
        fileindex = fileindex + (char)(48 + (int)(x / div));
        x = x % div;
        div = div / 10;
    }
    return fileindex;
}

// check the neighborhood of checkMask[i][j]
bool checkMask(cv::Mat &MaskImage,int x,int y,int m)        // Apply forground mask padding, if the pixel is classified as
{                                                           // background, but a pixel near it with the distance m is classified
   for(int i=x-m;i<=x+m;i++){                               // as foreground, we disregard the background pixel.
       if((i>=0)&&(i<IM_height)){
           uchar* pMI = MaskImage.ptr<uchar>(i);
           for(int j=y-m;j<=y+m;j++){
               if((j>=0)&&(j<IM_width)){
                  if(pMI[j]) return false;
               }
           }
       }
   }
   return true;
}

void CalcBackgroundImage(cv::Mat &InputImage, cv::Mat &MaskImage, cv::Mat &BGImg, cv::Mat &FluxImage)
{
    uchar B,G,R;
    int channels = InputImage.channels();
    int index = 0;
    int count = 0;
    int divv = 0;
    double FluxPixelSum = 0;
    int upper_padding_size;
    for(int i=0;i<IM_height;i++) {
        uchar* pBG = BGImg.ptr<uchar>(i);
        uchar* pIN = InputImage.ptr<uchar>(i);
        uchar* pMI = MaskImage.ptr<uchar>(i);
        uchar* pFI = FluxImage.ptr<uchar>(i);

        for (int j = 0; j < IM_width; j++) {
            if(pFI[j]) FluxPixelSum++;

            if(channels==3) {
                B = *pIN++;
                G = *pIN++;
                R = *pIN++;

                if ((pMI[j] == 0)&&checkMask(MaskImage,i,j,foreground_padding_size)) {  // if pixel is background and outside the foreground padding area
                    index = backLib_index[i][j];

                    backLib[index][i][j][0] = B;
                    backLib[index][i][j][1] = G;
                    backLib[index][i][j][2] = R;

                    backLib_index[i][j]++;
                    if (backLib_index[i][j] >= backLib_size) backLib_index[i][j] = 0;
                    backLib_num[i][j]++;
                    if (backLib_num[i][j] > backLib_size) backLib_num[i][j] = backLib_memory;
                }

                backLib_sum[i][j][0] = 0;
                backLib_sum[i][j][1] = 0;
                backLib_sum[i][j][2] = 0;

                divv = backLib_memory < backLib_num[i][j] ? backLib_memory : backLib_num[i][j];
                count = divv;

                while (count) {
                    backLib_sum[i][j][0] += backLib[index][i][j][0];        //based on the value of backLib_memory sum up all the pixel value in the library
                    backLib_sum[i][j][1] += backLib[index][i][j][1];
                    backLib_sum[i][j][2] += backLib[index][i][j][2];
                    count--;
                    index--;
                    if (index < 0) index = backLib_num[i][j]-1;
                }
                if(divv != 0) {
                    *pBG++ = (uchar)(backLib_sum[i][j][0] / divv);
                    *pBG++ = (uchar)(backLib_sum[i][j][1] / divv);
                    *pBG++ = (uchar)(backLib_sum[i][j][2] / divv);
                } else {
                    *pBG++ = B;
                    *pBG++ = G;
                    *pBG++ = R;
                }
            } else if (channels == 1) {             //same procedure for single channel image
                B = *pIN++;
                if ((pMI[j] == 0)&&checkMask(MaskImage,i,j,foreground_padding_size)) {
                    index = backLib_index[i][j];

                    backLib[index][i][j][0] = B;

                    backLib_index[i][j]++;
                    if (backLib_index[i][j] >= backLib_size) backLib_index[i][j] = 0;
                    backLib_num[i][j]++;
                    if (backLib_num[i][j] > backLib_size) backLib_num[i][j] = backLib_memory;
                }
                backLib_sum[i][j][0] = 0;

                divv = backLib_memory < backLib_num[i][j] ? backLib_memory : backLib_num[i][j];
                count = divv;
                while (count) {
                    backLib_sum[i][j][0] += backLib[index][i][j][0];
                    count--;
                    index--;
                    if (index < 0) index = backLib_num[i][j]-1;
                }
                if(divv != 0) {
                    *pBG++ = (uchar)(backLib_sum[i][j][0] / divv);
                } else {
                    *pBG++ = B;
                }
            }
        }
    }

    if(InputFrame_idx>=12) {                // dyanmically update backLib_memory and foreground_padding_size using Flux Tensor
        if(InputFrame_idx==12){
            Global_FluxSum = FluxPixelSum;
        } else {
            if (FluxPixelSum>Global_FluxSum) {          //FluxSum is increasing
                Global_FluxSum = Global_FluxSum*(1-FluxSum_LearningRate_Up) + FluxPixelSum*FluxSum_LearningRate_Up;
            } else {                                    //FluxSum is decreasing
                Global_FluxSum = Global_FluxSum*(1-FluxSum_LearningRate_Down) + FluxPixelSum*FluxSum_LearningRate_Down;
            }

            if(Global_FluxSum>=FluxSum_UpperLimit) backLib_memory = BackLib_Memory_Low;
            else if(Global_FluxSum<=FluxSum_LowerLimit) backLib_memory = BackLib_Memory_High;
            else backLib_memory = (int)(BackLib_Memory_High-((Global_FluxSum - FluxSum_LowerLimit)/(FluxSum_UpperLimit - FluxSum_LowerLimit))*(BackLib_Memory_High - BackLib_Memory_Low));
            upper_padding_size = (int)(IM_width/200);
            foreground_padding_size = (int)(upper_padding_size*((float)backLib_memory-BackLib_Memory_Low)/(BackLib_Memory_High-BackLib_Memory_Low)+1);
        }
    }
    //std::cout<<foreground_padding_size<<" "<<backLib_memory<<" "<<(float)Global_FluxSum/IM_height/IM_width*100<<" %    "<<FluxSum_UpperLimit<<"  "<<FluxSum_LowerLimit<<std::endl;
    //std::cout<<"Moving Part:   "<<(float)Global_FluxSum/IM_height/IM_width*100<<" %    "<<backLib_memory<<std::endl;
    //std::cout<<"FS="<<(long)Global_FluxSum<<"   BM="<<backLib_memory<<"  ID="<<InputFrame_idx<<"  PD="<<foreground_padding_size<<std::endl;
}
