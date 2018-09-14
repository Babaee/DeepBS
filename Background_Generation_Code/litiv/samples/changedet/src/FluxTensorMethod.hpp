#ifndef _SRC_FLUX_TENSOR_METHOD_HPP_
#define _SRC_FLUX_TENSOR_METHOD_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#define TENSOR_COMPONENTS_NUM	3

//#define THRESHOLD	(30)

using namespace cv;
using namespace std;

#define FIRST_DERIVATIVE_2 -0.5, 0, 0.5
#define SECOND_DERIVATIVE_2 1, -2, 1
const double FIRST_DERIVATIVE_MASK_2[] = {FIRST_DERIVATIVE_2};
const double SECOND_DERIVATIVE_MASK_2[] = {SECOND_DERIVATIVE_2};


#define FIRST_DERIVATIVE_4 		 0.0833, -0.6667, 0, 0.6667, -0.0833
#define SECOND_DERIVATIVE_4 	-0.0833, 1.3333, -2.5, 1.3333, -0.0833
const double FIRST_DERIVATIVE_MASK_4[] = {FIRST_DERIVATIVE_4};
const double SECOND_DERIVATIVE_MASK_4[] = {SECOND_DERIVATIVE_4};

#define FIRST_DERIVATIVE_6		-0.0166, 0.15, -0.75, 0, 0.75, -0.15, 0.0166
#define SECOND_DERIVATIVE_6     0.0111, -0.15, 1.5, -2.7222, 1.5, -0.15, 0.0111
const double FIRST_DERIVATIVE_MASK_6[] = {FIRST_DERIVATIVE_6};
const double SECOND_DERIVATIVE_MASK_6[] = {SECOND_DERIVATIVE_6};

#define GAUSSIAN_3 	0.3271,    0.3458,    0.3271
#define GAUSSIAN_5  0.1782,    0.2105,    0.2225,    0.2105,    0.1782
#define GAUSSIAN_7  0.1063,    0.1403,    0.1658,    0.1752,    0.1658,    0.1403,    0.1063

const uchar BLACK = (uchar) 0;
const uchar WHITE = (uchar) 255;

class FluxTensorMethod
{
    private:
		double threshold;
		// size of the derivative filter
		int nDs;
		// size of the temporal derivative filter
		int nDt;
		// size of the averaging filter
		int nAs;
		// size of the temporal averaging filter
		int nAt;

		void init_filters();
		Mat dx_filter;
		Mat dy_filter;
		Mat sx_filter;
		Mat sy_filter;
		Mat ax_filter;
		Mat ay_filter;

		double * dt_filter;
		double * dtt_filter;
		double * at_filter;

		int compute_Iyt(const Mat & input, Mat & result);
		int compute_Itt(const Mat & input, Mat & result);
		void compute_trace(const Mat * input, Mat & result);


		void apply_temporal_filter(deque<Mat> * queue_ptr, const double * filter, int filter_size, Mat & result_frame);

    public:
        FluxTensorMethod(int nDs = 3, int nDt = 3, int nAs = 3, int nAt = 3, double thershold = 30);
        ~FluxTensorMethod();

        int compute_Ixt(const Mat & input, Mat & result);
        int apply_averaging_filters(const Mat & input, Mat & result);

        void update(const Mat & input_frame, Mat & result_frame);

        void convert2grayscale(const Mat & input, Mat & output);
};

#endif 		/* _SRC_FLUX_TENSOR_METHOD_HPP_ */
