#include "FluxTensorMethod.hpp"

FluxTensorMethod::FluxTensorMethod(int nDs1, int nDt1, int nAs1, int nAt1, double thershold)
{
	this->threshold = thershold;
	this->nDs = nDs1;
	this->nDt = nDt1;
	this->nAs = nAs1;
	this->nAt = nAt1;
	init_filters();
}

FluxTensorMethod::~FluxTensorMethod()
{
	delete [] dt_filter;
	delete [] dtt_filter;
	delete [] at_filter;
}

void FluxTensorMethod ::init_filters()
{
	switch(nDs){
		case 3:
			dx_filter = (Mat_<double>(1, nDs) << FIRST_DERIVATIVE_2);
			dy_filter = (Mat_<double>(nDs, 1) << FIRST_DERIVATIVE_2);
			sx_filter = (Mat_<double>(1, nDs) << GAUSSIAN_3);
			sy_filter = (Mat_<double>(nDs, 1) << GAUSSIAN_3);
		break;
		case 5:
			dx_filter = (Mat_<double>(1, nDs) << FIRST_DERIVATIVE_4);
			dy_filter = (Mat_<double>(nDs, 1) << FIRST_DERIVATIVE_4);
			sx_filter = (Mat_<double>(1, nDs) << GAUSSIAN_5);
			sy_filter = (Mat_<double>(nDs, 1) << GAUSSIAN_5);
		break;
		case 7:
			dx_filter = (Mat_<double>(1, nDs) << FIRST_DERIVATIVE_6);
			dy_filter = (Mat_<double>(nDs, 1) << FIRST_DERIVATIVE_6);
			sx_filter = (Mat_<double>(1, nDs) << GAUSSIAN_7);
			sy_filter = (Mat_<double>(nDs, 1) << GAUSSIAN_7);
		break;
	}

	//sx_filter = Mat::ones( 1, nDs, CV_32F )/ (float)(nDs);
	//sy_filter = Mat::ones(nDs, 1, CV_32F )/ (float)(nDs);
	ax_filter = Mat::ones(1, nAs, CV_32F )/ (float)(nAs);
	ay_filter = Mat::ones(nAs, 1, CV_32F )/ (float)(nAs);

	dt_filter = new double [nDt];
	dtt_filter = new double [nDt];

	switch(nDt){
		case 3:
			memcpy(dt_filter, FIRST_DERIVATIVE_MASK_2, nDt*sizeof(double));
			memcpy(dtt_filter, SECOND_DERIVATIVE_MASK_2, nDt*sizeof(double));
		break;
		case 5:
			memcpy(dt_filter, FIRST_DERIVATIVE_MASK_4, nDt*sizeof(double));
			memcpy(dtt_filter, SECOND_DERIVATIVE_MASK_4, nDt*sizeof(double));
		break;
		case 7:
			memcpy(dt_filter, FIRST_DERIVATIVE_MASK_6, nDt*sizeof(double));
			memcpy(dtt_filter, SECOND_DERIVATIVE_MASK_6, nDt*sizeof(double));
		break;
	}

	at_filter = new double [nAt];
	for(int i=0; i<nAt; ++i)
		at_filter[i] = 1.0/nAt;
}

void FluxTensorMethod :: apply_temporal_filter(deque<Mat> * queue_ptr, const double * filter, int filter_size, Mat & result_frame)
{
    double * rgb;
    rgb = new double [filter_size];
    //for(int i=0; i<filter_size; ++i)
    	//rgb[i] = new double [3];

    const uchar ** input_pixel_ptr = new const uchar * [filter_size];
    uchar * result_pixel_ptr;

    int height = queue_ptr->front().rows;
    int width = queue_ptr->front().cols;

    for(int row = 0; row < height; ++row)
    {
    	for(int i=0; i<filter_size; i++)
    	{
    		input_pixel_ptr[i] = (*queue_ptr)[i].ptr(row);
    	}

        result_pixel_ptr = result_frame.ptr(row);

        for(int col = 0; col < width; ++col)
        {
        	//double res_pixel[] = {0, 0, 0};
            double res_pixel = 0.0;
        	for(int i=0; i<filter_size; ++i)
            {
            	//RGB reverted order
            	/*rgb[i][2] = (double) *(input_pixel_ptr[i])++;
            	rgb[i][1] = (double) *(input_pixel_ptr[i])++;
            	rgb[i][0] = (double) *(input_pixel_ptr[i])++;
            	res_pixel[0] += rgb[i][0]*filter[i];
            	res_pixel[1] += rgb[i][1]*filter[i];
            	res_pixel[2] += rgb[i][2]*filter[i];*/
            	rgb[i] = (double) *(input_pixel_ptr[i])++;
            	res_pixel += rgb[i]*filter[i];
            }

              for(int i = 0; i < 1; ++i)
              {
            	  res_pixel = (res_pixel < 0) ? 0 : res_pixel;
            	  *result_pixel_ptr++ = uchar(int(res_pixel));
            	  //*result_pixel_ptr++ = 255;
              }
        }
    }

    //for(int i=0; i<filter_size; ++i)
    //	delete [] rgb[i];
    delete [] rgb;
    delete [] input_pixel_ptr;
}

int FluxTensorMethod :: compute_Ixt(const Mat & input, Mat & result)
{
	static deque<Mat> idxsy_fifo;

	result = input.clone();

	Mat dx_result;
	filter2D(input, dx_result, -1, dx_filter);

	Mat dx_sy_result;
	filter2D(dx_result, dx_sy_result, -1, sy_filter);

	idxsy_fifo.push_back(dx_sy_result);

	if(idxsy_fifo.size() < (unsigned int)nDt)
		return 1;

	apply_temporal_filter(&idxsy_fifo, dt_filter, nDt, result);

	idxsy_fifo.pop_front();
	return 0;
}


int FluxTensorMethod::compute_Iyt(const Mat & input, Mat & result)
{
	static deque<Mat> isxdy_fifo;

	result = input.clone();

	Mat sx_result;
	filter2D(input, sx_result, -1, sx_filter);

	Mat sx_dy_result;
	filter2D(sx_result, sx_dy_result, -1, dy_filter);

	isxdy_fifo.push_back(sx_dy_result);
	if(isxdy_fifo.size() < (unsigned int)nDt)
		return 1;

	apply_temporal_filter(&isxdy_fifo, dt_filter, nDt, result);

	isxdy_fifo.pop_front();
	return 0;
}

int FluxTensorMethod::compute_Itt(const Mat & input, Mat & result)
{
	static deque<Mat> isxsy_fifo;

	result = input.clone();

	Mat sx_result;
	filter2D(input, sx_result, -1, sx_filter);

	Mat sx_sy_result;
	filter2D(sx_result, sx_sy_result, -1, sy_filter);

	isxsy_fifo.push_back(sx_sy_result);
	if(isxsy_fifo.size() < (unsigned int)nDt)
		return 1;

	apply_temporal_filter(&isxsy_fifo, dtt_filter, nDt, result);

	isxsy_fifo.pop_front();
	return 0;
}

void FluxTensorMethod::compute_trace(const Mat * input, Mat & result)
{
    double * rgb;
    rgb = new double [TENSOR_COMPONENTS_NUM];
   // for(int i=0; i<3; ++i)
    //	rgb[i] = new double [3];

    const uchar ** input_pixel_ptr = new const uchar * [TENSOR_COMPONENTS_NUM];
    uchar * result_pixel_ptr;

    int height = input[0].rows;
    int width = input[0].cols;

    for(int row = 0; row < height; ++row)
    {
    	for(int i=0; i<TENSOR_COMPONENTS_NUM; i++)
    		input_pixel_ptr[i] = input[i].ptr(row);

        result_pixel_ptr = result.ptr(row);

        for(int col = 0; col < width; ++col)
        {
        	//double res_pixel[] = {0, 0, 0};
        	//uchar res_pixel_u[] = {0, 0, 0};
        	double res_pixel = 0;
        	//uchar res_pixel_u = 0;
            for(int i=0; i<TENSOR_COMPONENTS_NUM; ++i)
            {
            	//RGB reverted order
            	/*rgb[i][2] = (double) *(input_pixel_ptr[i])++;
            	rgb[i][1] = (double) *(input_pixel_ptr[i])++;
            	rgb[i][0] = (double) *(input_pixel_ptr[i])++;
            	res_pixel[0] += rgb[i][0]*rgb[i][0];
            	res_pixel[1] += rgb[i][1]*rgb[i][1];
            	res_pixel[2] += rgb[i][2]*rgb[i][2];

            	res_pixel_u[0] += int(rgb[i][0]*rgb[i][0]);
            	res_pixel_u[1] += int(rgb[i][1]*rgb[i][1]);
            	res_pixel_u[2] += int(rgb[i][2]*rgb[i][2]);*/

            	rgb[i] = (double) *(input_pixel_ptr[i])++;
            	res_pixel += rgb[i]*rgb[i];

            	/*if(row == 100 && col == 100){
            		//cout << rgb[i][0]*rgb[i][0] << "   " << (unsigned int)(uchar(rgb[i][0]*rgb[i][0])) << "    ";
            		cout << res_pixel[0] << "    " << (unsigned int)(res_pixel_u[0]) << "    ";
            	}*/
            }

           	/*if(row == 100 && col == 100){
           			uchar temp = (uchar)(int(res_pixel));
            		cout << res_pixel << "   " << (unsigned int)temp << "    ";
            		//cout << (unsigned int)res_pixel_u << endl;
           	}*/
            for(int i = 0; i < 1; ++i)
            {
            	res_pixel = (res_pixel>255) ? 255 : res_pixel;
                *result_pixel_ptr++ = (uchar)(int(res_pixel));
            }
        }
    }
}

int FluxTensorMethod::apply_averaging_filters(const Mat & input, Mat & result)
{
	static deque<Mat> axay_fifo;

	result = input.clone();

	Mat ax_result;
	filter2D(input, ax_result, -1, ax_filter);

	Mat ax_ay_result;
	filter2D(ax_result, ax_ay_result, -1, ay_filter);

	axay_fifo.push_back(ax_ay_result);
	if(axay_fifo.size() < (unsigned int)nAt)
		return 1;

	apply_temporal_filter(&axay_fifo, at_filter, nAt, result);

	axay_fifo.pop_front();
	return 0;
}

void FluxTensorMethod :: update(const Mat & input, Mat & result_frame)
{
	Mat temp_res[TENSOR_COMPONENTS_NUM+2];

	Mat gray_frame;
	convert2grayscale(input, gray_frame);
	result_frame = gray_frame.clone();

	compute_Ixt(gray_frame, temp_res[0]);
	compute_Iyt(gray_frame, temp_res[1]);
	if(compute_Itt(gray_frame, temp_res[2]) == 1)
		return;

	temp_res[3] = gray_frame.clone();
	compute_trace(temp_res, temp_res[3]);

	if(apply_averaging_filters(temp_res[3], temp_res[4]) == 1)
		return;



   // double rgb[3];
    const uchar * input_pixel_ptr;
    uchar * result_pixel_ptr;
    uchar mask;

    int height = gray_frame.rows;
    int width = gray_frame.cols;
    for(int row = 0; row < height; ++row)
    {
    	input_pixel_ptr = temp_res[4].ptr(row);
        result_pixel_ptr = result_frame.ptr(row);
        for(int col = 0; col < width; ++col)
        {
            //RGB reverted order
           // rgb[2] = (double) *input_pixel_ptr++;
           // rgb[1] = (double) *input_pixel_ptr++;
           // rgb[0] = (double) *input_pixel_ptr++;
           // double avr = (rgb[0]+rgb[1]+rgb[2])/3;
            double avr = (double) *input_pixel_ptr++;
        	mask = (avr>threshold) ? WHITE : BLACK;

            for(int i = 0; i < 1; ++i)
                *result_pixel_ptr++ = mask;
        }
    }
}


void FluxTensorMethod :: convert2grayscale(const Mat & input, Mat & output)
{
	Mat grayscale(input.size(), CV_8U);
	output = grayscale.clone();

    double rgb[3];
    const uchar * input_pixel_ptr;
    uchar * result_pixel_ptr;
    uchar gray_value;

    int height = input.rows;
    int width = input.cols;
    for(int row = 0; row < height; ++row)
    {
    	input_pixel_ptr = input.ptr(row);
        result_pixel_ptr = output.ptr(row);
        for(int col = 0; col < width; ++col)
        {
            //RGB reverted order
            rgb[2] = (double) *input_pixel_ptr++;
            rgb[1] = (double) *input_pixel_ptr++;
            rgb[0] = (double) *input_pixel_ptr++;
            //gray_value = (rgb[0]+rgb[1]+rgb[2])/3;
            gray_value = 0.2126*rgb[0]+0.7152*rgb[1]+0.0722*rgb[2];

            //for(int i = 0; i < 3; ++i)
                *result_pixel_ptr++ = gray_value;
        }
    }
}


