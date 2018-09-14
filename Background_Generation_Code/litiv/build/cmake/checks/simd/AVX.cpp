#include <opencv2/core.hpp>

#define CV_HARDWARE_SUPPORT_CHECK_FLAG_PREFIX CV_CPU_
#define CV_HARDWARE_SUPPORT_CHECK_FLAG_NAME AVX
#define XSTR_CONCAT(s1,s2) XSTR_CONCAT_BASE(s1,s2)
#define XSTR_CONCAT_BASE(s1,s2) s1##s2

int main(int,char**) {
    return (int)cv::checkHardwareSupport(XSTR_CONCAT(CV_HARDWARE_SUPPORT_CHECK_FLAG_PREFIX,CV_HARDWARE_SUPPORT_CHECK_FLAG_NAME));
}
