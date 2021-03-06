
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

// NOTE:
// CMake parses this file and automatically fills in the missing values;
// you should never modify it directly via "defines.hpp", but rather
// via "defines.hpp.in". Besides, keep includes out of here!

#pragma once

#ifndef __CMAKE_VAR_DEF_DROP_IN__
#define __CMAKE_VAR_DEF_DROP_IN__
// required for cmake cached variable drop-in
#define ON        1
#define TRUE      1
#define OFF       0
#define FALSE     0
#endif //__CMAKE_VAR_DEF_DROP_IN__

#define XSTR_CONCAT(s1,s2) XSTR_CONCAT_BASE(s1,s2)
#define XSTR_CONCAT_BASE(s1,s2) s1##s2
#define XSTR(s) XSTR_BASE(s)
#define XSTR_BASE(s) #s

#define TIMER_TIC(x) int64 XSTR_CONCAT(__nCPUTimerTick_,x) = cv::getTickCount()
#define TIMER_TOC(x) int64 XSTR_CONCAT(__nCPUTimerVal_,x) = cv::getTickCount()-XSTR_CONCAT(__nCPUTimerTick_,x)
#define TIMER_ELAPSED_MS(x) (double(XSTR_CONCAT(__nCPUTimerVal_,x))/(cv::getTickFrequency()/1000))

#define lvError(msg) throw lv::Exception(std::string("[lvError: ")+msg+"]",__PRETTY_FUNCTION__,__FILE__,__LINE__)
#define lvError_(msg,...) throw lv::Exception(std::string("[lvError: ")+msg+"]",__PRETTY_FUNCTION__,__FILE__,__LINE__,__VA_ARGS__)
#define lvAssert(expr) {if(!!(expr)); else throw lv::Exception(std::string("[lvAssertError] ("#expr")"),__PRETTY_FUNCTION__,__FILE__,__LINE__);}
#define lvAssert_(expr,msg) {if(!!(expr)); else throw lv::Exception(std::string("[lvAssertError: ")+msg+"] ("#expr")",__PRETTY_FUNCTION__,__FILE__,__LINE__);}
#define lvAssert__(expr,msg,...) {if(!!(expr)); else throw lv::Exception(std::string("[lvAssertError: ")+msg+"] ("#expr")",__PRETTY_FUNCTION__,__FILE__,__LINE__,__VA_ARGS__);}
#define glErrorCheck { \
    GLenum __errn = glGetError(); \
    if(__errn!=GL_NO_ERROR) \
        lvError_("glErrorCheck failed [code=%d, msg=%s]",__errn,gluErrorString(__errn)); \
}
#ifdef _DEBUG
#define lvDbgAssert(expr) lvAssert(expr)
#define lvDbgAssert_(expr,msg) lvAssert_(expr,msg)
#define lvDbgAssert__(expr,msg,...) lvAssert__(expr,msg,__VA_ARGS__)
#define lvDbgExceptionWatch lv::UncaughtExceptionLogger XSTR_CONCAT(__logger,__LINE__)(__PRETTY_FUNCTION__,__FILE__,__LINE__)
#define glDbgErrorCheck glErrorCheck
#else //ndefined(_DEBUG)
#define lvDbgAssert(expr)
#define lvDbgAssert_(expr,msg)
#define lvDbgAssert__(expr,msg,...)
#define lvDbgExceptionWatch
#define glDbgErrorCheck
#endif //ndefined(_DEBUG)
#define lvIgnore(x) (void)(x)
#define UNUSED(x) lvIgnore(x)

#ifndef D2R
#define D2R(d) ((d)*(M_PI/180.0))
#endif //ndefined(D2R)
#ifndef R2D
#define R2D(r) ((r)*(180.0/M_PI))
#endif //ndefined(R2D)
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif //ndefined(__STDC_FORMAT_MACROS)
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif //ndefined(_USE_MATH_DEFINES)

#if defined(_MSC_VER)
#ifndef __PRETTY_FUNCTION__
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif //ndefined(__PRETTY_FUNCTION__)
#ifndef NOMINMAX
#define NOMINMAX // go ahead, try using std::numeric_limits<...>::max() with the 'max' macro enabled
#endif //ndefined(NOMINMAX)
#ifndef __func__
#define __func__ __FUNCTION__
#endif //ndefined(__func__)
#define lvAssertHR(expr) [](HRESULT hr){if(FAILED(hr)) lvError_("HR assertion failed ("#expr" = 0x%x), %ws",hr,_com_error(hr).ErrorMessage());}(expr)
#ifdef _DEBUG
#define lvDbgAssertHR(expr) lvAssertHR(expr)
#else //ndefined(_DEBUG)
#define lvDbgAssertHR(expr)
#endif //ndefined(_DEBUG)
#if _MSC_VER<1900 // need >= MSVC 2015 (v140) toolchain
#error "This project requires full C++11 support (including constexpr and SFINAE)."
#endif //_MSC_VER<1900
#else //ndefined(_MSC_VER)
#if __cplusplus<201103L
#error "This project requires full C++11 support (including constexpr and SFINAE)."
#endif //__cplusplus<201103L
#endif //ndefined(_MSC_VER)
#define DEFAULT_NB_THREADS  1
#if DEFAULT_NB_THREADS<1
#error "Bad default number of threads specified."
#endif //DEFAULT_NB_THREADS<1

#define LITIV_VERSION       1.3.1
#define LITIV_VERSION_STR   XSTR(LITIV_VERSION)
#define LITIV_VERSION_MAJOR 1
#define LITIV_VERSION_MINOR 3
#define LITIV_VERSION_PATCH 1
#define LITIV_VERSION_SHA1  "4a82fbc570b7abecb40a03d9e08cf42920f07fce"

#if (defined(_MSC_VER) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__))
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define LV_PUBLIC __attribute__ ((dllexport))
#else //ndef(__GNUC__)
#define LV_PUBLIC __declspec(dllexport)
#endif //ndef(__GNUC__)
#else //ndef(BUILDING_DLL)
#ifdef __GNUC__
#define LV_PUBLIC __attribute__ ((dllimport))
#else //ndef(__GNUC__)
#define LV_PUBLIC __declspec(dllimport)
#endif //ndef(__GNUC__)
#endif //ndef(BUILDING_DLL)
#define LV_LOCAL
#else // if not on windows
#if (defined(__GNUC__) && (__GNUC__>=4))
#define LV_PUBLIC __attribute__ ((visibility ("default")))
#define LV_LOCAL  __attribute__ ((visibility ("hidden")))
#else // if not gcc, or not compat version
#define LV_PUBLIC
#define LV_LOCAL
#endif // if not gcc, or not compat version
#endif // if not on windows

#define EXTERNAL_DATA_ROOT  XSTR(/Users/Capricorn/Desktop/litiv-a/data)
#define SAMPLES_DATA_ROOT   XSTR(/Users/Capricorn/Desktop/litiv-a/samples/data)
#define CACHE_MAX_SIZE_GB   6LLU
#define TARGET_PLATFORM_x64 TRUE

#define HAVE_GLSL           OFF
#if HAVE_GLSL
#define TARGET_GL_VER_MAJOR 
#define TARGET_GL_VER_MINOR 
#define GLEW_EXPERIMENTAL   
#define HAVE_GLFW           
#define HAVE_FREEGLUT       
#endif //HAVE_GLSL

#define HAVE_CUDA           0
#if HAVE_CUDA
// ...
#endif //HAVE_CUDA

#define HAVE_OPENCL         0
#if HAVE_OPENCL
// ...
#endif //HAVE_OPENCL

#define HAVE_OPENGM         OFF
#if HAVE_OPENGM
#define HAVE_OPENGM_EXTLIB  ON
#define HAVE_OPENGM_CPLEX   
#define HAVE_OPENGM_GUROBI  
#define HAVE_OPENGM_HDF5    
#endif //HAVE_OPENGM

#define HAVE_MMX            OFF
#define HAVE_SSE            ON
#define HAVE_SSE2           ON
#define HAVE_SSE3           ON
#define HAVE_SSSE3          ON
#define HAVE_SSE4_1         ON
#define HAVE_SSE4_2         ON
#define HAVE_POPCNT         ON
#define HAVE_AVX            ON
#define HAVE_AVX2           ON

#define HAVE_STL_ALIGNED_ALLOC    
#define HAVE_POSIX_ALIGNED_ALLOC  1

#ifndef USE_VPTZ_STANDALONE
#define USE_VPTZ_STANDALONE       
#endif //USE_VPTZ_STANDALONE
#ifndef USE_BSDS500_BENCHMARK
#define USE_BSDS500_BENCHMARK     ON
#endif //USE_BSDS500_BENCHMARK
#define USE_KINECTSDK_STANDALONE  TRUE
