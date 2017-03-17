/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * (c) 2013 Quantitative Engineering Design (http://qed.ai)
 * Authors: William Wu, Jiehua Chen, Zhang Zhiming, Michał Łazowik
 * Primary Contact: William Wu (w@qed.ai)
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include <math.h>
#include "normxcorr2_masked.hpp"

#if __AVX2__
    #include <immintrin.h>
#endif

namespace cv {

//Xcorr_opencv constructor
Xcorr_opencv::Xcorr_opencv()
{
    eps=EPS/100.0;
}
//Xcorr_opencv destructor
Xcorr_opencv::~Xcorr_opencv()
{
    ;
}

//Initializing some member variables before calculation
int Xcorr_opencv::Initialization(
    string fixedImageName,
    string fixedMaskName,
    string movingImageName,
    string movingMaskName,
    double requiredFractionOfOverlappingPixels,
    double requiredNumberOfOverlappingPixels
){
    this->requiredFractionOfOverlappingPixels = requiredFractionOfOverlappingPixels;
    this->requiredNumberOfOverlappingPixels = requiredNumberOfOverlappingPixels;

    cv::Mat tmpImage;

    //load images
    cv::Mat fixedImage = cv::imread(fixedImageName.c_str());
    cv::Mat fixedMask = cv::imread(fixedMaskName.c_str());
    cv::Mat movingImage = cv::imread(movingImageName.c_str());
    cv::Mat movingMask = cv::imread(movingMaskName.c_str());

    // print dimensions
    cout << "Fixed image (scene): " << fixedImageName << ": " << fixedImage.size() << endl;
    cout << "Fixed mask (scene mask): " << fixedMaskName << ": " << fixedMask.size() << endl;
    cout << "Moving image (template): " << movingImageName << ": " << movingImage.size() << endl;
    cout << "Moving mask (template mask): " <<  movingMaskName << ": " << movingMask.size() << endl;

    channelnum = fixedImage.channels();

    // Ensure that the masks consist of only 0s and 1s.
    // Anything <= 0 is set to 0, and everything else is set to 1.
    threshold(fixedMask,fixedMask,0,1,CV_THRESH_BINARY);
    threshold(movingMask,movingMask,0,1,CV_THRESH_BINARY);

    // The fixed and moving images need to be masked for the equations below to
    // work correctly.
    fixedImage.copyTo(tmpImage,fixedMask);
    fixedImage = tmpImage.clone();
    movingImage.copyTo(tmpImage,movingMask);
    movingImage = tmpImage.clone();

    // Flip the moving image and mask in both dimensions so that its correlation
    // can be more easily handled.
    cv::Mat t,f;
    transpose(movingImage,t);
    flip(t,movingImage,1);
    transpose(movingMask,t);
    flip(t,movingMask,1);

    transpose(movingImage,t);
    flip(t,movingImage,1);
    transpose(movingMask,t);
    flip(t,movingMask,1);

    // Compute optimal FFT size
    // cvGetOptimalDFTSize returns minimum N >= size such that N = 2^p x 3^q x 5^r
    // for some p, q, r, which enables fast computation.
    combinedSize[0] = fixedImage.rows + movingImage.rows - 1;
    combinedSize[1] = fixedImage.cols + movingImage.cols - 1;

    optimalSize[0] = cvGetOptimalDFTSize(combinedSize[0]);
    optimalSize[1] = cvGetOptimalDFTSize(combinedSize[1]);
    // optimalSize[0] = combinedSize[0];
    // optimalSize[1] = combinedSize[1];
    optimalCvsize = cvSize(optimalSize[1], optimalSize[0]);

    fnorm = double(1) * double(optimalSize[0]) * double(optimalSize[1]) / 2.0;

    cout << "Dimensions of combined image: " << combinedSize[0] <<" x " << combinedSize[1] << endl;
    cout << "Optimal larger dimensions for fast DFT: " << optimalSize[0] <<" x " << optimalSize[1] << endl;

    // split image into separate channel images
    sbgr_fixedImage.resize(channelnum);
    sbgr_fixedMask.resize(channelnum);
    sbgr_movingImage.resize(channelnum);
    sbgr_movingMask.resize(channelnum);
    C_result.resize(channelnum);
    numberOfOverlapMaskedPixels_result.resize(channelnum);

    split(fixedImage,sbgr_fixedImage);
    split(fixedMask,sbgr_fixedMask);
    split(movingImage,sbgr_movingImage);
    split(movingMask,sbgr_movingMask);

    // initialize matrices
    optFixedImage.create(optimalSize[0],optimalSize[1],CV_32FC1);
    optFixedImage_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    optFixedMask.create(optimalSize[0],optimalSize[1],CV_32FC1);
    optFixedMask_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    optMovingImage.create(optimalSize[0],optimalSize[1],CV_32FC1);
    optMovingImage_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    optMovingMask.create(optimalSize[0],optimalSize[1],CV_32FC1);
    optMovingMask_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    optFixedSquared.create(optimalSize[0],optimalSize[1],CV_32FC1);
    optFixedSquared_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    optMovingSquared.create(optimalSize[0],optimalSize[1],CV_32FC1);
    optMovingSquared_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);

    /*
    // display images
    std::string windowName = "fixedImage";
    cv::namedWindow( windowName, CV_WINDOW_AUTOSIZE);
    cv::imshow( windowName, fixedImage );
    cv::waitKey();
    cv::imshow( windowName, fixedMask*255 );
    cv::waitKey();
    cv::imshow( windowName, movingImage );
    cv::waitKey();
    cv::imshow( windowName, movingMask*255 );
    cv::waitKey();
    */

    return 0;
}

//Calculate the masked correlations of all channels.
int Xcorr_opencv::CalXcorr()
{

    for(int i = 0; i < channelnum; i++)
    {
        optFixedImage = Mat::zeros(optimalSize[0],optimalSize[1],CV_32FC1);
        optFixedMask = Mat::zeros(optimalSize[0],optimalSize[1],CV_32FC1);
        optMovingImage = Mat::zeros(optimalSize[0],optimalSize[1],CV_32FC1);
        optMovingMask = Mat::zeros(optimalSize[0],optimalSize[1],CV_32FC1);
        optFixedSquared = Mat::zeros(optimalSize[0],optimalSize[1],CV_32FC1);
        optMovingSquared = Mat::zeros(optimalSize[0],optimalSize[1],CV_32FC1);
        for(int j = 0; j < sbgr_fixedImage[i].rows; j++)
        {
            for(int k = 0; k < sbgr_fixedImage[i].cols; k++)
            {
                optFixedImage.at<float>(j,k) = sbgr_fixedImage[i].at<unsigned char>(j,k);
            }
        }
        for(int j = 0; j < sbgr_fixedMask[i].rows; j++)
        {
            for(int k = 0; k < sbgr_fixedMask[i].cols; k++)
            {
                optFixedMask.at<float>(j,k) = sbgr_fixedMask[i].at<unsigned char>(j,k);
            }
        }
        for(int j = 0; j < sbgr_movingImage[i].rows; j++)
        {
            for(int k = 0; k < sbgr_movingImage[i].cols; k++)
            {
                optMovingImage.at<float>(j,k) = sbgr_movingImage[i].at<unsigned char>(j,k);
            }
        }
        for(int j = 0; j < sbgr_movingMask[i].rows; j++)
        {
            for(int k = 0; k < sbgr_movingMask[i].cols; k++)
            {
                optMovingMask.at<float>(j,k) = sbgr_movingMask[i].at<unsigned char>(j,k);
            }
        }
        double t = (double)getTickCount();
        CalculateOneChannelXcorr(i);
        t = (double)getTickCount() - t;
        printf("\tCalculated cross-correlation for one channel in %g ms\n", t*1000/getTickFrequency());
    }

    return 0;
}

//Calculate the masked correlation of one channel.
int Xcorr_opencv::CalculateOneChannelXcorr(int curChannel)
{

    if((!optFixedImage.isContinuous()))
    {
        printf("error: not continuous\n");
        exit(1);
    }
    // Only 6 FFTs are needed.
    FFT_opencv( optFixedImage,optFixedImage_FFT,FFT_SIGN_TtoF, sbgr_fixedMask[0].rows);
    FFT_opencv( optMovingImage, optMovingImage_FFT,FFT_SIGN_TtoF, sbgr_movingImage[0].rows);
    FFT_opencv( optFixedMask, optFixedMask_FFT,FFT_SIGN_TtoF, sbgr_fixedMask[0].rows);
    FFT_opencv( optMovingMask, optMovingMask_FFT,FFT_SIGN_TtoF, sbgr_movingImage[0].rows);

    // Compute and save these results
    cv::Mat numberOfOverlapMaskedPixels;
    IplImage *numberOfOverlapMaskedPixels_FFT;
    numberOfOverlapMaskedPixels.create(optimalSize[0],optimalSize[1],CV_32FC1);
    numberOfOverlapMaskedPixels_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    cvMulSpectrums(optMovingMask_FFT, optFixedMask_FFT, numberOfOverlapMaskedPixels_FFT, 0);
    FFT_opencv( numberOfOverlapMaskedPixels, numberOfOverlapMaskedPixels_FFT,FFT_SIGN_FtoT, optimalSize[0]);

    RoundClampDoubleMatrix(numberOfOverlapMaskedPixels, eps / 1000);

    cv::Mat maskCorrelatedFixed;
    IplImage *maskCorrelatedFixedFFT;
    maskCorrelatedFixed.create(optimalSize[0],optimalSize[1],CV_32FC1);
    maskCorrelatedFixedFFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    cvMulSpectrums(optMovingMask_FFT,optFixedImage_FFT,maskCorrelatedFixedFFT,0);
    FFT_opencv( maskCorrelatedFixed, maskCorrelatedFixedFFT,FFT_SIGN_FtoT, optimalSize[0]);

    cv::Mat maskCorrelatedRotatedMoving;
    IplImage *maskCorrelatedRotatedMovingFFT;
    maskCorrelatedRotatedMoving.create(optimalSize[0],optimalSize[1],CV_32FC1);
    maskCorrelatedRotatedMovingFFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    cvMulSpectrums(optFixedMask_FFT,optMovingImage_FFT,maskCorrelatedRotatedMovingFFT,0);
    FFT_opencv( maskCorrelatedRotatedMoving, maskCorrelatedRotatedMovingFFT,FFT_SIGN_FtoT, optimalSize[0]);

    cv::Mat numerator;
    IplImage *numerator_FFT;
    numerator.create(optimalSize[0],optimalSize[1],CV_32FC1);
    numerator_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    cvMulSpectrums(optMovingImage_FFT,optFixedImage_FFT,numerator_FFT,0);
    FFT_opencv( numerator, numerator_FFT,FFT_SIGN_FtoT, optimalSize[0]);

    numerator = numerator -( (maskCorrelatedFixed.mul(maskCorrelatedRotatedMoving)) / numberOfOverlapMaskedPixels);

    optFixedSquared = optFixedImage.mul(optFixedImage);
    FFT_opencv( optFixedSquared, optFixedSquared_FFT,FFT_SIGN_TtoF, sbgr_fixedImage[0].rows);

    cv::Mat fixedDenom;
    IplImage *fixedDenom_FFT;
    fixedDenom.create(optimalSize[0],optimalSize[1],CV_32FC1);
    fixedDenom_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    cvMulSpectrums(optMovingMask_FFT,optFixedSquared_FFT,fixedDenom_FFT,0);
    FFT_opencv( fixedDenom, fixedDenom_FFT,FFT_SIGN_FtoT, optimalSize[0]);

    fixedDenom = fixedDenom - ((maskCorrelatedFixed.mul(maskCorrelatedFixed)) / numberOfOverlapMaskedPixels);
    ThresholdLower(fixedDenom,0);

    optMovingSquared = optMovingImage.mul(optMovingImage);
    FFT_opencv( optMovingSquared, optMovingSquared_FFT,FFT_SIGN_TtoF, sbgr_movingImage[0].rows);

    cv::Mat movingDenom;
    IplImage *movingDenom_FFT;
    movingDenom.create(optimalSize[0],optimalSize[1],CV_32FC1);
    movingDenom_FFT = cvCreateImage(optimalCvsize,IPL_DEPTH_32F,2);
    cvMulSpectrums(optFixedMask_FFT,optMovingSquared_FFT,movingDenom_FFT,0);
    FFT_opencv( movingDenom, movingDenom_FFT,FFT_SIGN_FtoT, optimalSize[0]);

    movingDenom = movingDenom - ((maskCorrelatedRotatedMoving.mul(maskCorrelatedRotatedMoving)) / numberOfOverlapMaskedPixels);
    ThresholdLower(movingDenom,0);

    cv::Mat denom = fixedDenom.mul(movingDenom);
    denom.convertTo(denom, CV_64FC1); // Have to convert to double here
    sqrt(denom,denom);

    // denom is the sqrt of the product of positive numbers so it must be
    // positive or zero.  Therefore, the only danger in dividing the numerator
    // by the denominator is when dividing by zero.
    // Since the correlation value must be between -1 and 1, we therefore
    // saturate at these values.
    cv::Mat C = Mat::zeros(numerator.rows,numerator.cols,CV_32FC1);
    double maxAbs = MaxAbsValue(denom);
    double tol = 1000 * eps * maxAbs;

    double maximumNumberOfOverlappingPixels = 0;
    minMaxLoc(numberOfOverlapMaskedPixels, NULL, &maximumNumberOfOverlappingPixels);
    this->requiredNumberOfOverlappingPixels = max(
        this->requiredNumberOfOverlappingPixels,
        requiredFractionOfOverlappingPixels * maximumNumberOfOverlappingPixels
    );

    PostProcessing(C, numerator, denom, tol, -1, 1, numberOfOverlapMaskedPixels, this->requiredNumberOfOverlappingPixels);

    // Crop out the correct size.
    C_result[curChannel] = C(Range(0,combinedSize[0]),Range(0,combinedSize[1]));
    numberOfOverlapMaskedPixels_result[curChannel] = numberOfOverlapMaskedPixels(Range(0,combinedSize[0]),Range(0,combinedSize[1]));
    return 0;
}

// Divides numerator matrix by denominator matrix elementwise, scales results to
// [-1,1], and discards points with small overlap:
//  if (overlap < templateMask.size): set correlation to 0
int Xcorr_opencv::PostProcessing(cv::Mat &matC, cv::Mat &matNumerator, cv::Mat &matDenom, double tol, double minimum, double maximum, cv::Mat &numberOfOverlapMaskedPixels, double minimumOverlapSize)
{
    for(int i = 0; i < matDenom.rows;i++)
    {
        const float *numberOfOverlapMaskedPixelsRow = numberOfOverlapMaskedPixels.ptr<float>(i);
        float *matCRow = matC.ptr<float>(i);
        const double *matDenomRow = matDenom.ptr<double>(i);
        const float *matNumeratorRow = matNumerator.ptr<float>(i);
        for(int j = 0; j < matDenom.cols;j++)
        {
            if (numberOfOverlapMaskedPixelsRow[j] < minimumOverlapSize) {
                matCRow[j] = 0;
            } else {
                if(std::abs(matDenomRow[j]) > tol)
                {
                    matCRow[j] = matNumeratorRow[j] / matDenomRow[j];
                }
                if(matCRow[j] > maximum)
                {
                    matCRow[j] = maximum;
                }
                if(matCRow[j] < minimum)
                {
                    matCRow[j] = minimum;
                }
            }
        }
    }
    return 0;
}

// Return maximum absolute value of all elements in matrix.
double Xcorr_opencv::MaxAbsValue(cv::Mat &matImage )
{
    double minVal, maxVal;
    minMaxLoc(matImage, &minVal, &maxVal, NULL, NULL, noArray());
    return std::max(std::abs(minVal),std::abs(maxVal));
}

//Calculate the FFT of image Image_mat and return the result in Image_FFT
//if sign equals to FFT_SIGN_TtoF. If sign equals to FFT_SIGN_FtoT,
//calculate the IFFT of Image_FFT and return the result in Image_mat.
int Xcorr_opencv::FFT_opencv(const cv::Mat &Image_mat, IplImage *Image_FFT, int sign, int nonzerorows)
{
    // IPP DFT function is only called when nonzerorows = 0
    nonzerorows = 0;
    if(sign == FFT_SIGN_TtoF)
    {
        IplImage *dst = Image_FFT;
        const IplImage Image_Ipl = IplImage(Image_mat);
        const IplImage *src = &Image_Ipl;
        {
             IplImage *image_Re = 0, *image_Im = 0, *Fourier = 0;
             image_Re = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, 1);
             //Imaginary part
             image_Im = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, 1);
             //2 channels (image_Re, image_Im)
             Fourier = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, 2);
             // Real part conversion from u8 to 64f (double)
             cvConvertScale(src, image_Re, 1, 0);
             // Imaginary part (zeros)
             cvZero(image_Im);
             // Join real and imaginary parts and stock them in Fourier image
             cvMerge(image_Re, image_Im, 0, 0, Fourier);
             // Application of the forward Fourier transform
             cvDFT(Fourier, dst, CV_DXT_FORWARD, nonzerorows);
             cvReleaseImage(&image_Re);
             cvReleaseImage(&image_Im);
             cvReleaseImage(&Fourier);
        }
    }
    else
    {
        IplImage *ImageIm;
        IplImage *dst;
        IplImage Image_Ipl = IplImage(Image_mat);
        IplImage *ImageRe = &Image_Ipl;
        ImageIm = cvCreateImage(cvGetSize(ImageRe),IPL_DEPTH_32F,1);
        dst = cvCreateImage(cvGetSize(ImageRe),IPL_DEPTH_32F,2);
        cvDFT(Image_FFT,dst,CV_DXT_INV_SCALE, nonzerorows);
        cvSplit(dst,ImageRe,ImageIm,0,0);
        cvReleaseImage(&ImageIm);
        cvReleaseImage(&dst);
    }
    return 0;
}

//Scan matrix and compare each element with minimum.
//Assign all values less than minimum to minimum.
int Xcorr_opencv::ThresholdLower(cv::Mat &matImage, float min)
{
#if __AVX2__
    const __m256 __min = _mm256_set1_ps(min);
#endif
    for(int i = 0; i < matImage.rows; i++)
    {
        float *rrow = matImage.ptr<float>(i);
        int j = 0;
#if __AVX2__
        for (; j <= matImage.cols - 8; j += 8)
        {
            _mm256_storeu_ps(&rrow[j], _mm256_max_ps(_mm256_loadu_ps(&rrow[j]), __min));
        }
#endif
        for(; j < matImage.cols; j++)
        {
            if(rrow[j] < min)
            {
                rrow[j] = min;
            }
        }
    }
    return 0;
}

//Call round function on each element of the matrix.
int Xcorr_opencv::RoundClampDoubleMatrix(cv::Mat &matImage, float min)
{
#if __AVX2__
    const __m256 __min = _mm256_set1_ps(min);
#endif
    for(int i =0;i < matImage.rows;i++)
    {
        float *rrow = matImage.ptr<float>(i);
        int j = 0;
#if __AVX2__
        for (; j <= matImage.cols - 8; j += 8)
        {
            _mm256_storeu_ps(&rrow[j], _mm256_max_ps(__min,
                _mm256_round_ps(_mm256_loadu_ps(&rrow[j]), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
        }
#endif
        for(;j < matImage.cols;j++)
        {
            rrow[j] = round(rrow[j]);
        }
    }
    return 0;
}

//Get results for one channel.
double Xcorr_opencv::GetResult(cv::Mat &matC, cv::Mat &matNumberOfOverlapMaskedPixels, int intChannel)
{
    if(intChannel >= channelnum || intChannel < 0)
    {
        return -1;
    }
    matC = C_result[intChannel].clone();
    matNumberOfOverlapMaskedPixels = numberOfOverlapMaskedPixels_result[intChannel].clone();
    return 0;
}

}

