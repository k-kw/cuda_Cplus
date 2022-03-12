#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include "my_all.h"
#include "Bmp_class_dll.h"
#include "complex_array_class_dll.h"

#include <opencv2//opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

#ifndef __CUDACC__
#define __CUDACC__
#endif 

#define sqr(x) ((x)*(x))
#define SX 512
#define SY 512


//追加の依存ファイル設定の代わり
//opencvはDLLのPATHを通して動的リンクライブラリ(暗黙的リンク)として
#pragma comment(lib, "opencv_world454.lib")
#pragma comment(lib, "opencv_world454d.lib")


//bmpクラスを動的リンク(暗黙的リンク)
#pragma comment(lib, "Dll_bmp_class.lib")
//複素配列クラスを動的リンク(暗黙的リンク)
#pragma comment(lib, "DllComArray.lib")

//余分な警告削除
#pragma warning(disable:4996)

using namespace std;
using namespace cv;

string impath = "./lena512x512.bmp";
string ompath = "./output.bmp";


void set_cufftcomplex(cufftComplex* cuconp, double* Re, double* Im, int size) {
	for (int i = 0; i < size; i++) {
		cuconp[i] = make_cuComplex((float)Re[i], (float)Im[i]);
	}
}

void set_Re_cufftcomplex(cufftComplex* cuconp, double* Re, int size) {
	for (int i = 0; i < size; i++) {
		cuconp[i] = make_cuComplex((float)Re[i], 0.0f);
	}
}

//CUDA_FFT2D
void fft_2D_cuda(int x, int y, cufftComplex* host) {
	cufftHandle plan;

	cufftComplex* dev;

	cudaMalloc((void**)&dev, sizeof(cufftComplex) * x * y);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * x * y, cudaMemcpyHostToDevice);

	//フーリエ変換
	cufftPlan2d(&plan, x, y, CUFFT_C2C);
	cufftExecC2C(plan, dev, dev, CUFFT_FORWARD);

	cudaMemcpy(host, dev, sizeof(cufftComplex) * x * y, cudaMemcpyDeviceToHost);


	cudaFree(dev);
	cufftDestroy(plan);
}

//CUDA_IFFT2D
void ifft_2D_cuda(int x, int y, cufftComplex* host) {
	cufftHandle plan;

	cufftComplex* dev;

	cudaMalloc((void**)&dev, sizeof(cufftComplex) * x * y);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * x * y, cudaMemcpyHostToDevice);

	//フーリエ変換
	cufftPlan2d(&plan, x, y, CUFFT_C2C);
	cufftExecC2C(plan, dev, dev, CUFFT_INVERSE);

	cudaMemcpy(host, dev, sizeof(cufftComplex) * x * y, cudaMemcpyDeviceToHost);

	cudaFree(dev);
	cufftDestroy(plan);
}


int main(void) {
	/*cv::Mat inimg = cv::imread(impath);
	cv::imshow("lena", inimg);


	memcpy(padRe, bin_mat_pjr.data, PJRSX * PJRSY * sizeof(unsigned char));
	bin_mat_pjr.release();




	cv::Mat outimg;
	cvtColor(inimg, outimg, COLOR_BGR2GRAY);


	cv::waitKey(0);
	cv::imwrite(ompath, outimg);*/


    //入力画像を読みこみ
	My_Bmp* inimg;
	inimg = new My_Bmp(SX, SY);

	inimg->img_read(impath);

	//画像から複素配列に移動
	My_ComArray_2D* com;
	com = new My_ComArray_2D(SX * SY, SX, SY);

	inimg->ucimg_to_double(com->Re);

	//ホストメモリを確保
	cufftComplex* host;
	host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
	//複素配列の実部をセット
	set_Re_cufftcomplex(host, com->Re, SX * SY);


	//FFT
	fft_2D_cuda(SX, SY, host);

	//IFFT
	ifft_2D_cuda(SX, SY, host);


	

	for (int i = 0; i < SX * SY; i++) {
		com->Re[i] = (double)sqrt(sqr(cuCrealf(host[i])) + sqr(cuCimagf(host[i])));
	}




	


	My_Bmp* outimg;
	outimg = new My_Bmp(SX, SY);

	outimg->data_to_ucimg(com->Re);
	outimg->img_write(ompath);

	delete inimg;
	delete com;
	delete outimg;
	free(host);
	return 0;
}