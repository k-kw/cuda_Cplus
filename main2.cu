#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>
#include <cufft.h>
#include <cuda_runtime.h>

//#include "my_all.h"
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

__global__ void pad(double* out, int x, int y, double* in)
{
	int X = 2 * x;
	int Y = 2 * y;

	//入力された画像データを０埋めして倍の大きさの画像にする
	for (int i = Y / 4; i < y + Y / 4; i++) {
		for (int j = X / 4; j < x + X / 4; j++) {
			out[i * X + j] = in[(i - Y / 4) * x + (j - X / 4)];
		}
	}
}



//__global__ void H(cufftComplex* H, double lam, double z, double d, int x, int y) {
//	double u = 1 / ((double)x * d), v = 1 / ((double)y * d);
//
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//	int idy = blockDim.y * blockIdx.y + threadIdx.y;
//	
//	H[idx] = make_cuComplex((float)Re[i], (float)Im[i]);
//}



//角スペクトル法のHを直接計算する関数
void H_kaku(My_ComArray_2D* H, double lam, double z, double d, int x, int y) {

	My_ComArray_2D* tmp;
	tmp = new My_ComArray_2D(x * y, x, y);

	double u = 1 / ((double)x * d), v = 1 / ((double)y * d);
	//H計算
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			tmp->Re[i * x + j] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((double)j - x / 2)) - sqr(v * ((double)i - y / 2))));
			tmp->Im[i * x + j] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((double)j - x / 2)) - sqr(v * ((double)i - y / 2))));
		}
	}
	//Hシフト
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			if (j < x / 2 && i < y / 2) {
				H->Re[i * x + j] = tmp->Re[(i + y / 2) * x + (j + x / 2)];
				H->Re[(i + y / 2) * x + (j + x / 2)] = tmp->Re[i * x + j];
				H->Im[i * x + j] = tmp->Im[(i + y / 2) * x + (j + x / 2)];
				H->Im[(i + y / 2) * x + (j + x / 2)] = tmp->Im[i * x + j];
			}
			else if (j >= x / 2 && i < y / 2) {
				H->Re[i * x + j] = tmp->Re[(i + y / 2) * x + (j - x / 2)];
				H->Re[(i + y / 2) * x + (j - x / 2)] = tmp->Re[i * x + j];
				H->Im[i * x + j] = tmp->Im[(i + y / 2) * x + (j - x / 2)];
				H->Im[(i + y / 2) * x + (j - x / 2)] = tmp->Im[i * x + j];
			}
		}
	}

	delete tmp;
};


//2D画像の0pad関数(縦横それぞれ２倍にして0埋め、inとoutはサイズ違う)
void Opad(double* img_out, int x, int y, double* img_in) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* img_tmp;
	img_tmp = new double[X * Y];

	for (int i = 0; i < X * Y; i++) {
		img_tmp[i] = 0;
	}

	//入力された画像データを０埋めして倍の大きさの画像にする
	for (int i = Y / 4; i < y + Y / 4; i++) {
		for (int j = X / 4; j < x + X / 4; j++) {
			img_tmp[i * X + j] = img_in[(i - Y / 4) * x + (j - X / 4)];
		}
	}

	for (int i = 0; i < X * Y; i++) {
		img_out[i] = img_tmp[i];
	}

	delete[]img_tmp;

}

void fft_2D_cuda_dev(int x, int y, cufftComplex* dev)
{
	cufftHandle plan;
	cufftPlan2d(&plan, x, y, CUFFT_C2C);
	cufftExecC2C(plan, dev, dev, CUFFT_FORWARD);
	cufftDestroy(plan);
}


void ifft_2D_cuda_dev(int x, int y, cufftComplex* dev)
{
	cufftHandle plan;
	cufftPlan2d(&plan, x, y, CUFFT_C2C);
	cufftExecC2C(plan, dev, dev, CUFFT_INVERSE);
	cufftDestroy(plan);
}

void cufftcom_to_mycom(My_ComArray_2D* out, cufftComplex* in, int s) {
	for (int i = 0; i < s; i++) {
		out->Re[i] = (double)cuCrealf(in[i]);
		out->Im[i] = (double)cuCimagf(in[i]);

	}
}

//複素数配列乗算関数
void mul_com(int size, My_ComArray_2D* in1, My_ComArray_2D* in2, My_ComArray_2D* out) {
	double* Retmp, * Imtmp;
	Retmp = new double[size];
	Imtmp = new double[size];

	for (int i = 0; i < size; i++) {
		Retmp[i] = in1->Re[i] * in2->Re[i] - in1->Im[i] * in2->Im[i];
		Imtmp[i] = in1->Re[i] * in2->Im[i] + in1->Im[i] * in2->Re[i];
	}

	for (int i = 0; i < size; i++) {
		out->Re[i] = Retmp[i];
		out->Im[i] = Imtmp[i];
	}
	delete[]Retmp;
	delete[]Imtmp;
};

void elim(My_ComArray_2D* in, int ix, int iy, My_ComArray_2D* out) {
	int x, y;
	x = ix / 2;
	y = iy / 2;

	for (int i = iy / 4; i < y + iy / 4; i++) {
		for (int j = ix / 4; j < x + ix / 4; j++) {
			out->Re[(i - iy / 4) * x + (j - ix / 4)] = in->Re[i * ix + j];
			out->Im[(i - iy / 4) * x + (j - ix / 4)] = in->Im[i * ix + j];

		}
	}

}


__global__ void H(double d, double lam, double z, My_ComArray_2D* H, int x, int y)
{
	double u = 1 / ((double)x * d), v = 1 / ((double)y * d);
	//H計算
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {


			H->Re[i * x + j] = cos(2 * M_PI * z * sqrt((1 / lam) * (1 / lam) - (u * ((double)j - x / 2)) * (u * ((double)j - x / 2)) - (v * ((double)i - y / 2)) * (v * ((double)i - y / 2))));
			H->Im[i * x + j] = sin(2 * M_PI * z * sqrt((1 / lam) * (1 / lam) - (u * ((double)j - x / 2)) * (u * ((double)j - x / 2)) - (v * ((double)i - y / 2)) * (v * ((double)i - y / 2))));
			
		}
	}
}


__global__ void shift(cufftComplex* out, My_ComArray_2D* in, int x, int y)
{

	float tmpRe, tmpIm;

	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {

			if (j < x / 2 && i < y / 2) {

			    tmpRe = in->Re[(i + y / 2) * x + (j + x / 2)];
				tmpIm = in->Im[(i + y / 2) * x + (j + x / 2)];
				out[i * x + j] = make_cuComplex((float)tmpRe, (float)tmpIm);

				tmpRe = in->Re[i * x + j];
				tmpIm = in->Im[i * x + j];
				out[(i + y / 2) * x + (j + x / 2)] = make_cuComplex((float)tmpRe, (float)tmpIm);
			}

			else if (j >= x / 2 && i < y / 2) {

				tmpRe = in->Re[(i + y / 2) * x + (j - x / 2)];
				tmpIm = in->Im[(i + y / 2) * x + (j - x / 2)];
				out[i * x + j] = make_cuComplex((float)tmpRe, (float)tmpIm);

				tmpRe = in->Re[i * x + j];
				tmpIm = in->Im[i * x + j];
				out[(i + y / 2) * x + (j + x / 2)] = make_cuComplex((float)tmpRe, (float)tmpIm);
			}
		}
	}
}



void kaku(My_ComArray_2D* in, int x, int y, double lamda, double d, double z)
{

	My_ComArray_2D* tmp;
	tmp = new My_ComArray_2D(4 * x * y, 2 * x, 2 * y);
	in->zeropad(tmp);



	cufftComplex* host;
	host = (cufftComplex*)malloc(sizeof(cufftComplex) * x * y * 4);
	set_cufftcomplex(host, tmp->Re, tmp->Im, x * y * 4);


	cufftComplex* dev;
	cudaMalloc((void**)&dev, sizeof(cufftComplex) * x * y * 4);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * x * y * 4, cudaMemcpyHostToDevice);


	fft_2D_cuda_dev(2 * x, 2 * y, dev);
	cudaMemcpy(host, dev, sizeof(cufftComplex) * x * y * 4, cudaMemcpyDeviceToHost);
	cufftcom_to_mycom(tmp, host, 4 * x * y);


	My_ComArray_2D* H;
	H = new My_ComArray_2D(4 * x * y, 2 * x, 2 * y);
	H_kaku(H, lamda, z, d, 2 * x, 2 * y);
	mul_com(4 * x * y, tmp, H, tmp);
	set_cufftcomplex(host, tmp->Re, tmp->Im, 4 * x * y);

	cudaMemcpy(dev, host, sizeof(cufftComplex) * x * y * 4, cudaMemcpyHostToDevice);

	ifft_2D_cuda_dev(2 * x, 2 * y, dev);


	cudaMemcpy(host, dev, sizeof(cufftComplex) * x * y * 4, cudaMemcpyDeviceToHost);
	cufftcom_to_mycom(tmp, host, 4 * x * y);

	//inに出力
	elim(tmp, 2 * x, 2 * y, in);
	

	free(host);
	cudaFree(dev);
	delete tmp;
	delete H;

}

string impath = "./lena512x512.bmp";
string ompath = "./output.bmp";
string ompath2 = "./output2.bmp";

int main(void) {
	My_Bmp* img;
	img = new My_Bmp(SX, SY);
	img->img_read(impath);

	My_ComArray_2D* com;
	com = new My_ComArray_2D(SX * SY, SX, SY);

	img->ucimg_to_double(com->Re);

	clock_t t1 = clock();

	kaku(com, SX, SY, 532e-09, 1.496e-05, 0.1);

	clock_t t2 = clock();
	cout << "計算時間:" << (double)(t2 - t1) << endl;
	com->power(com->Re);

	My_Bmp* img2;
	img2 = new My_Bmp(SX, SY);
	img2->data_to_ucimg(com->Re);
	img2->img_write(ompath);









	My_ComArray_2D* com2;
	com2 = new My_ComArray_2D(SX * SY, SX, SY);
	img->ucimg_to_double(com2->Re);

	clock_t t3 = clock();

	My_ComArray_2D* H2;
	H2 = new My_ComArray_2D(4 * SX * SY, 2 * SX, 2 * SY);
	H2->H_kaku(532e-09, 0.1, 1.496e-05);

	H2->kaku(com2, com2);
	
	clock_t t4 = clock();
	cout << "計算時間:" << (double)(t4 - t3) << endl;

	
	com2->power(com2->Re);


	My_Bmp* img3;
	img3 = new My_Bmp(SX, SY);
	img3->data_to_ucimg(com2->Re);
	img3->img_write(ompath2);

	delete com;
	delete img;
	delete img2;
	delete com2;
	delete img3;
	return 0;
}