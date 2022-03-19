#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>

#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

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
	//doubleのときはZ2Z?
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


__global__ void Hcuda(double* Re, double* Im, int x, int y, double u, double v, double z, double lam)
{



	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idy < y && idx < x) {
		Re[idy * x + idx] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((double)idx - x / 2)) - sqr(v * ((double)idy - y / 2))));
		Im[idy * x + idx] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((double)idx - x / 2)) - sqr(v * ((double)idy - y / 2))));
	}


}



__global__ void  shift(double* ore, double* oim, double* re, double* im, int x, int y)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idy < y && idx < x) {

		if (idx < x / 2 && idy < y / 2) {
			ore[idy * x + idx] = re[(idy + y / 2) * x + (idx + x / 2)];
			ore[(idy + y / 2) * x + (idx + x / 2)] = re[idy * x + idx];
			oim[idy * x + idx] = im[(idy + y / 2) * x + (idx + x / 2)];
			oim[(idy + y / 2) * x + (idx + x / 2)] = im[idy * x + idx];
		}
		else if (idx >= x / 2 && idy < y / 2) {
			ore[idy * x + idx] = re[(idy + y / 2) * x + (idx - x / 2)];
			ore[(idy + y / 2) * x + (idx - x / 2)] = re[idy * x + idx];
			oim[idy * x + idx] = im[(idy + y / 2) * x + (idx - x / 2)];
			oim[(idy + y / 2) * x + (idx - x / 2)] = im[idy * x + idx];
		}



	}
}






__global__ void mulcom(double* ore, double* oim, double* re, double* im, double* re2, double* im2, int s)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < s) {
		ore[idx] = re[idx] * re2[idx] - im[idx] * im2[idx];
		oim[idx] = re[idx] * im2[idx] + im[idx] * re2[idx];

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

	double* re;
	cudaMalloc((void**)&re, sizeof(double) * x * y * 4);
	cudaMemcpy(re, tmp->Re, sizeof(double) * x * y * 4, cudaMemcpyHostToDevice);
	double* im;
	cudaMalloc((void**)&im, sizeof(double) * x * y * 4);
	cudaMemcpy(im, tmp->Im, sizeof(double) * x * y * 4, cudaMemcpyHostToDevice);





	double* ReH, * ImH;
	cudaMalloc((void**)&ReH, sizeof(double) * x * y * 4);
	cudaMalloc((void**)&ImH, sizeof(double) * x * y * 4);

	double* ReHs, * ImHs;
	cudaMalloc((void**)&ReHs, sizeof(double) * x * y * 4);
	cudaMalloc((void**)&ImHs, sizeof(double) * x * y * 4);

	double u = 1 / ((double)2 * SX * d), v = 1 / ((double)2 * SY * d);
	dim3 grid(32, 32), block(32, 32);
	Hcuda << <grid, block >> > (ReH, ImH, 2 * SX, 2 * SY, u, v, z, lamda);
	shift << <grid, block >> > (ReHs, ImHs, ReH, ImH, 2 * x, 2 * y);


	//掛け算
	double* re3;
	cudaMalloc((void**)&re3, sizeof(double) * x * y * 4);
	double* im3;
	cudaMalloc((void**)&im3, sizeof(double) * x * y * 4);

	//dim3 grid(128, 128), block(128, 128);
	mulcom << <1024, 1024 >> > (re3, im3, re, im, ReHs, ImHs, x * y * 4);

	cudaMemcpy(tmp->Re, re3, sizeof(double) * x * y * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp->Im, im3, sizeof(double) * x * y * 4, cudaMemcpyDeviceToHost);

	set_cufftcomplex(host, tmp->Re, tmp->Im, x * y * 4);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * x * y * 4, cudaMemcpyHostToDevice);



	ifft_2D_cuda_dev(2 * x, 2 * y, dev);


	cudaMemcpy(host, dev, sizeof(cufftComplex) * x * y * 4, cudaMemcpyDeviceToHost);
	cufftcom_to_mycom(tmp, host, 4 * x * y);

	//inに出力
	elim(tmp, 2 * x, 2 * y, in);


	free(host);
	cudaFree(dev);
	cudaFree(re);
	cudaFree(im);
	cudaFree(ReH);
	cudaFree(ImH);
	cudaFree(ReHs);
	cudaFree(ImHs);
	cudaFree(re3);
	cudaFree(im3);


	delete tmp;

}

//__global__ void set_cufftDoubleComplex(cufftDoubleComplex* out, double* Re, double* Im, int s) {
//
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if (idx < s) {
//		out[idx] = make_cuDoubleComplex(Re[idx], Im[idx]);
//	}
//}
//
//void cufftdoublecom_to_mycom(My_ComArray_2D* out, cufftDoubleComplex* in, int s) {
//
//	for (int i = 0; i < s; i++) {
//		out->Re[i] = cuCreal(in[i]);
//		out->Im[i] = cuCimag(in[i]);
//
//	}
//}

__global__ void Hcudaf(float* Re, float* Im, int x, int y, float u, float v, float z, float lam)
{



	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idy < y && idx < x) {
		Re[idy * x + idx] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2))));
		Im[idy * x + idx] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2))));
	}


}

__global__ void  shiftf(float* ore, float* oim, float* re, float* im, int x, int y)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idy < y && idx < x) {

		if (idx < x / 2 && idy < y / 2) {
			ore[idy * x + idx] = re[(idy + y / 2) * x + (idx + x / 2)];
			ore[(idy + y / 2) * x + (idx + x / 2)] = re[idy * x + idx];
			oim[idy * x + idx] = im[(idy + y / 2) * x + (idx + x / 2)];
			oim[(idy + y / 2) * x + (idx + x / 2)] = im[idy * x + idx];
		}
		else if (idx >= x / 2 && idy < y / 2) {
			ore[idy * x + idx] = re[(idy + y / 2) * x + (idx - x / 2)];
			ore[(idy + y / 2) * x + (idx - x / 2)] = re[idy * x + idx];
			oim[idy * x + idx] = im[(idy + y / 2) * x + (idx - x / 2)];
			oim[(idy + y / 2) * x + (idx - x / 2)] = im[idy * x + idx];
		}



	}
}

//floatXcufftCom
__global__ void mulcomcufftcom(cufftComplex* out, float* re, float* im, cufftComplex* in, int s)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < s) {

		out[idx] = make_cuComplex(re[idx] * cuCrealf(in[idx]) - im[idx] * cuCimagf(in[idx]),
			re[idx] * cuCimagf(in[idx]) + im[idx] * cuCrealf(in[idx]));

	}
}

void kakucuda(My_ComArray_2D* in, int x, int y, float lamda, float d, float z) {

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

	float* ReH, * ImH;
	cudaMalloc((void**)&ReH, sizeof(float) * x * y * 4);
	cudaMalloc((void**)&ImH, sizeof(float) * x * y * 4);

	float* ReHs, * ImHs;
	cudaMalloc((void**)&ReHs, sizeof(float) * x * y * 4);
	cudaMalloc((void**)&ImHs, sizeof(float) * x * y * 4);

	float u = 1 / ((float)2 * SX * d), v = 1 / ((float)2 * SY * d);
	dim3 grid(32, 32), block(32, 32);
	Hcudaf<<<grid, block>>>(ReH, ImH, 2 * SX, 2 * SY, u, v, z, lamda);
	shiftf<<<grid, block>>>(ReHs, ImHs, ReH, ImH, 2 * x, 2 * y);


	//掛け算
	cufftComplex* rslt;
	cudaMalloc((void**)&rslt, sizeof(cufftComplex) * x * y * 4);
	mulcomcufftcom<<<1024, 1024 >>>(rslt, ReHs, ImHs, dev, 4 * x * y);


	ifft_2D_cuda_dev(2 * x, 2 * y, rslt);


	cudaMemcpy(host, rslt, sizeof(cufftComplex) * x * y * 4, cudaMemcpyDeviceToHost);
	cufftcom_to_mycom(tmp, host, 4 * x * y);

	//inに出力
	tmp->extract_center(in);

	delete tmp;
	free(host);

	cudaFree(dev);
	cudaFree(ReH);
	cudaFree(ImH);
	cudaFree(ReHs);
	cudaFree(ImHs);

	cudaFree(rslt);
}

string impath = "./lena512x512.bmp";
string ompath = "./output.bmp";
string ompath2 = "./output2.bmp";
string lastpath = "./cudakaku.bmp";

int main(void) {
	My_Bmp* img;
	img = new My_Bmp(SX, SY);
	img->img_read(impath);

	/*My_ComArray_2D* com;
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
	img2->img_write(ompath);*/

	

	My_ComArray_2D* comcuda;
	comcuda = new My_ComArray_2D(SX * SY, SX, SY);
	img->ucimg_to_double(comcuda->Re);

	clock_t t5 = clock();

	kaku(comcuda, SX, SY, (float)532e-09, (float)1.496e-05, (float)0.1);

	clock_t t6 = clock();

	cout << "計算時間:" << (double)(t6 - t5) << endl;
	comcuda->power(comcuda->Re);

	My_Bmp* img4;
	img4 = new My_Bmp(SX, SY);
	img4->data_to_ucimg(comcuda->Re);
	img4->img_write(lastpath);





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

	//delete com;
	delete img;
	delete img4;
	//delete img2;
	delete comcuda;
	delete com2;
	delete H2;
	delete img3;
	return 0;
}