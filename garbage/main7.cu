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


__global__ void pad_cufftcom2cufftcom(cufftComplex* out, int lx, int ly, cufftComplex* in, int sx, int sy)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < sx && idy < sy) {
		out[(idy + ly / 4) * lx + (idx + lx / 4)] = in[idy * sx + idx];
	}


}


__global__ void elimpad(cufftComplex* out, int sx, int sy, cufftComplex* in, int lx, int ly)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < sx && idy < sy) {
		out[idy * sx + idx] = in[(idy + ly / 4) * lx + (idx + lx / 4)];
	}
}





void kaku_cuda(cufftComplex* devicein, float* ReHs, float* ImHs, int x, int y) {

	cufftComplex* devpad;
	cudaMalloc((void**)&devpad, sizeof(cufftComplex) * 4 * x * y);
	cudaMemset(devpad, 0, sizeof(cufftComplex) * 4 * x * y);

	dim3 grid(16, 16), block(32, 32);
	pad_cufftcom2cufftcom << <grid, block >> > (devpad, 2 * x, 2 * y, devicein, x, y);


	////デバッグ
	//cufftComplex* debug;
	//debug = (cufftComplex*)malloc(sizeof(cufftComplex) * 4 * x * y);
	//cudaMemcpy(debug, devpad, sizeof(cufftComplex) * 4 * x * y, cudaMemcpyDeviceToHost);
	//My_ComArray_2D* test;
	//test = new My_ComArray_2D(4 * x * y, 2 * x, 2 * y);
	//cufftcom_to_mycom(test, debug, 4 * x * y);
	//My_Bmp* timg;
	//timg = new My_Bmp(2 * x, 2 * y);
	//timg->data_to_ucimg(test->Re);
	//string wp = "./pad.bmp";
	//timg->img_write(wp);



	fft_2D_cuda_dev(2 * x, 2 * y, devpad);


	////デバッグ
	//cudaMemcpy(debug, devpad, sizeof(cufftComplex) * 4 * x * y, cudaMemcpyDeviceToHost);
	//cufftcom_to_mycom(test, debug, 4 * x * y);
	//My_ComArray_2D* H;
	//H = new My_ComArray_2D(4 * x * y, 2 * x, 2 * y);
	//H->H_kaku(532e-09, 0.1, 1.496e-05);
	//test->mul_complex(H);
	//cout << test->Re[4 * x * y - 500] <<endl<< test->Re[4 * x * y - 100] << endl;



	//掛け算
	cufftComplex* rslt;
	cudaMalloc((void**)&rslt, sizeof(cufftComplex) * x * y * 4);
	mulcomcufftcom << <1024, 1024 >> > (rslt, ReHs, ImHs, devpad, 4 * x * y);




 //   //デバッグ
	//cudaMemcpy(debug, rslt, sizeof(cufftComplex) * 4 * x * y, cudaMemcpyDeviceToHost);
	//cufftcom_to_mycom(test, debug, 4 * x * y);
	//cout << test->Re[4 * x * y - 500] <<endl<< test->Re[4 * x * y - 100] << endl;


	ifft_2D_cuda_dev(2 * x, 2 * y, rslt);


	////デバッグ
	//cudaMemcpy(debug, rslt, sizeof(cufftComplex) * 4 * x * y, cudaMemcpyDeviceToHost);
	//cufftcom_to_mycom(test, debug, 4 * x * y);
	//test->power(test->Re);
	//timg->data_to_ucimg(test->Re);
	//string wp2 = "./beforeelim.bmp";
	//timg->img_write(wp2);



	//deviceinへ0elim
	dim3 grid2(32, 32), block2(32, 32);
	elimpad<<<grid2, block2>>>(devicein, x, y, rslt, 2 * x, 2 * y);

}


void Hcudaf_shiftf(float* devReH, float* devImH, int x, int y, float d, float z, float lamda) {
	float* ReH, * ImH;
	cudaMalloc((void**)&ReH, sizeof(float) * x * y);
	cudaMalloc((void**)&ImH, sizeof(float) * x * y);

	float u = 1 / (x * d), v = 1 / (y * d);
	dim3 grid(32, 32), block(32, 32);
	Hcudaf << <grid, block >> > (ReH, ImH, x, y, u, v, z, lamda);
	shiftf << <grid, block >> > (devReH, devImH, ReH, ImH, x, y);

	cudaFree(ReH);
	cudaFree(ImH);
}


string impath = "./lena512x512.bmp";

float d = 1.496e-5;
float lamda = 532e-09;
float z = 0.1;


int main(void) {
	My_Bmp* img;
	img = new My_Bmp(SX, SY);
	img->img_read(impath);

	My_ComArray_2D* com;
	com = new My_ComArray_2D(SX * SY, SX, SY);
	img->ucimg_to_double(com->Re);

	/*float* debug;
	debug = (float*)malloc(sizeof(float) * SX * SY * 4);*/

	clock_t start = clock();


	//ページ固定でもOK
	cufftComplex* host;
	cudaMallocHost((void**)&host, sizeof(cufftComplex) * SX * SY);
	//host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
	set_cufftcomplex(host, com->Re, com->Im, SX * SY);

	cufftComplex* dev;
	cudaMalloc((void**)&dev, sizeof(cufftComplex) * SX * SY);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * SX * SY, cudaMemcpyHostToDevice);

	//Hをデバイスで計算
	/*float* ReH, * ImH;
	cudaMalloc((void**)&ReH, sizeof(float) * SX * SY * 4);
	cudaMalloc((void**)&ImH, sizeof(float) * SX * SY * 4);*/
	float* ReHs, * ImHs;
	cudaMalloc((void**)&ReHs, sizeof(float) * SX * SY * 4);
	cudaMalloc((void**)&ImHs, sizeof(float) * SX * SY * 4);
	Hcudaf_shiftf(ReHs, ImHs, 2 * SX, 2 * SY, d, z, lamda);
	/*float u = 1 / (2 * SX * d), v = 1 / (2 * SY * d);
	dim3 grid(32, 32), block(32, 32);
	Hcudaf << <grid, block >> > (ReH, ImH, 2 * SX, 2 * SY, u, v, z, lamda);
	shiftf << <grid, block >> > (ReHs, ImHs, ReH, ImH, 2 * SX, 2 * SY);
	cudaFree(ReH);
	cudaFree(ImH);*/


	////Hデバッグ
	//cudaMemcpy(debug, ReHs, sizeof(float) * 4 * SX * SY, cudaMemcpyDeviceToHost);
	//cout << debug[2 * SX * SY - 1] <<"\n" << debug[4 * SX * SY - 2] << endl;


	kaku_cuda(dev, ReHs, ImHs, SX, SY);


	cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
	My_ComArray_2D* out;
	out = new My_ComArray_2D(SX * SY, SX, SY);
	cufftcom_to_mycom(out, host, SX * SY);

	////デバッグ
	//cout <<"real" << out->Re[SX * SY - 1] << "\n" << out->Re[SX * SY - 200] << endl;
	//cout <<"imag" << out->Im[SX * SY - 1] << "\n" << out->Im[SX * SY - 200] << endl;

	out->power(out->Re);
	

	clock_t end = clock();
	cout << "計算時間:" << (double)(end - start) << endl;

	////デバッグ
	//cout <<"power" << out->Re[SX * SY - 1] << "\n" << out->Re[SX * SY - 200] << endl;





	/*My_ComArray_2D* test;
	test = new My_ComArray_2D(4 * SX * SY, 2 * SX, 2 * SY);
	test->H_kaku(lamda, z, d);

	cout << test->Re[2 * SX * SY - 1] <<"\n" << test->Re[4 * SX * SY - 2] << endl;*/



	My_Bmp* timg;
	timg = new My_Bmp(SX, SY);
	timg->data_to_ucimg(out->Re);
	string wp = "./cuda.bmp";
	timg->img_write(wp);

	cudaFree(dev);
	cudaFree(host);
	cudaFree(ReHs);
	cudaFree(ImHs);
	return 0;
}
