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

__global__ void pad_cufftcom(cufftComplex* out, int lx, int ly, double* Re, double* Im, int sx, int sy)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < sx && idy < sy) {
		out[(idy + ly / 4) * lx + (idx + lx / 4)] 
			= make_cuComplex((float)Re[idy * sx + idx], (float)Im[idy * sx + idx]);
	}


}

void kakucuda(My_ComArray_2D* in, int x, int y, float lamda, float d, float z) {

	cudaStream_t s1, s2, s3;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);
	cudaStreamCreate(&s3);

	double* devRe, * devIm;
	cudaMalloc((void**)&devRe, sizeof(double) * x * y);
	cudaMalloc((void**)&devIm, sizeof(double) * x * y);





	float* ReH, * ImH;
	cudaMalloc((void**)&ReH, sizeof(float) * x * y * 4);
	cudaMalloc((void**)&ImH, sizeof(float) * x * y * 4);

	float* ReHs, * ImHs;
	cudaMalloc((void**)&ReHs, sizeof(float) * x * y * 4);
	cudaMalloc((void**)&ImHs, sizeof(float) * x * y * 4);

	//s1~3でメモリコピーとHの計算を非同期実行
	float u = 1 / ((float)2 * x * d), v = 1 / ((float)2 * y * d);
	dim3 grid(32, 32), block(32, 32);
	Hcudaf << <grid, block, 0, s3>> > (ReH, ImH, 2 * x, 2 * y, u, v, z, lamda);
	shiftf << <grid, block, 0, s3>> > (ReHs, ImHs, ReH, ImH, 2 * x, 2 * y);

	cudaMemcpyAsync(devRe, in->Re, sizeof(double) * x * y, cudaMemcpyHostToDevice, s1);
	cudaMemcpyAsync(devIm, in->Im, sizeof(double) * x * y, cudaMemcpyHostToDevice, s2);
	
	//ストリーム終わり
	cudaStreamQuery(s1);
	cudaStreamQuery(s2);
	cudaStreamQuery(s3);


	//pad兼cufftcom
	cufftComplex* devpad;
	cudaMalloc((void**)&devpad, sizeof(cufftComplex) * 4 * x * y);
	cudaMemset(devpad, 0, sizeof(cufftComplex) * 4 * x * y);
	dim3 grid2(16, 16), block2(32, 32);
	pad_cufftcom << <grid2, block2 >> > (devpad, 2 * x, 2 * y, devRe, devIm, x, y);



	fft_2D_cuda_dev(2 * x, 2 * y, devpad);

	


	//掛け算
	cufftComplex* rslt;
	cudaMalloc((void**)&rslt, sizeof(cufftComplex) * x * y * 4);
	//dim3 grid3(1024, 1024), block3(1024, 1024);
	mulcomcufftcom<<<1024, 1024>>>(rslt, ReHs, ImHs, devpad, 4 * x * y);


	ifft_2D_cuda_dev(2 * x, 2 * y, rslt);



	cufftComplex* host;
	host = (cufftComplex*)malloc(sizeof(cufftComplex) * x * y * 4);

	/*cudaMallocHost((void**)&host, sizeof())*/
	cudaMemcpy(host, rslt, sizeof(cufftComplex) * x * y * 4, cudaMemcpyDeviceToHost);
	My_ComArray_2D* tmp;
	tmp = new My_ComArray_2D(4 * x * y, 2 * x, 2 * y);
	cufftcom_to_mycom(tmp, host, 4 * x * y);

	//inに出力
	tmp->extract_center(in);

	delete tmp;
	free(host);

	cudaFree(devpad);
	cudaFree(ReH);
	cudaFree(ImH);
	cudaFree(ReHs);
	cudaFree(ImHs);

	cudaFree(rslt);
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
	pad_cufftcom2cufftcom<<<grid, block>>>(devpad, 2 * x, 2 * y, devicein, x, y);
	fft_2D_cuda_dev(2 * x, 2 * y, devpad);

	//掛け算
	cufftComplex* rslt;
	cudaMalloc((void**)&rslt, sizeof(cufftComplex) * x * y * 4);
	mulcomcufftcom<<<1024, 1024>>>(rslt, ReHs, ImHs, devpad, 4 * x * y);

	ifft_2D_cuda_dev(2 * x, 2 * y, rslt);

	//deviceinへ0elim
	elimpad<< <grid, block >> > (devicein, x, y, devpad, 2 * x, 2 * y);

}

string impath = "./lena512x512.bmp";
string ompath2 = "./output2.bmp";
string lastpath = "./cudakaku.bmp";
string lastpath2 = "./cudakaku2.bmp";


int main(void) {
	My_Bmp* img;
	img = new My_Bmp(SX, SY);
	img->img_read(impath);





	//GPU,Hもkakuで計算
	My_ComArray_2D* comcuda;
	comcuda = new My_ComArray_2D(SX * SY, SX, SY);
	img->ucimg_to_double(comcuda->Re);

	clock_t t5 = clock();
	kakucuda(comcuda, SX, SY, (float)532e-09, (float)1.496e-05, (float)0.1);
	clock_t t6 = clock();

	cout << "GPU計算時間:" << (double)(t6 - t5) << endl;
	comcuda->power(comcuda->Re);

	My_Bmp* img4;
	img4 = new My_Bmp(SX, SY);
	img4->data_to_ucimg(comcuda->Re);
	img4->img_write(lastpath);







	//GPU,Hは別で計算して渡す
	My_ComArray_2D* comcuda2;
	comcuda2 = new My_ComArray_2D(SX * SY, SX, SY);
	img->ucimg_to_double(comcuda2->Re);

	clock_t ts = clock();

	cufftComplex* host;
	//cudaMallocHost((void**)&host, sizeof(cufftComplex) * SX * SY);
	host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
	set_cufftcomplex(host, comcuda2->Re, comcuda2->Im, SX * SY);
	cufftComplex* dev;
	cudaMalloc((void**)&dev, sizeof(cufftComplex) * SX * SY);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * SX * SY, cudaMemcpyHostToDevice);

	//Hをデバイスで計算
	float* ReH, * ImH;
	cudaMalloc((void**)&ReH, sizeof(float) * SX * SY * 4);
	cudaMalloc((void**)&ImH, sizeof(float) * SX * SY * 4);
	float* ReHs, * ImHs;
	cudaMalloc((void**)&ReHs, sizeof(float) * SX * SY * 4);
	cudaMalloc((void**)&ImHs, sizeof(float) * SX * SY * 4);
	float u = 1 / ((float)2 * SX * (float)1.496e-05), v = 1 / ((float)2 * SY * (float)1.496e-05);
	dim3 grid(32, 32), block(32, 32);
	Hcudaf << <grid, block>> > (ReH, ImH, 2 * SX, 2 * SY, u, v, (float)0.1, (float)532e-09);
	shiftf << <grid, block>> > (ReHs, ImHs, ReH, ImH, 2 * SX, 2 * SY);
	cudaFree(ReH);
	cudaFree(ImH);

	kaku_cuda(dev, ReHs, ImHs, SX, SY);
	cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
	My_ComArray_2D* out;
	out = new My_ComArray_2D(SX * SY, SX, SY);
	cufftcom_to_mycom(out, host, SX * SY);


	clock_t tl = clock();
	cout << "GPU2計算時間:" << (double)(tl - ts) << endl;

	out->power(out->Re);
	img4->data_to_ucimg(out->Re);
	img4->img_write(lastpath2);





	//CPU
	My_ComArray_2D* com2;
	com2 = new My_ComArray_2D(SX * SY, SX, SY);
	img->ucimg_to_double(com2->Re);

	clock_t t3 = clock();

	My_ComArray_2D* H2;
	H2 = new My_ComArray_2D(4 * SX * SY, 2 * SX, 2 * SY);
	H2->H_kaku(532e-09, 0.1, 1.496e-05);

	H2->kaku(com2, com2);

	clock_t t4 = clock();
	cout << "CPU計算時間:" << (double)(t4 - t3) << endl;


	com2->power(com2->Re);


	My_Bmp* img3;
	img3 = new My_Bmp(SX, SY);
	img3->data_to_ucimg(com2->Re);
	img3->img_write(ompath2);

	delete img;
	delete img4;
	delete comcuda;
	delete com2;
	delete H2;
	delete img3;
	return 0;
}