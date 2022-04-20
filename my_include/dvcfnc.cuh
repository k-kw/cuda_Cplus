#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//乱数ライブラリインクルード
#include <curand.h>
#include <curand_kernel.h>


//CUDA
#ifndef __CUDACC__
#define __CUDACC__
#endif 

//関数群

__global__ void cusetcufftcomplex(cuComplex* com, double* Re, double* Im, int size);


__global__ void normfft(cufftComplex* dev, int x, int y);

void fft_2D_cuda_dev(int x, int y, cufftComplex* dev);


void ifft_2D_cuda_dev(int x, int y, cufftComplex* dev);


__global__ void Hcudaf(float* Re, float* Im, int x, int y, float u, float v, float z, float lam);


__global__ void HcudacuCom(cuComplex* H, int x, int y, float z, float d, float lam);

__global__ void  shiftf(float* ore, float* oim, float* re, float* im, int x, int y);

__global__ void shiftCom(cuComplex* out, cuComplex* in, int x, int y);

//floatXcufftCom
__global__ void mulcomcufftcom(cufftComplex* out, float* re, float* im, cufftComplex* in, int s);


//doubleXcufftCom
__global__ void muldoublecomcufftcom(cufftComplex* out, double* re, double* im, cufftComplex* in, int s);

__global__ void Cmulfft(cufftComplex* out, cufftComplex* fin, cuComplex* in, int s);


__global__ void pad_cufftcom2cufftcom(cufftComplex* out, int lx, int ly, cufftComplex* in, int sx, int sy);

__global__ void elimpad(cufftComplex* out, int sx, int sy, cufftComplex* in, int lx, int ly);


void Hcudaf_shiftf(float* devReH, float* devImH, int x, int y, float d, float z, float lamda, dim3 grid, dim3 block);

void Hcudashiftcom(cuComplex* dev, int x, int y, float z, float d, float lamda, dim3 grid, dim3 block);


__global__ void cucompower(double* power, cuComplex* dev, int s);


__global__ void cunormaliphase(cuComplex* out, double* normali, int s);



