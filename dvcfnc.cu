#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>

#include "my_all.h"
#include "Bmp_class_dll.h"
#include "complex_array_class_dll.h"
#include "dvcfnc.cuh"

#include <opencv2//opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

//copy
#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//乱数ライブラリインクルード
#include <curand.h>
#include <curand_kernel.h>
//copy

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

//CUDA
#ifndef __CUDACC__
#define __CUDACC__
#endif 

#define sqr(x) ((x)*(x))


//関数群

//void set_cufftcomplex(cufftComplex* cuconp, double* Re, double* Im, int size) {
//    for (int i = 0; i < size; i++) {
//        cuconp[i] = make_cuComplex((float)Re[i], (float)Im[i]);
//    }
//}

//void set_cufftcomplex(cuComplex* cuconp, double* Re, double* Im, int size) {
//    for (int i = 0; i < size; i++) {
//        cuconp[i] = make_cuComplex((float)Re[i], (float)Im[i]);
//    }
//}


__global__ void cusetcufftcomplex(cuComplex* com, double* Re, double* Im, int size)
{

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size) {
        com[idx] = make_cuComplex((float)Re[idx], (float)Im[idx]);
    }
}


__global__ void normfft(cufftComplex* dev, int x, int y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < x * y) {
        dev[idx] = make_cuComplex(cuCrealf(dev[idx]) / (x * y), cuCimagf(dev[idx]) / (x * y));
    }
}



void fft_2D_cuda_dev(int x, int y, cufftComplex* dev)
{
    cufftHandle plan;

    //cufftPlan2d 第2引数 : 最も遅く変化する次元のサイズ
    //cufftPlan2d 第3引数 : 最も速く変化する次元のサイズ
    cufftPlan2d(&plan, y, x, CUFFT_C2C);
    cufftExecC2C(plan, dev, dev, CUFFT_FORWARD);
    cufftDestroy(plan);
}


void ifft_2D_cuda_dev(int x, int y, cufftComplex* dev)
{
    cufftHandle plan;

    //cufftPlan2d 第2引数 : 最も遅く変化する次元のサイズ
    //cufftPlan2d 第3引数 : 最も速く変化する次元のサイズ
    cufftPlan2d(&plan, y, x, CUFFT_C2C);
    cufftExecC2C(plan, dev, dev, CUFFT_INVERSE);
    cufftDestroy(plan);
}

void cufftcom2mycom(My_ComArray_2D* out, cufftComplex* in, int s) {
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


__global__ void HcudacuCom(cuComplex* H, int x, int y, float z, float d, float lam)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    float u = 1 / (x * d), v = 1 / (y * d);


    if (idy < y && idx < x) {
        H[idy * x + idx] = make_cuComplex(cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2)))),
            sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2)))));
    }
}


//void Hnotgpu(float* Re, float* Im, int x, int y, float u, float v, float z, float lam)
//{
//    for (int i = 0; i < y; i++) {
//        for (int j = 0; j < x; j++) {
//            Re[i * x + j] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)j - x / 2)) - sqr(v * ((float)i - y / 2))));
//            Im[i * x + j] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)j - x / 2)) - sqr(v * ((float)i - y / 2))));
//        }
//    }
//}


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

__global__ void shiftCom(cuComplex* out, cuComplex* in, int x, int y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idy < y && idx < x) {

        if (idx < x / 2 && idy < y / 2) {
            out[idy * x + idx] = in[(idy + y / 2) * x + (idx + x / 2)];
            out[(idy + y / 2) * x + (idx + x / 2)] = in[idy * x + idx];

        }
        else if (idx >= x / 2 && idy < y / 2) {
            out[idy * x + idx] = in[(idy + y / 2) * x + (idx - x / 2)];
            out[(idy + y / 2) * x + (idx - x / 2)] = in[idy * x + idx];

        }
    }
}


//void shiftnotgpu(float* ore, float* oim, float* re, float* im, int x, int y) {
//
//    for (int i = 0; i < y; i++) {
//        for (int j = 0; j < x; j++) {
//            if (j < x / 2 && i < y / 2) {
//                ore[i * x + j] = re[(i + y / 2) * x + (j + x / 2)];
//                ore[(i + y / 2) * x + (j + x / 2)] = re[i * x + j];
//                oim[i * x + j] = im[(i + y / 2) * x + (j + x / 2)];
//                oim[(i + y / 2) * x + (j + x / 2)] = im[i * x + j];
//            }
//            else if (j >= x / 2 && i < y / 2) {
//                ore[i * x + j] = re[(i + y / 2) * x + (j - x / 2)];
//                ore[(i + y / 2) * x + (j - x / 2)] = re[i * x + j];
//                oim[i * x + j] = im[(i + y / 2) * x + (j - x / 2)];
//                oim[(i + y / 2) * x + (j - x / 2)] = im[i * x + j];
//            }
//        }
//    }
//}

//floatXcufftCom
__global__ void mulcomcufftcom(cufftComplex* out, float* re, float* im, cufftComplex* in, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        out[idx] = make_cuComplex(re[idx] * cuCrealf(in[idx]) - im[idx] * cuCimagf(in[idx]),
            re[idx] * cuCimagf(in[idx]) + im[idx] * cuCrealf(in[idx]));

    }
}


//doubleXcufftCom
__global__ void muldoublecomcufftcom(cufftComplex* out, double* re, double* im, cufftComplex* in, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        out[idx] = make_cuComplex((float)re[idx] * cuCrealf(in[idx]) - (float)im[idx] * cuCimagf(in[idx]),
            (float)re[idx] * cuCimagf(in[idx]) + (float)im[idx] * cuCrealf(in[idx]));

    }
}

__global__ void Cmulfft(cufftComplex* out, cufftComplex* fin, cuComplex* in, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    cufftComplex tmp1, tmp2;
    tmp1 = make_cuComplex(cuCrealf(fin[idx]), cuCimagf(fin[idx]));
    tmp2 = make_cuComplex(cuCrealf(in[idx]), cuCimagf(in[idx]));

    if (idx < s) {

        out[idx] = cuCmulf(tmp1, tmp2);

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

//
//
//void kaku_cuda(cufftComplex* devicein, float* ReHs, float* ImHs, int x, int y, dim3 grid, dim3 block, int mulgrid, int mulblock) {
//
//    cufftComplex* devpad;
//    cudaMalloc((void**)&devpad, sizeof(cufftComplex) * 4 * x * y);
//    cudaMemset(devpad, 0, sizeof(cufftComplex) * 4 * x * y);
//
//    pad_cufftcom2cufftcom << <grid, block >> > (devpad, 2 * x, 2 * y, devicein, x, y);
//
//
//
//    fft_2D_cuda_dev(2 * x, 2 * y, devpad);
//
//
//    //掛け算
//    cufftComplex* rslt;
//    cudaMalloc((void**)&rslt, sizeof(cufftComplex) * x * y * 4);
//    mulcomcufftcom << <mulgrid, mulblock >> > (rslt, ReHs, ImHs, devpad, 4 * x * y);
//
//
//    ifft_2D_cuda_dev(2 * x, 2 * y, rslt);
//
//
//
//    //deviceinへ0elim
//    elimpad << <grid, block >> > (devicein, x, y, rslt, 2 * x, 2 * y);
//
//
//
//    cudaFree(devpad);
//    cudaFree(rslt);
//
//}
//
//void kakucuda(cufftComplex* devin, int inx, int iny, cufftComplex* devpad, cufftComplex* mul, float* ReHs, float* ImHs
//    , dim3 gridpadelim, dim3 block, int mulgrid, int mulblock) {
//
//    cudaMemset(devpad, 0, sizeof(cufftComplex) * 4 * inx * iny);
//    pad_cufftcom2cufftcom<<<gridpadelim, block >>>(devpad, 2 * inx, 2 * iny, devin, inx, iny);
//    fft_2D_cuda_dev(2 * inx, 2 * iny, devpad);
//    mulcomcufftcom<<<mulgrid, mulblock >> > (mul, ReHs, ImHs, devpad, 4 * inx * iny);
//    ifft_2D_cuda_dev(2 * inx, 2 * iny, mul);
//    elimpad<<<gridpadelim, block >>>(devin, inx, iny, mul, 2 * inx, 2 * iny);
//}
//

void Hcudaf_shiftf(float* devReH, float* devImH, int x, int y, float d, float z, float lamda, dim3 grid, dim3 block) {
    float* ReH, * ImH;
    cudaMalloc((void**)&ReH, sizeof(float) * x * y);
    cudaMalloc((void**)&ImH, sizeof(float) * x * y);

    float u = 1 / (x * d), v = 1 / (y * d);

    Hcudaf << <grid, block >> > (ReH, ImH, x, y, u, v, z, lamda);
    shiftf << <grid, block >> > (devReH, devImH, ReH, ImH, x, y);

    cudaFree(ReH);
    cudaFree(ImH);
}

void Hcudashiftcom(cuComplex* dev, int x, int y, float z, float d, float lamda, dim3 grid, dim3 block) {
    cuComplex* tmp;
    cudaMalloc((void**)&tmp, sizeof(cuComplex) * x * y);

    HcudacuCom << <grid, block >> > (tmp, x, y, z, d, lamda);
    shiftCom << <grid, block >> > (dev, tmp, x, y);

    cudaFree(tmp);

}


__global__ void cucompower(double* power, cuComplex* dev, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        power[idx] = sqrt((double)sqr(cuCrealf(dev[idx])) + (double)sqr(cuCimagf(dev[idx])));

    }
}





__global__ void cunormaliphase(cuComplex* out, double* normali, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        out[idx] = make_cuComplex((float)cos(2 * M_PI * normali[idx]), (float)sin(2 * M_PI * normali[idx]));

    }

}



//void Hnotgpushift(float* devReH, float* devImH, int x, int y, float d, float z, float lamda, dim3 grid, dim3 block) {
//   /* float* ReH, * ImH;
//    cudaMalloc((void**)&ReH, sizeof(float) * x * y);
//    cudaMalloc((void**)&ImH, sizeof(float) * x * y);*/
//
//    float* Re, * Im, * Res, * Ims;
//    Re = new float[x * y];
//    Im = new float[x * y];
//    Res = new float[x * y];
//    Ims = new float[x * y];
//
//    float u = 1 / (x * d), v = 1 / (y * d);
//
//    Hnotgpu(Re, Im, x, y, u, v, z, lamda);
//
//    shiftnotgpu(Res, Ims, Re, Im, x, y);
//
//    //cudaMemcpy(ReH, Re, sizeof(float) * x * y, cudaMemcpyHostToDevice);
//    //cudaMemcpy(ImH, Im, sizeof(float) * x * y, cudaMemcpyHostToDevice);
//
//    //shiftf << <grid, block >> > (devReH, devImH, ReH, ImH, x, y);
//
//    /*cudaFree(ReH);
//    cudaFree(ImH);*/
//
//    cudaMemcpy(devReH, Res, sizeof(float) * x * y, cudaMemcpyHostToDevice);
//    cudaMemcpy(devImH, Ims, sizeof(float) * x * y, cudaMemcpyHostToDevice);
//
//
//    delete[]Re;
//    delete[]Im;
//    delete[]Res;
//    delete[]Ims;
//
//}

