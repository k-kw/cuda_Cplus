#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>

#include "my_all.h"
#include "complex_array_class_dll.h"
#include "dvcfnc.cuh"



//copy
#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//乱数ライブラリインクルード
#include <curand.h>
#include <curand_kernel.h>
//copy


//複素配列クラスを動的リンク(暗黙的リンク)
#pragma comment(lib, "DllComArray.lib")

//余分な警告削除
#pragma warning(disable:4996)

using namespace std;

//CUDA
#ifndef __CUDACC__
#define __CUDACC__
#endif 

#define sqr(x) ((x)*(x))


//関数群

////template of under function
//template <class Type>
//__global__ void cusetcucomplex(cuComplex* com, Type* Re, Type* Im, int size)
//{
//
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//    if (idx < size) {
//        com[idx] = make_cuComplex((float)Re[idx], (float)Im[idx]);
//    }
//}


//use
__global__ void cusetcucomplex(cuComplex* com, double* Re, double* Im, int size)
{

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size) {
        com[idx] = make_cuComplex((float)Re[idx], (float)Im[idx]);
    }
}




//use
__global__ void normfft(cufftComplex* dev, int x, int y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < x * y) {
        dev[idx] = make_cuComplex(cuCrealf(dev[idx]) / (x * y), cuCimagf(dev[idx]) / (x * y));
    }
}


//use
void fft_2D_cuda_dev(int x, int y, cufftComplex* dev)
{
    cufftHandle plan;

    //cufftPlan2d 第2引数 : 最も遅く変化する次元のサイズ
    //cufftPlan2d 第3引数 : 最も速く変化する次元のサイズ
    cufftPlan2d(&plan, y, x, CUFFT_C2C);
    cufftExecC2C(plan, dev, dev, CUFFT_FORWARD);
    cufftDestroy(plan);
}

//use
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

//use
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

//use
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


//use
__global__ void Cmulfft(cuComplex* out, cuComplex* fin, cuComplex* in, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //レジスタver
    //cuComplex tmp1, tmp2;
    

    if (idx < s) {
        /*tmp1 = make_cuComplex(cuCrealf(fin[idx]), cuCimagf(fin[idx]));
        tmp2 = make_cuComplex(cuCrealf(in[idx]), cuCimagf(in[idx]));*/
        //out[idx] = cuCmulf(tmp1, tmp2);

        out[idx] = cuCmulf(fin[idx], in[idx]);

    }

}

//use
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

__global__ void elimpad2Cmulfft(cuComplex* outmlt, cuComplex* opponent, 
    int sx, int sy, cuComplex* in, int lx, int ly)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    cuComplex tmp1, tmp2;

    if (idx < sx && idy < sy) {
        //真ん中を取り出す
        tmp1 = make_cuComplex(cuCrealf(in[(idy + ly / 4) * lx + (idx + lx / 4)]), 
            cuCimagf(in[(idy + ly / 4) * lx + (idx + lx / 4)]));

        //レンズ配列等
        tmp2 = make_cuComplex(cuCrealf(opponent[idy * sx + idx]), cuCimagf(opponent[idy * sx + idx]));

        outmlt[idy * sx + idx] = cuCmulf(tmp1, tmp2);
    }

}


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

//use
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

//use
__global__ void elimpadcucompower(double* power ,int sx, int sy, cuComplex* dev, int lx, int ly)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    //レジスタver
    double tmp1, tmp2;

    if (idx < sx && idy < sy) {
        tmp1 = (double)sqr(cuCrealf(dev[(idy + ly / 4) * lx + (idx + lx / 4)]));
        tmp2 = (double)sqr(cuCimagf(dev[(idy + ly / 4) * lx + (idx + lx / 4)]));
        power[idy * sx + idx] = sqrt( tmp1 + tmp2 );
    }
}



//use
__global__ void cunormaliphase(cuComplex* out, double* normali, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        out[idx] = make_cuComplex((float)cos(2 * M_PI * normali[idx]), (float)sin(2 * M_PI * normali[idx]));

    }

}


