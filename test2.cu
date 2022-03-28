#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>

#include "my_all.h"
#include "Bmp_class_dll.h"
#include "complex_array_class_dll.h"

#include <opencv2//opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//乱数ライブラリインクルード
#include <curand.h>
#include <curand_kernel.h>

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

//パラメータ
#define BX 28       //bindat横
#define BY 28       //bindatの縦

//SX,SYは今のところ2の階乗の正方形のみ
#define SX 4096     //SLMでの横画素数(4で割れる整数に限る)
#define SY 2048   //SLMでの縦画素数(4で割れる整数に限る)

#define short 1024     //短辺

#define N 1       //画像の枚数
#define LENS_SIZE 32 //拡散板レンズのレンズサイズ

#define CHECK_NUM N  //シミュレーション画像をチェックする番号

//#define lam 532e-09  //波長
//#define d 1.496e-05 //画素ピッチ
//#define a 0.1 //伝搬距離1
//#define b 0.03 //伝搬距離2
//#define f 0.03 //焦点距離

float lamda = 532e-09;
float d = 3.74e-06;
float a = 0.1;
float b = 0.001;
float f = 0.001;


#define resolution pow(2, 8) //解像度
#define approx false    //レンズの式の近似


//CUDA
#define sqr(x) ((x)*(x))

#ifndef __CUDACC__
#define __CUDACC__
#endif 

//0埋め後画像サイズ
#define SX2 2*SX
#define SY2 2*SY

//1次元のグリッドとブロック
//総スレッド数
#define Nthread SX2*SY2
//ブロック内のスレッド数1=<BS=<1024
#define BS 1024



//二次元のグリッドとブロック
//ブロック当たりのスレッド数は合計1024までなので、block(32,32)より増やせない
//gridは上限ない？
//SX,SY=512,512
//dim3 grid(32, 32), block(32, 32), grid2(16, 16);


//今のところ2の階乗の正方形でしかできない,デバッグの必要あり？
//SX,SY=4096,4096
//dim3 grid(256, 256), block(32, 32), grid2(128, 128);

#define blockx 32
#define blocky 32

dim3 grid((SX2 + blockx - 1) / blockx, (SY2 + blocky - 1) / blocky), block(blockx, blocky), grid2((SX + blockx - 1) / blockx, (SY + blocky - 1) / blocky);





//関数群
void set_cufftcomplex(cufftComplex* cuconp, double* Re, double* Im, int size) {
    for (int i = 0; i < size; i++) {
        cuconp[i] = make_cuComplex((float)Re[i], (float)Im[i]);
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

    /*if (idy >= y || idx >= x) {
        return;
    }

    Re[idy * x + idx] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2))));
    Im[idy * x + idx] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2))));*/
    if (idy < y && idx < x) {
        Re[idy * x + idx] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2))));
        Im[idy * x + idx] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2))));
    }
}

void Hnotgpu(float* Re, float* Im, int x, int y, float u, float v, float z, float lam)
{
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            Re[i * x + j] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)j - x / 2)) - sqr(v * ((float)i - y / 2))));
            Im[i * x + j] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)j - x / 2)) - sqr(v * ((float)i - y / 2))));
        }
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

void shiftnotgpu(float* ore, float* oim, float* re, float* im, int x, int y) {

    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            if (j < x / 2 && i < y / 2) {
                ore[i * x + j] = re[(i + y / 2) * x + (j + x / 2)];
                ore[(i + y / 2) * x + (j + x / 2)] = re[i * x + j];
                oim[i * x + j] = im[(i + y / 2) * x + (j + x / 2)];
                oim[(i + y / 2) * x + (j + x / 2)] = im[i * x + j];
            }
            else if (j >= x / 2 && i < y / 2) {
                ore[i * x + j] = re[(i + y / 2) * x + (j - x / 2)];
                ore[(i + y / 2) * x + (j - x / 2)] = re[i * x + j];
                oim[i * x + j] = im[(i + y / 2) * x + (j - x / 2)];
                oim[(i + y / 2) * x + (j - x / 2)] = im[i * x + j];
            }
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


void Hnotgpushift(float* devReH, float* devImH, int x, int y, float d, float z, float lamda, dim3 grid, dim3 block) {
    /* float* ReH, * ImH;
     cudaMalloc((void**)&ReH, sizeof(float) * x * y);
     cudaMalloc((void**)&ImH, sizeof(float) * x * y);*/

    float* Re, * Im, * Res, * Ims;
    Re = new float[x * y];
    Im = new float[x * y];
    Res = new float[x * y];
    Ims = new float[x * y];

    float u = 1 / (x * d), v = 1 / (y * d);

    Hnotgpu(Re, Im, x, y, u, v, z, lamda);

    shiftnotgpu(Res, Ims, Re, Im, x, y);

    //cudaMemcpy(ReH, Re, sizeof(float) * x * y, cudaMemcpyHostToDevice);
    //cudaMemcpy(ImH, Im, sizeof(float) * x * y, cudaMemcpyHostToDevice);

    //shiftf << <grid, block >> > (devReH, devImH, ReH, ImH, x, y);

    /*cudaFree(ReH);
    cudaFree(ImH);*/

    cudaMemcpy(devReH, Res, sizeof(float) * x * y, cudaMemcpyHostToDevice);
    cudaMemcpy(devImH, Ims, sizeof(float) * x * y, cudaMemcpyHostToDevice);


    delete[]Re;
    delete[]Im;
    delete[]Res;
    delete[]Ims;

}














//ファイルパス
string binpath = "../../../../dat/bindat/1byte/fm_28_1.dat";
string simpath = "../../../../dat/simdat/SLM_phase/1byte/lsd/test_sim.dat";
string oriimg = "./test.bmp";
string simimg = "./testsim.bmp";
string t = "exp.bmp";

string impath = "./pad.bmp";


#define shx 4096
#define shy 2048

#define size SX*SY
#define pads 4*SX*SY

int main() {
    My_Bmp* img;
    img = new My_Bmp(SX, SY);
    img->img_read(impath);

    


    ////画像データを拡大するときCV_8Uでやる
    ////画像データをcv::Matにコピー
    //Mat bin_mat(SY, SX, CV_8U);
    //memcpy(bin_mat.data, img->img, SX * SY * sizeof(unsigned char));
    //imshow("View", bin_mat);
    //waitKey(0);
   
    ////拡大
    //Mat bin_mat_res(shy, shx, CV_8U);
    //resize(bin_mat, bin_mat_res, Size(shx, shy));
    //bin_mat.release();
    //imshow("View", bin_mat_res);
    //waitKey(0);

    My_ComArray_2D* com;
    com = new My_ComArray_2D(shy * shx, shx, shy);

    /*unsigned char* res;
    res = new unsigned char[shy * shx];

    memcpy(res, bin_mat_res.data, shy * shx * sizeof(unsigned char));*/

    com->data_to_ReIm(img->img);

    My_Bmp* out;
    out = new My_Bmp(shx, shy);
    out->data_to_ucimg(com->Re);
    string r = "./onlyres.bmp";
    out->img_write(r);

    cufftComplex* host;
    cudaMallocHost((void**)&host, sizeof(cufftComplex) * shy * shx);
    //host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
    set_cufftcomplex(host, com->Re, com->Im, shy * shx);


    cufftComplex* dev;
    cudaMalloc((void**)&dev, sizeof(cufftComplex) * shx * shy);
    cudaMemcpy(dev, host, sizeof(cufftComplex) * shx * shy, cudaMemcpyHostToDevice);

    fft_2D_cuda_dev(shx, shy, dev);

    cudaMemcpy(host, dev, sizeof(cufftComplex) * shx * shy, cudaMemcpyDeviceToHost);
    cufftcom2mycom(com, host, shx * shy);

    My_ComArray_2D* H;
    H = new My_ComArray_2D(shx * shy, shx, shy);
    H->H_kaku((double)lamda, (double)a, (double)d);

    H->mul_complex(com);
    set_cufftcomplex(host, H->Re, H->Im, shx * shy);
    cudaMemcpy(dev, host, sizeof(cufftComplex) * shx * shy, cudaMemcpyHostToDevice);

    /*com->power(com->Re);



    out->data_to_ucimg(com->Re);
    string fft = "./fftlena.bmp";
    out->img_write(fft);*/


    ifft_2D_cuda_dev(shx, shy, dev);

    cudaMemcpy(host, dev, sizeof(cufftComplex) * shx * shy, cudaMemcpyDeviceToHost);

    cufftcom2mycom(com, host, shx * shy);

    com->power(com->Re);


   
    out->data_to_ucimg(com->Re);
    string test = "./restore_rec.bmp";
    out->img_write(test);




    //float* Re, * Im;
    //Re = new float[pads];
    //Im = new float[pads];


    //float* Re2, * Im2;
    //Re2 = new float[pads];
    //Im2 = new float[pads];



    ////H配列をデバイス側で作成
    //float* ReHa, * ImHa;
    //cudaMalloc((void**)&ReHa, sizeof(float) * pads);
    //cudaMalloc((void**)&ImHa, sizeof(float) * pads);





    ////ブロック当たりのスレッド数は合計1024までなので、block(32,32)より増やせない
    ////gridは上限ない？
    //Hcudaf_shiftf(ReHa, ImHa, SX2, SY2, d, a, lamda, grid, block);
    //
    //cudaMemcpy(Re, ReHa, sizeof(float) * pads, cudaMemcpyDeviceToHost);
    //cudaMemcpy(Im, ImHa, sizeof(float) * pads, cudaMemcpyDeviceToHost);

    ////cout << Re[SX] << "\t" << Im[SX] << "\n";

    //

    //Hnotgpushift(ReHa, ImHa, SX2, SY2, d, a, lamda, grid, block);


    //cudaMemcpy(Re2, ReHa, sizeof(float) * pads, cudaMemcpyDeviceToHost);
    //cudaMemcpy(Im2, ImHa, sizeof(float) * pads, cudaMemcpyDeviceToHost);

    ////cout << Re[SX] << "\t" << Im[SX] << "\n";


    //My_ComArray_2D H(SX2* SY2, SX2, SY2);
    //H.H_kaku((double)lamda, (double)a, (double)d);


    //for (int i = 0; i < SY2; i++) {
    //    for (int j = 0; j < SX2; j++) {
    //        cout << Re[i * SX2 + j] << "\t" << Re2[i * SX2 + j] <<"\t" << H.Re[i * SX2 + j] << "\n";
    //    }
    //}

    ////float* ReHb, * ImHb;
    ////cudaMalloc((void**)&ReHb, sizeof(float) * pads);
    ////cudaMalloc((void**)&ImHb, sizeof(float) * pads);
    //////ブロック当たりのスレッド数は合計1024までなので、block(32,32)より増やせない
    //////gridは上限ない？
    ////Hcudaf_shiftf(ReHb, ImHb, 2 * shx, 2 * shy, d, b, lamda, grid, block);

    ////Hnotgpushift(ReHb, ImHb, 2 * shx, 2 * shy, d, b, lamda, grid, block);





















    return 0;
}