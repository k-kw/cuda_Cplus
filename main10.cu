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
#define SX 1024     //SLMでの横画素数(4で割れる整数に限る)
#define SY 1024     //SLMでの縦画素数(4で割れる整数に限る)

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

    Hcudaf<<<grid, block >>>(ReH, ImH, x, y, u, v, z, lamda);
    shiftf<<<grid, block >>>(devReH, devImH, ReH, ImH, x, y);

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

int main() {
    clock_t start, lap;
    start = clock();

    //読み込みバイト確認
    int byte_num;
    do {
        cout << "\nバイナリデータを4バイトで読み込み：4を入力\t1バイトで読み込み：1を入力\n";
        cout << " 1 or 4: "; cin >> byte_num;

    } while (byte_num != 4 && byte_num != 1);


    //書き込みバイト確認
    int byte_numw;
    do {
        cout << "\nバイナリデータを4バイトで書き込み：4を入力\t1バイトで書き込み：1を入力\n";
        cout << " 1 or 4: "; cin >> byte_numw;

    } while (byte_numw != 4 && byte_numw != 1);



    //画像データを振幅情報(実部)とするか、位相に変換するか確認
    int ampl_or_phase;
    do {
        cout << "\n画像データを振幅情報(実部)とする場合：0を入力\t正規化後、位相情報とする場合：1を入力\n";
        cout << " 0 or 1 : "; cin >> ampl_or_phase;

    } while (ampl_or_phase != 0 && ampl_or_phase != 1);



    //ランダム位相かレンズアレイか確認
    int rand_or_lsd;
    do {
        cout << "\n拡散板をランダム位相とする場合：0を入力\t拡散板を微小レンズアレイとする場合：1を入力\n";
        cout << " 0 or 1 : "; cin >> rand_or_lsd;

    } while (rand_or_lsd != 0 && rand_or_lsd != 1);

    //ファイル入力・バイナリストリームオープン
    ifstream ifs(binpath, ios::binary /*| ios::in*/);
    //ファイル出力・バイナリストリームオープン
    ofstream ofs(simpath, ios::binary/* | ios::out*/);



    //両方オープンできたか確認
    if ((ifs) && (ofs)) {

        My_LensArray* Lens;
        Lens = new My_LensArray(SX * SY, SX, SY, approx, (double)f, (double)lamda, (double)d);

        if (rand_or_lsd == 0) {
            //ランダム拡散板
            Lens->diffuser_Random(0);

        }
        else {
            //レンズアレイ拡散板
            Lens->diffuser_Lensarray(LENS_SIZE);


        }


        //レンズの配列をデバイスへ送る
        double* ReL, * ImL;
        cudaMalloc((void**)&ReL, sizeof(double) * SX * SY);
        cudaMalloc((void**)&ImL, sizeof(double) * SX * SY);
        cudaMemcpy(ReL, Lens->Re, sizeof(double) * SX * SY, cudaMemcpyHostToDevice);
        cudaMemcpy(ImL, Lens->Im, sizeof(double) * SX * SY, cudaMemcpyHostToDevice);

        //画像データを格納するhost
        //ページ固定でもOK
        cufftComplex* host;
        cudaMallocHost((void**)&host, sizeof(cufftComplex)* SX* SY);
        //host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);

        //hostをコピーするデバイス側のメモリ確保
        cufftComplex* dev;
        cudaMalloc((void**)&dev, sizeof(cufftComplex) * SX * SY);

        //パディング後のメモリ確保
        cufftComplex* devpad;
        cudaMalloc((void**)&devpad, sizeof(cufftComplex) * SX * SY * 4);


        
        //H配列をデバイス側で作成
        float* ReHa, * ImHa;
        cudaMalloc((void**)&ReHa, sizeof(float) * SX * SY * 4);
        cudaMalloc((void**)&ImHa, sizeof(float) * SX * SY * 4);
        //ブロック当たりのスレッド数は合計1024までなので、block(32,32)より増やせない
        //gridは上限ない？
        Hcudaf_shiftf(ReHa, ImHa, 2 * SX, 2 * SY, d, a, lamda, grid, block);

        //Hnotgpushift(ReHa, ImHa, 2 * SX, 2 * SY, d, a, lamda, grid, block);

        float* ReHb, * ImHb;
        cudaMalloc((void**)&ReHb, sizeof(float) * SX * SY * 4);
        cudaMalloc((void**)&ImHb, sizeof(float) * SX * SY * 4);
        //ブロック当たりのスレッド数は合計1024までなので、block(32,32)より増やせない
        //gridは上限ない？
        Hcudaf_shiftf(ReHb, ImHb, 2 * SX, 2 * SY, d, b, lamda, grid, block);

        //Hnotgpushift(ReHb, ImHb, 2 * SX, 2 * SY, d, b, lamda, grid, block);


        //掛け算の出力メモリ確保
        cufftComplex* mul;
        cudaMalloc((void**)&mul, sizeof(cufftComplex) * SX * SY * 4);

        //レンズの掛け算出力メモリ確保
        cufftComplex* rslt;
        cudaMalloc((void**)&rslt, sizeof(cufftComplex)* SX* SY);


        for (int k = 0; k < N; k++) {
            //進捗状況表示
            if (k == 0) {
                cout << "\n\n\n-------------------------------simdataファイル作成中---------------------------------\n\n\n";
            }

            //バイナリ読み込み配列ポインタ
            unsigned char* chRe;
            int* intRe;



            chRe = new unsigned char[BX * BY];
            intRe = new int[BX * BY];



            //data読み取り
            if (byte_num == 1) {
                //1byteで一枚分読み込み
                ifs.read((char*)chRe, sizeof(unsigned char) * BX * BY);
                //上下反転
                invert_img<unsigned char>(chRe, chRe, BX, BY);

            }
            else {
                //4byteで一枚分読み込み
                ifs.read((char*)intRe, sizeof(int) * BX * BY);
                //上下反転
                invert_img<int>(intRe, intRe, BX, BY);


            }


            //画像データ確認
            if (k == N - 1) {

                My_Bmp* check;
                check = new My_Bmp(BX, BY);

                if (byte_num == 1) {

                    check->uc_to_img(chRe);
                    check->img_write(oriimg);
                }
                else {
                    check->data_to_ucimg(intRe);
                    check->img_write(oriimg);

                }



                delete check;
            }
            delete[]intRe;

            //画像データを拡大するときCV_8Uでやる
            //画像データをcv::Matにコピー
            Mat bin_mat(BY, BX, CV_8U);
            memcpy(bin_mat.data, chRe, BX * BY * sizeof(unsigned char));
            /*imshow("View", bin_mat);
            waitKey(0);*/
            delete[]chRe;

            //拡大
            Mat bin_mat_res(short, short, CV_8U);
            resize(bin_mat, bin_mat_res, Size(short, short));
            bin_mat.release();
            /*string resizeimg = "resize.bmp";
            imwrite(resizeimg, bin_mat_res);
            imshow("View", bin_mat_res);
            waitKey(0);*/

            //ゼロ埋めして合わせる
            Mat bin_mat_pjr(SY, SX, CV_8U);
            copyMakeBorder(bin_mat_res, bin_mat_pjr, (int)(SY - short) / 2, (int)(SY - short) / 2, (int)(SX - short) / 2, (int)(SX - short) / 2, BORDER_CONSTANT, 0);
            bin_mat_res.release();
            /*string padimg = "pad.bmp";
            imwrite(padimg, bin_mat_pjr);
            imshow("View", bin_mat_pjr);
            waitKey(0);*/

            unsigned char* padRe;
            padRe = new unsigned char[SX * SY];


            //拡大したcv::MatをpadReにコピー
            memcpy(padRe, bin_mat_pjr.data, SX * SY * sizeof(unsigned char));
            bin_mat_pjr.release();



            //画像データ確認
            if (k == N - 1) {


                My_Bmp* check;
                check = new My_Bmp(SX, SY);

                check->uc_to_img(padRe);
                check->img_write(t);

                delete check;

            }



            My_ComArray_2D* Complex;
            Complex = new My_ComArray_2D(SX * SY, SX, SY);

            Complex->data_to_ReIm(padRe);



            delete[]padRe;


            if (ampl_or_phase == 1) {
                //位相情報にする
                Complex->to_phase(Complex->Re);
            }




            //CUDAによるシミュレーション
            
            set_cufftcomplex(host, Complex->Re, Complex->Im, SX * SY);
            cudaMemcpy(dev, host, sizeof(cufftComplex) * SX * SY, cudaMemcpyHostToDevice);

            //角スペクトル
            cudaMemset(devpad, 0, sizeof(cufftComplex) * 4 * SX * SY);
            pad_cufftcom2cufftcom<<<grid2, block >>>(devpad, 2 * SX, 2 * SY, dev, SX, SY);

            //デバッグ
            cufftComplex* deb;
            deb = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY * 4);
            cudaMemcpy(deb, devpad, sizeof(cufftComplex)* SX* SY * 4, cudaMemcpyDeviceToHost);
            My_ComArray_2D* de;
            de = new My_ComArray_2D(SX * SY * 4, SX2, SY2);
            cufftcom2mycom(de, deb, SX* SY * 4);
            de->power(de->Re);
            My_Bmp* debug;
            debug = new My_Bmp(SX2, SY2);
            debug->data_to_ucimg(de->Re);
            string debugimg = "./pad.bmp";
            debug->img_write(debugimg);



            fft_2D_cuda_dev(SX2, SY2, devpad);

            //デバッグ
            cudaMemcpy(deb, devpad, sizeof(cufftComplex)* SX* SY * 4, cudaMemcpyDeviceToHost);
            cufftcom2mycom(de, deb, SX* SY * 4);
            de->power(de->Re);
            debug->data_to_ucimg(de->Re);
            string fftimg = "./fft1.bmp";
            debug->img_write(fftimg);

            //normfft<<<(Nthread + BS - 1) / BS, BS >>>(devpad, 2 * SX, 2 * SY);
            //掛け算
            mulcomcufftcom<<<(Nthread + BS - 1) / BS, BS >>>(mul, ReHa, ImHa, devpad, 4 * SX * SY);
            ifft_2D_cuda_dev(SX2, SY2, mul);
            //deviceinへ0elim
            elimpad<<<grid2, block >>>(dev, SX, SY, mul, 2 * SX, 2 * SY);

            //デバッグ
            cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            cufftcom2mycom(Complex, host, SX * SY);
            Complex->power(Complex->Re);
            My_Bmp* debug2;
            debug2 = new My_Bmp(SX, SY);
            debug2->data_to_ucimg(Complex->Re);
            string one = "./kaku1-1.bmp";
            debug2->img_write(one);


            //デバッグ
            cudaMemcpy(deb, devpad, sizeof(cufftComplex) * SX * SY * 4, cudaMemcpyDeviceToHost);
            cufftcom2mycom(de, deb, SX * SY * 4);
            My_ComArray_2D H(4 * SX * SY, 2 * SX, 2 * SY);
            H.H_kaku((double)lamda, (double)a, (double)d);
            H.mul_complex(de);
            set_cufftcomplex(deb, H.Re, H.Im, 4 * SX * SY);
            cudaMemcpy(mul, deb, sizeof(cufftComplex) * 4 * SX * SY, cudaMemcpyHostToDevice);
            ifft_2D_cuda_dev(SX2, SY2, mul);
            //deviceinへ0elim
            elimpad << <grid2, block >> > (dev, SX, SY, mul, 2 * SX, 2 * SY);





            ////デバッグ」
            //ifft_2D_cuda_dev(SX2, SY2, devpad);
            //elimpad << <grid2, block >> > (dev, SX, SY, devpad, 2 * SX, 2 * SY);

            //デバッグ
            cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            cufftcom2mycom(Complex, host, SX * SY);
            Complex->power(Complex->Re);
            debug2->data_to_ucimg(Complex->Re);
            string one2 = "./kaku1-2.bmp";
            debug2->img_write(one2);

            //レンズを掛け算
            muldoublecomcufftcom<<<(SX * SY + BS - 1) / BS, BS >>>(rslt, ReL, ImL, dev, SX * SY);

            //角スペクトル
            cudaMemset(devpad, 0, sizeof(cufftComplex) * 4 * SX * SY);
            pad_cufftcom2cufftcom<<<grid2, block >>>(devpad, 2 * SX, 2 * SY, rslt, SX, SY);
            fft_2D_cuda_dev(2 * SX, 2 * SY, devpad);
            mulcomcufftcom<<<(Nthread + BS - 1) / BS, BS >>>(mul, ReHb, ImHb, devpad, 4 * SX * SY);
            ifft_2D_cuda_dev(2 * SX, 2 * SY, mul);
            elimpad<<<grid2, block >>>(dev, SX, SY, mul, 2 * SX, 2 * SY);

            cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            cufftcom2mycom(Complex, host, SX * SY);

            

            //振幅計算
            Complex->power(Complex->Re);

            if (k == CHECK_NUM - 1) {

                My_Bmp* check;
                check = new My_Bmp(SX, SY);

                check->data_to_ucimg(Complex->Re);
                check->img_write(simimg);

                delete check;

            }


            double* Pline;
            Pline = new double[SX];

            mid_line<double>(Complex->Re, SX, SY, Pline);
            delete Complex;

            //書き込み配列
            int* intw;
            unsigned char* chw;
            intw = new int[SX];
            chw = new unsigned char[SX];

            norm_reso_n<double>(Pline, intw, (int)(resolution - 1), SX);
            delete[]Pline;
            to_uch(intw, SX, chw);


            //書き込み
            if (byte_numw == 1) {
                ofs.write((char*)chw, sizeof(unsigned char) * SX);

            }
            else {
                ofs.write((char*)intw, sizeof(int) * SX);

            }


            delete[]intw;
            delete[]chw;


            if ((k + 1) % 100 == 0) {
                cout << k + 1 << "まで完了----------------------------------------------\n";
                lap = clock();
                cout << setprecision(4) << (double)(lap - start) / CLOCKS_PER_SEC / 60 << "分経過\n\n";

            }
        }
        delete Lens;

        cudaFree(mul);
        cudaFree(rslt);
        cudaFree(ReL);
        cudaFree(ImL);
        cudaFree(host);
        cudaFree(dev);
        cudaFree(devpad);
        cudaFree(ReHa);
        cudaFree(ImHa);
        cudaFree(ReHb);
        cudaFree(ImHb);

    }

    else {
        cout << "データファイルを開けませんでした\n終了します。";

    }

    return 0;
}