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

//パラメータ
#define BX 28       //bindat横
#define BY 28       //bindatの縦

//SLMに合わせたほうがいい？

//#define SX 4096     //SLMでの横画素数(4で割れる整数に限る)
//#define SY 2400     //SLMでの縦画素数(4で割れる整数に限る)
//float d = 3.74e-06;
//#define short 2400    //短辺

//SLM解像度
#define SLMX 4096     //SLMでの横画素数(4で割れる整数に限る)
#define SLMY 2400     //SLMでの縦画素数(4で割れる整数に限る)
#define short 2400    //短辺

//シミュレーション配列サイズ
#define SX 8192
#define SY 4800

//SX,SYの画素ピッチ
float d = 1.87e-06;

//0埋め後画像サイズ
#define SX2 (2*SX)
#define SY2 (2*SY)
#define SIZE (SX*SY)      //パディング前サイズ
#define PADSIZE (SX2*SY2) //パディング後サイズ

#define N 6       //画像の枚数
#define CHECK_NUM N  //シミュレーション画像をチェックする番号

//#define lam 532e-09  //波長
//#define d 1.496e-05 //画素ピッチ
//#define a 0.1 //伝搬距離1
//#define b 0.03 //伝搬距離2
//#define f 0.03 //焦点距離

//波長
float lamda = 532e-09;

//レンズ拡散版の寸法とSLMから決める
//#define LENS_SIZE 32 //拡散板レンズのレンズサイズ
//1mm(レンズ角)/d(SLMピッチ)=267より
#define LENS_SIZE 512

//伝搬距離と焦点距離
float a = 0.0066;
//float b = 0.03;
float b = 0.0033;
//float f = 0.001;
//フライアイレンズのデータシートより
float f = 0.0033;

////NEW
////SLM解像度に対する、カメラの解像度の割合
//#define SC 0.5
////カメラの解像度
//#define CAMX (int)(SX*SC)
//#define CAMY (int)(SY*SC)
////NEW

#define resolution pow(2, 8) //解像度
#define approx false    //レンズの式の近似
#define sqr(x) ((x)*(x))

//copy
//CUDA
#ifndef __CUDACC__
#define __CUDACC__
#endif 
//copy


//1次元のグリッドとブロック
//総スレッド数
// PADSIZEに同じ
//#define Nthread SX2*SY2
// 
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




//テンプレート関数だけ別にするとうまくいかない
template <class Type>
__global__ void cunormali(Type* devin, Type* devout, Type max, Type min, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        devout[idx] = (devin[idx] - min) / (max - min);

    }
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
        //double* ReL, * ImL;
        //cudaMalloc((void**)&ReL, sizeof(double) * SX * SY);
        //cudaMalloc((void**)&ImL, sizeof(double) * SX * SY);
        //cudaMemcpy(ReL, Lens->Re, sizeof(double) * SX * SY, cudaMemcpyHostToDevice);
        //cudaMemcpy(ImL, Lens->Im, sizeof(double) * SX * SY, cudaMemcpyHostToDevice);
        //cuComplex* Lhost, * Ldev;
        //cudaMallocHost((void**)&Lhost, sizeof(cuComplex) * SX * SY);
        //cudaMalloc((void**)&Ldev, sizeof(cuComplex) * SX * SY);
        //set_cufftcomplex(Lhost, Lens->Re, Lens->Im, SX * SY);

        //デバイス、double メモリ
        double* dvbfd, * dvbfd2;
        cudaMalloc((void**)&dvbfd, sizeof(double) * SIZE);
        cudaMalloc((void**)&dvbfd2, sizeof(double) * SIZE);

        
        cudaMemcpy(dvbfd, Lens->Re, sizeof(cuComplex) * SX * SY, cudaMemcpyHostToDevice);
        cudaMemcpy(dvbfd2, Lens->Im, sizeof(cuComplex) * SX * SY, cudaMemcpyHostToDevice);

        cuComplex* Ldev;
        cudaMalloc((void**)&Ldev, sizeof(cuComplex) * SIZE);
        cusetcufftcomplex<<<(SIZE + BS - 1) / BS, BS >>>(Ldev, dvbfd, dvbfd2, SIZE);

        //cudaMemcpy(Ldev, Lhost, sizeof(cuComplex) * SX * SY, cudaMemcpyHostToDevice);
        //画像データを格納するhost
        //ページ固定でもOK
        //cufftComplex* host;
        //cudaMallocHost((void**)&host, sizeof(cufftComplex)* SX* SY);
        ////host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
        ////hostをコピーするデバイス側のメモリ確保
        //cufftComplex* devbuf_cufc;
        //cudaMalloc((void**)&devbuf_cufc, sizeof(cufftComplex) * SX * SY);


        //デバイス,cufftComplexメモリ
        cufftComplex* dvbffc;
        cudaMalloc((void**)&dvbffc, sizeof(cufftComplex) * SIZE);


        //デバイス,cufftComplex,PADSIZEメモリ
        cufftComplex* dvbffcpd;
        cudaMalloc((void**)&dvbffcpd, sizeof(cufftComplex)* PADSIZE);

        //H配列をデバイス側で作成
        //float* ReHa, * ImHa;
        //cudaMalloc((void**)&ReHa, sizeof(float) * SX * SY * 4);
        //cudaMalloc((void**)&ImHa, sizeof(float) * SX * SY * 4);
        ////ブロック当たりのスレッド数は合計1024までなので、block(32,32)より増やせない
        ////gridは上限ない？
        //Hcudaf_shiftf(ReHa, ImHa, 2 * SX, 2 * SY, d, a, lamda, grid, block);
        ////Hnotgpushift(ReHa, ImHa, 2 * SX, 2 * SY, d, a, lamda, grid, block);
        //float* ReHb, * ImHb;
        //cudaMalloc((void**)&ReHb, sizeof(float) * SX * SY * 4);
        //cudaMalloc((void**)&ImHb, sizeof(float) * SX * SY * 4);
        ////ブロック当たりのスレッド数は合計1024までなので、block(32,32)より増やせない
        ////gridは上限ない？
        //Hcudaf_shiftf(ReHb, ImHb, 2 * SX, 2 * SY, d, b, lamda, grid, block);
        ////Hnotgpushift(ReHb, ImHb, 2 * SX, 2 * SY, d, b, lamda, grid, block);


        cuComplex* Ha;
        cudaMalloc((void**)&Ha, sizeof(cuComplex)* SX * SY * 4);
        Hcudashiftcom(Ha, SX2, SY2, a, d, lamda, grid, block);
        cuComplex* Hb;
        cudaMalloc((void**)&Hb, sizeof(cuComplex) * SX * SY * 4);
        Hcudashiftcom(Hb, SX2, SY2, b, d, lamda, grid, block);



        ////掛け算の出力メモリ確保
        //cufftComplex* dvbf_cfc_pad;
        //cudaMalloc((void**)&dvbf_cfc_pad, sizeof(cufftComplex) * SX * SY * 4);
        ////レンズの掛け算出力メモリ確保
        //cufftComplex* devbuf_cufc_2;
        //cudaMalloc((void**)&devbuf_cufc_2, sizeof(cufftComplex)* SX* SY);
        ////振幅格納配列
        //double* devbuf_db;
        //cudaMalloc((void**)&devbuf_db, sizeof(double) * SIZE);
        //double* devRe;

        


        for (int k = 0; k < N; k++) {
            //進捗状況表示
            if (k == 0) {
                cout << "\n\n\n-------------------------------出力ファイル作成中---------------------------------\n\n\n";
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
            Mat bin_mat_pjr(SLMY, SLMX, CV_8U);
            copyMakeBorder(bin_mat_res, bin_mat_pjr, (int)(SLMY - short) / 2, (int)(SLMY - short) / 2, (int)(SLMX - short) / 2, (int)(SLMX - short) / 2, BORDER_CONSTANT, 0);
            bin_mat_res.release();
            /*string padimg = "pad.bmp";
            imwrite(padimg, bin_mat_pjr);
            imshow("View", bin_mat_pjr);
            waitKey(0);*/

            unsigned char* padRe;
            padRe = new unsigned char[SLMX * SLMY];


            //拡大したcv::MatをpadReにコピー
            memcpy(padRe, bin_mat_pjr.data, SLMX * SLMY * sizeof(unsigned char));
            bin_mat_pjr.release();



            //画像データ確認
            if (k == N - 1) {

                My_Bmp* check;
                check = new My_Bmp(SLMX, SLMY);

                check->uc_to_img(padRe);
                check->img_write(t);

                delete check;

            }

            My_ComArray_2D* Complex, *tmp;
            Complex = new My_ComArray_2D(SX * SY, SX, SY);
            tmp = new My_ComArray_2D(SLMX * SLMY, SLMX, SLMY);

            tmp->data_to_ReIm(padRe);

            //tmpをComplexに拡大,格納



            delete[]tmp;

            
            cudaMemcpy(dvbfd, Complex->Re, sizeof(double) * SIZE, cudaMemcpyHostToDevice);
            cudaMemcpy(dvbfd2, Complex->Im, sizeof(double) * SIZE, cudaMemcpyHostToDevice);

            if (ampl_or_phase == 0) {
                //振幅変調
                cusetcufftcomplex<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbfd, dvbfd2, SIZE);

            }
            else {
                //位相変調
                double* Remax, * Remin;
                Remax = new double;
                Remin = new double;
                *Remax = get_max<double>(Complex->Re, SIZE);
                *Remin = get_min<double>(Complex->Re, SIZE);
                
                
                cunormali<double><<<(SIZE + BS - 1) / BS, BS >>>(dvbfd, dvbfd2, *Remax, *Remin, SIZE);
                cunormaliphase<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbfd2, SIZE);
                delete Remax; delete Remin;
            }

            delete[]padRe;

            //OLD
            //if (ampl_or_phase == 1) {
            //    //位相情報にする
            //    Complex->to_phase(Complex->Re);
            //}
            ////CUDAによるシミュレーション
            //
            //set_cufftcomplex(host, Complex->Re, Complex->Im, SX * SY);
            //cudaMemcpy(dev, host, sizeof(cufftComplex) * SX * SY, cudaMemcpyHostToDevice);
            //OLD

            //角スペクトル
            cudaMemset(dvbffcpd, 0, sizeof(cufftComplex) * 4 * SX * SY);
            pad_cufftcom2cufftcom<<<grid2, block >>>(dvbffcpd, 2 * SX, 2 * SY, dvbffc, SX, SY);

            ////デバッグ
            //cufftComplex* deb;
            //deb = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY * 4);
            //cudaMemcpy(deb, devpad, sizeof(cufftComplex)* SX* SY * 4, cudaMemcpyDeviceToHost);
            //My_ComArray_2D* de;
            //de = new My_ComArray_2D(SX * SY * 4, SX2, SY2);
            //cufftcom2mycom(de, deb, SX* SY * 4);
            //de->power(de->Re);
            //My_Bmp* debug;
            //debug = new My_Bmp(SX2, SY2);
            //debug->data_to_ucimg(de->Re);
            //string debugimg = "./pad.bmp";
            //debug->img_write(debugimg);



            fft_2D_cuda_dev(SX2, SY2, dvbffcpd);

            ////デバッグ
            //cudaMemcpy(deb, devpad, sizeof(cufftComplex)* SX* SY * 4, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(de, deb, SX* SY * 4);
            //de->power(de->Re);
            //debug->data_to_ucimg(de->Re);
            //string fftimg = "./fft1.bmp";
            //debug->img_write(fftimg);

            //normfft<<<(Nthread + BS - 1) / BS, BS >>>(devpad, 2 * SX, 2 * SY);
            
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Ha, SX2 * SY2);



            ifft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            //deviceinへ0elim
            elimpad<<<grid2, block >>>(dvbffc, SX, SY, dvbffcpd, 2 * SX, 2 * SY);

            ////デバッグ
            //cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(Complex, host, SX * SY);
            //Complex->power(Complex->Re);
            //My_Bmp* debug2;
            //debug2 = new My_Bmp(SX, SY);
            //debug2->data_to_ucimg(Complex->Re);
            //string one = "./kaku1-1.bmp";
            //debug2->img_write(one);
            ////デバッグ
            //cudaMemcpy(deb, devpad, sizeof(cufftComplex) * SX * SY * 4, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(de, deb, SX * SY * 4);
            //My_ComArray_2D H(4 * SX * SY, 2 * SX, 2 * SY);
            //H.H_kaku((double)lamda, (double)a, (double)d);
            //H.mul_complex(de);
            //set_cufftcomplex(deb, H.Re, H.Im, 4 * SX * SY);
            //cudaMemcpy(mul, deb, sizeof(cufftComplex) * 4 * SX * SY, cudaMemcpyHostToDevice);
            //ifft_2D_cuda_dev(SX2, SY2, mul);
            ////deviceinへ0elim
            //elimpad << <grid2, block >> > (dev, SX, SY, mul, 2 * SX, 2 * SY);
            ////デバッグ
            //ifft_2D_cuda_dev(SX2, SY2, devpad);
            //elimpad << <grid2, block >> > (dev, SX, SY, devpad, 2 * SX, 2 * SY);
            ////デバッグ
            //cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(Complex, host, SX * SY);
            //Complex->power(Complex->Re);
            //debug2->data_to_ucimg(Complex->Re);
            //string one2 = "./kaku1-2.bmp";
            //debug2->img_write(one2);

            
            Cmulfft<<<(SX * SY + BS - 1) / BS, BS >>>(dvbffc, dvbffc, Ldev, SX * SY);

            //角スペクトル
            cudaMemset(dvbffcpd, 0, sizeof(cufftComplex) * 4 * SX * SY);
            pad_cufftcom2cufftcom<<<grid2, block >>>(dvbffcpd, 2 * SX, 2 * SY, dvbffc, SX, SY);
            fft_2D_cuda_dev(2 * SX, 2 * SY, dvbffcpd);
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Hb, SX2 * SY2);
            ifft_2D_cuda_dev(2 * SX, 2 * SY, dvbffcpd);
            elimpad<<<grid2, block >>>(dvbffc, SX, SY, dvbffcpd, 2 * SX, 2 * SY);
            cucompower<<<(SIZE + BS - 1) / BS, BS >>>(dvbfd, dvbffc, SIZE);



            //cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(Complex, host, SX * SY);
            //
            ////振幅計算
            //Complex->power(Complex->Re);
            //if (k == CHECK_NUM - 1) {
            //    My_Bmp* check;
            //    check = new My_Bmp(SX, SY);
            //    check->data_to_ucimg(Complex->Re);
            //    check->img_write(simimg);
            //    delete check;
            //}

            cudaMemcpy(Complex->Re, dvbfd, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);

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
                cout << "-----------------------------------" << k + 1 << "--------------------------------------\n";
                lap = clock();
                cout << setprecision(4) << (double)(lap - start) / CLOCKS_PER_SEC / 60 << "分経過\n\n";

            }
        }
        delete Lens;

        
        cudaFree(dvbffc);
        cudaFree(dvbfd);
        cudaFree(dvbfd2);
        cudaFree(dvbffcpd);
        cudaFree(Ldev);
        cudaFree(Ha);
        cudaFree(Hb);
        //cudaFree(dvbf_cfc_pad);
        //cudaFree(devbuf_cufc_2);
        //cudaFree(devbuf_db);
        //cudaFree(devRe); cudaFree(devIm);
        //cudaFree(Lhost);
        //cudaFree(host);
        //cudaFree(devbuf_cufc);
        //cudaFree(devpad);
        //cudaFree(ReL);
        //cudaFree(ImL);
        //cudaFree(ReHa);
        //cudaFree(ImHa);
        //cudaFree(ReHb);
        //cudaFree(ImHb);
    }

    else {
        cout << "データファイルを開けませんでした\n終了します。";

    }

    return 0;
}