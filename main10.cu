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
#define SLMSIZE (SLMX*SLMY)

//シミュレーション配列サイズ
#define SX 8192
#define SY 4800
#define SIZE (SX*SY)      //パディング前サイズ

//SX,SYの画素ピッチ
float d = 1.87e-06;

//0埋め後画像サイズ
#define SX2 (2*SX)
#define SY2 (2*SY)
#define PADSIZE (SX2*SY2) //パディング後サイズ

#define N 70000       //画像の枚数
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
float a = 0.04;
//float b = 0.03;
float b = 0.04;
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

//shared memoryは1ブロックに16KB, floatなら4096個, doubleならその半分


//テンプレート関数だけ別にするとうまくいかない
//テンプレート関数の型まで明示したインスタンスを定義のファイルにおいておけばOK？

//use
template <class Type>
__global__ void cunormali(Type* devin, Type* devout, Type max, Type min, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        devout[idx] = (devin[idx] - min) / (max - min);

    }
}

bool samevalue_sclup(My_ComArray_2D *out, My_ComArray_2D *in) {
    int xml, yml, inx, iny, outx, outy;
    inx = in->x;
    iny = in->y;
    outx = out->x;
    outy = out->y;

    if ((outx <= inx) || (outy <= iny)) {
        cout << "出力配列の幅と高さはいずれも入力より大きくしてください" << endl;
        return false;
    }

    
    xml = (outx + inx - 1) / inx;
    yml = (outy + iny - 1) / iny;
    

    //cout << xml << yml << endl;

    for (int i = 0; i < outy; i++) {
        for (int j = 0; j < outx; j++) {
            out->Re[i * outx + j] = in->Re[(int)(i / yml) * inx + (int)(j / xml)];
            out->Im[i * outx + j] = in->Im[(int)(i / yml) * inx + (int)(j / xml)];

        }
    }
    return true;
}

//
//__global__ void samevl_sclup_cuda(double* out, int outx, int outy, double* in, int inx, int iny) {
//
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    int idy = blockDim.y * blockIdx.y + threadIdx.y;
//
//    int xml, yml, tmpy, tmpx;
//    
//    xml = (outx + inx - 1) / inx;
//    yml = (outy + iny - 1) / iny;
//    tmpy = (int)idy / yml;
//    tmpx = (int)idx / xml;
//
//    if (idx < outx && idy < outy) {
//        out[idy * outx + idx] = in[tmpy * inx + tmpx];
//
//    }
//}

//テンプレート
template <class Type>
__global__ void samevl_sclup_cuda_anytype2double(double* out, int outx, int outy, Type* in, int inx, int iny) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int xml, yml, tmpy, tmpx;

    xml = (outx + inx - 1) / inx;
    yml = (outy + iny - 1) / iny;
    tmpy = (int)idy / yml;
    tmpx = (int)idx / xml;

    if (idx < outx && idy < outy) {
        out[idy * outx + idx] = (double)in[tmpy * inx + tmpx];

    }
}

//
//__global__ void samevl_sclup_cuda_uc2double(double* out, int outx, int outy, unsigned char* in, int inx, int iny) {
//
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    int idy = blockDim.y * blockIdx.y + threadIdx.y;
//
//    int xml, yml, tmpy, tmpx;
//
//    xml = (outx + inx - 1) / inx;
//    yml = (outy + iny - 1) / iny;
//    tmpy = (int)idy / yml;
//    tmpx = (int)idx / xml;
//
//    if (idx < outx && idy < outy) {
//        out[idy * outx + idx] = (double)in[tmpy * inx + tmpx];
//
//    }
//}

//sx:lxとsy:lyが同じ比率に限る
void sum_scldown(double* out, int sx, int sy, double* in, int lx, int ly) {
    int mul;
    mul = (lx + sx - 1) / sx;

    //初期化
    for (int i = 0; i < sy; i++) {
        for (int j = 0; j < sx; j++) {
            out[i * sx + j] = 0;
        }
    }
    
    for (int i = 0; i < ly; i++) {
        for (int j = 0; j < lx; j++) {
            out[(i / mul) * sx + (j / mul)] += in[i * lx + j];
        }
    }
}

//CUDAでやると出力がおかしい
//出力メモリはcudaMemsetで０にしておくべき
__global__ void sum_scldwn_cuda(double* out, int sx, int sy, double* in, int lx, int ly) {
    int mul;
    mul = (lx + sx - 1) / sx;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    //int tmpx, tmpy;

    //__shared__ double tmpsum;

    if (idx < lx && idy < ly) {
        /*tmpx = idx / mul;
        tmpy = idy / mul;*/
        out[(idy / mul) * sx + (idx / mul)] += in[idy * lx + idx];
    }

}


//ファイルパス
string binpath = "../../../../dat/bindat/1byte/fm_28_1.dat";
string simpath = "../../../../dat/simdat/SLM_phase/1byte/lsd/fm/fm_4cm_4cm_3.3mm_notpad_4096x2400_samevalue_sclup_to8192x4800_sim.dat";
string oriimg = "./test.bmp";
string simimg = "./testsim_last.bmp";
string scaledown = "./scdwn_last.bmp";
string oriimgexp = "exp.bmp";
string debug_bflens = "bfrlens.bmp";


int main() {
    clock_t start, lap;
    start = clock();

    //画像データをそのままリサイズするか、正方形のまま拡大後横にパディングするか
    int cf_pad;
    do {
        cout << "\n正方形画像データをそのまま長方形に拡大する場合：0を入力\n正方形で拡大後パディングして長方形とする場合：1を入力\n";
        cout << " 0 or 1 : "; cin >> cf_pad;

    } while (cf_pad != 0 && cf_pad != 1);



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
        My_ComArray_2D* Lenspad;
        Lens = new My_LensArray(SIZE, SX, SY, approx, (double)f, (double)lamda, (double)d);
        Lenspad = new My_ComArray_2D(PADSIZE, SX2, SY2);

        if (rand_or_lsd == 0) {
            //ランダム拡散板
            Lens->diffuser_Random(0);

        }
        else {
            //レンズアレイ拡散板
            Lens->diffuser_Lensarray(LENS_SIZE);

        }
        Lens->zeropad(Lenspad);
        delete Lens;

        //LENS
        double* dvbfdpd, * dvbfdpd2;
        cudaMalloc((void**)&dvbfdpd, sizeof(double) * PADSIZE);
        cudaMalloc((void**)&dvbfdpd2, sizeof(double) * PADSIZE);
        cudaMemcpy(dvbfdpd, Lenspad->Re, sizeof(double) * PADSIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(dvbfdpd2, Lenspad->Im, sizeof(double) * PADSIZE, cudaMemcpyHostToDevice);
        delete Lenspad;

        cuComplex* Ldev;
        cudaMalloc((void**)&Ldev, sizeof(cuComplex) * PADSIZE);
        cusetcucomplex<<<(PADSIZE + BS - 1) / BS, BS >>>(Ldev, dvbfdpd, dvbfdpd2, PADSIZE);

        cudaFree(dvbfdpd);cudaFree(dvbfdpd2);
        //LENS


        unsigned char* dvbfucq;
        cudaMalloc((void**)&dvbfucq, sizeof(unsigned char) * SLMSIZE);

        //デバイス、double メモリ
        double* dvbfd, * dvbfd2;
        cudaMalloc((void**)&dvbfd, sizeof(double) * SIZE);
        cudaMalloc((void**)&dvbfd2, sizeof(double) * SIZE);

        //デバイス,cufftComplexメモリ
        cufftComplex* dvbffc;
        cudaMalloc((void**)&dvbffc, sizeof(cufftComplex) * SIZE);


        //デバイス,cufftComplex,PADSIZEメモリ
        cufftComplex* dvbffcpd;
        cudaMalloc((void**)&dvbffcpd, sizeof(cufftComplex)* PADSIZE);

        //Hメモリ
        cuComplex* Ha;
        cudaMalloc((void**)&Ha, sizeof(cuComplex) * PADSIZE);
        Hcudashiftcom(Ha, SX2, SY2, a, d, lamda, grid, block);
        cuComplex* Hb;
        cudaMalloc((void**)&Hb, sizeof(cuComplex) * PADSIZE);
        Hcudashiftcom(Hb, SX2, SY2, b, d, lamda, grid, block);


        //ホスト側ページ固定メモリ
        double* hostbfd;
        cudaMallocHost((void**)&hostbfd, sizeof(double) * SIZE);

        unsigned char* hostbfuc;
        cudaMallocHost((void**)&hostbfuc, sizeof(unsigned char) * SLMSIZE);

        
        //ホスト側通常メモリ
        unsigned char* chRe;
        chRe = new unsigned char[BX * BY];

        double* scldwn, * Pline;
        scldwn = new double[SLMSIZE];
        Pline = new double[SLMX];

        int* intw;
        unsigned char* chw;
        intw = new int[SLMX];
        chw = new unsigned char[SLMX];

        for (int k = 0; k < N; k++) {
            //進捗状況表示
            if (k == 0) {
                cout << "\n\n\n-------------------------------出力ファイル作成中---------------------------------\n\n\n";
            }

            //バイナリ読み込み配列ポインタ
            


            //data読み取り
            //1byteで一枚分読み込み
            ifs.read((char*)chRe, sizeof(unsigned char) * BX * BY);
            //上下反転
            invert_img<unsigned char>(chRe, chRe, BX, BY);

            //画像データ確認
            if (k == N - 1) {

                My_Bmp* check;
                check = new My_Bmp(BX, BY);
                check->uc_to_img(chRe);
                check->img_write(oriimg);
                delete check;
            }

            //画像データを拡大するときCV_8Uでやる
            //画像データをcv::Matにコピー
            Mat bin_mat(BY, BX, CV_8U);
            memcpy(bin_mat.data, chRe, BX * BY * sizeof(unsigned char));
            /*imshow("View", bin_mat);
            waitKey(0);*/

            if (cf_pad == 0) {
                //正方形から長方形に直接リサイズ
                Mat bin_mat_res(SLMX, SLMY, CV_8U);
                resize(bin_mat, bin_mat_res, Size(SLMX, SLMY));
                bin_mat.release();

                //拡大したcv::MatをpadReにコピー
                memcpy(hostbfuc, bin_mat_res.data, SLMSIZE * sizeof(unsigned char));
                bin_mat_res.release();
            }
            else {
                //正方形から正方形へ拡大後、横をパディングして長方形に
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

                //拡大したcv::MatをpadReにコピー
                memcpy(hostbfuc, bin_mat_pjr.data, SLMSIZE * sizeof(unsigned char));
                bin_mat_pjr.release();

            }

            //画像データ確認
            if (k == N - 1) {

                My_Bmp* check;
                check = new My_Bmp(SLMX, SLMY);

                check->uc_to_img(hostbfuc);
                check->img_write(oriimgexp);

                delete check;

            }

            cudaMemcpy(dvbfucq, hostbfuc, sizeof(unsigned char) * SLMSIZE, cudaMemcpyHostToDevice);
            
            samevl_sclup_cuda_anytype2double<unsigned char><<<grid2, block >>>(dvbfd, SX, SY, dvbfucq, SLMX, SLMY);
            cudaMemset(dvbfd2, 0, sizeof(double)* SIZE);
            
            if (ampl_or_phase == 0) {
                //振幅変調
                cusetcucomplex<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbfd, dvbfd2, SIZE);

            }
            else {
                //位相変調
                double* Remax, * Remin;
                Remax = new double;
                Remin = new double;
                
                *Remax = (double)get_max<unsigned char>(hostbfuc, SLMSIZE);
                *Remin = (double)get_min<unsigned char>(hostbfuc, SLMSIZE);

                
                cunormali<double><<<(SIZE + BS - 1) / BS, BS >>>(dvbfd, dvbfd2, *Remax, *Remin, SIZE);
                cunormaliphase<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbfd2, SIZE);
                delete Remax; delete Remin;
            }


            //パディング
            cudaMemset(dvbffcpd, 0, sizeof(cufftComplex) * PADSIZE);
            pad_cufftcom2cufftcom<<<grid2, block >>>(dvbffcpd, SX2, SY2, dvbffc, SX, SY);
            //角スペクトル
            fft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Ha, PADSIZE);
            ifft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            normfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, SX2, SY2);

            //デバッグ
            if (k == CHECK_NUM - 1) {
                elimpadcucompower<<<grid2, block >>>(dvbfd, SX, SY, dvbffcpd, SX2, SY2);
                cudaMemcpy(hostbfd, dvbfd, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);
                My_Bmp* check;
                check = new My_Bmp(SX, SY);

                check->data_to_ucimg(hostbfd);
                check->img_write(debug_bflens);
                delete check;

            }
            //レンズをかける
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Ldev, PADSIZE);
            //角スペクトル
            fft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Hb, PADSIZE);
            ifft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            normfft << <(PADSIZE + BS - 1) / BS, BS >> > (dvbffcpd, SX2, SY2);
            //複素振幅出力
            elimpadcucompower<<<grid2, block >>>(dvbfd, SX, SY, dvbffcpd, SX2, SY2);
            //ホストへ
            cudaMemcpy(hostbfd, dvbfd, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);

            if (k == CHECK_NUM - 1) {

                My_Bmp* check;
                check = new My_Bmp(SX, SY);
                check->data_to_ucimg(hostbfd);
                check->img_write(simimg);
                delete check;

            }

            //CPUで出力振幅をカメラの解像度くらいまで落とす
            if ((int)(SX / SLMX) != (int)(SY / SLMY)) {
                //同じ比率でないなら終了
                cout << "SLM解像度とシミュレーション配列は縦横同じ比率にしてください。\n";
                return 0;
            }
            //memset(scldwn, 0, sizeof(double) * SLMSIZE);
            //近くの値を足しながら縮小、(0で初期化こみ)
            sum_scldown(scldwn, SLMX, SLMY, hostbfd, SX, SY);
            //デバッグ
            if (k == CHECK_NUM - 1) {

                My_Bmp* check;
                check = new My_Bmp(SLMX, SLMY);

                check->data_to_ucimg(scldwn);
                check->img_write(scaledown);
                delete check;

            }           
            mid_line<double>(scldwn, SLMX, SLMY, Pline);

            //書き込み配列
            
            /*norm_reso_n<double>(Pline, intw, (int)(resolution - 1), SX);*/
            norm_reso_n<double>(Pline, intw, (int)(resolution - 1), SLMX);
            
            //to_uch(intw, SX, chw);
            to_uch(intw, SLMX, chw);

            //書き込み
            ofs.write((char*)chw, sizeof(unsigned char)* SLMX);

            if ((k + 1) % 100 == 0) {
                cout << "-----------------------------------" << k + 1 << "--------------------------------------\n";
                lap = clock();
                cout << setprecision(4) << (double)(lap - start) / CLOCKS_PER_SEC / 60 << "分経過\n\n";

            }
        }
        //delete[]intRe;
        delete[]chRe;
        //delete tmp;
        //delete Complex;
        delete[]scldwn;
        delete[]Pline;
        delete[]intw;
        delete[]chw;
        //delete[]padRe;
        cudaFree(hostbfd);
        cudaFree(hostbfuc);
        //cudaFree(dvbfdq);
        cudaFree(dvbfucq);
        cudaFree(dvbffc);
        cudaFree(dvbfd);
        cudaFree(dvbfd2);
        cudaFree(dvbffcpd);
        cudaFree(Ldev);
        cudaFree(Ha);
        cudaFree(Hb);
        
    }

    else {
        cout << "データファイルを開けませんでした\n終了します。";

    }

    return 0;
}