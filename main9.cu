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

//パラメータ1
#define BX 28       //bindat横
#define BY 28       //bindatの縦

#define SX 540     //SLMでの横画素数(4で割れる整数に限る)
#define SY 540     //SLMでの縦画素数(4で割れる整数に限る)
//#define PJRSX 500     //SLMでの横画素数(4で割れる整数に限る)
//#define PJRSY 500     //SLMでの縦画素数(4で割れる整数に限る)

#define short 540     //PJRSYとPJRSXの短辺
//#define short 500     //PJRSYとPJRSXの短辺

#define N 70       //画像の枚数
#define LENS_SIZE 60 //拡散板レンズのレンズサイズ
//#define LENS_SIZE 25

#define CHECK_NUM N  //シミュレーション画像をチェックする番号
#define lam 532e-09  //波長
#define d 1.496e-05 //画素ピッチ
//#define d 6e-05
#define a 0.1 //伝搬距離1
#define b 0.03 //伝搬距離2
#define f 0.03 //焦点距離
//#define a 0.2
//#define b 3
//#define f 0.2
#define resolution pow(2, 8) //解像度
#define approx true    //レンズの式の近似

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
        Lens = new My_LensArray(SX * SY, SX, SY, approx, f, lam, d);

        if (rand_or_lsd == 0) {
            //ランダム拡散板
            Lens->diffuser_Random(0);

        }
        else {
            //レンズアレイ拡散板
            Lens->diffuser_Lensarray(LENS_SIZE);


        }


        //H配列直接計算
        //被写体から拡散板
        My_ComArray_2D* Ha, * Hb;
        Ha = new My_ComArray_2D(4 * SX * SY, 2 * SX, 2 * SY);

        Ha->H_kaku(lam, a, d);


        //被写体からセンサ
        Hb = new My_ComArray_2D(4 * SX * SY, 2 * SX, 2 * SY);

        Hb->H_kaku(lam, b, d);



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



            //拡散板までの伝搬計算
            Ha->kaku(Complex, Complex);

            //拡散板X画像
            Complex->mul_complex(Lens);

            //ラインセンサまで伝搬計算
            Hb->kaku(Complex, Complex);



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

        delete Ha;
        delete Hb;

    }

    else {
        cout << "データファイルを開けませんでした\n終了します。";

    }

    return 0;
}