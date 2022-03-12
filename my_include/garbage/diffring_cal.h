#pragma once

//振幅もしくは実部配列を正規化後、位相情報へ変換
void to_phase(double* Re, double* Im, int size, double* orig);

//ポインタ配列の真ん中１列を取り出す
void get_mid_line(double* one_dim, double* two_dim, int Y, int X);

//複素振幅計算関数
void power_complex(double* pow, int size, double* Re, double* Im);

//複素振幅の2乗計算関数
void power_2_complex(double* pow, int size, double* Re, double* Im);

//複素数配列乗算関数
void mul_complex(int size, double* Re_in1, double* Im_in1, double* Re_in2, double* Im_in2, double* Re_out, double* Im_out);

//角スペクトル法のHを直接計算する関数
void H_kaku(double* ReH, double* ImH, double lam, double z, double d, int x, int y);

//角スペクトル法(入出力サイズは全てx・y、0埋めされていない画像データを入力、一番左はifft後0pad部を取り除いたものを出力、その右はifft→振幅計算→正規化→0pad排除したものを出力)
void kaku(double* Re, double* Im, double* POWER, int x, int y, double lam, double z, double d, double* ReG, double* ImG);

//角スペクトル法(入出力サイズは全てx・y、0埋めされていない画像データを入力、一番左はifft後0pad部を取り除いたものを出力、その右はifft→振幅計算→正規化→0pad排除したものを出力)
void kaku_2(double* Re, double* Im, double* POWER, int x, int y, double lam, double z, double d, double* ReG, double* ImG);

//角スペクトル法ver3(振幅を出力しない)
void kaku_3(double* Re, double* Im, int x, int y, double lam, double z, double d, double* ReG, double* ImG);

//角スペクトル法ver4(H配列を引数として扱う)
void kaku_4(double* Re, double* Im, int x, int y, double lam, double d, double* ReG, double* ImG, double* ReH, double* ImH);

////フレネル回折のhを計算
//void fresnel_h(double* Re, double* Im, double lam, double z, double d, int x, int y);