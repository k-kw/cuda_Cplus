#pragma once
//bmpファイル書き込み関数（0～1のイメージデータを入力、256階調化）
void bmp_gray_256_write(char* imgname, int x, int y, double* img) ;

//bmpファイル読み込み関数（画像データのみ格納、カラーは256X４を想定）
double bmp_gray_256_read(double* img, int x, int y, char* imgname);

//正規化関数①
void normali_1(double* out, int size, double* in) ;

//正規化関数②
void normali_2(double* out, int size, double* in) ;