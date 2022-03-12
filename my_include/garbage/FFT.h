#pragma once

//FFT関数ver2
void fft_2D_ver2(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in);

//FFT関数
void fft_2D(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in);
//IFFT関数
void ifft_2D(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in);

//2D画像の0pad関数(縦横それぞれ２倍にして0埋め、inとoutはサイズ違う)
void Opad(double* img_out, int in_x, int in_y, double* img_in);
//2D画像の0pad部分を取り除く関数(縦横それぞれ1/2倍にして真ん中を取得、inとoutはサイズ違う)
void elim_0(double* img_out, int in_x, int in_y, double* img_in);

