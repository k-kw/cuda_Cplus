#pragma once

//レンズの位相変換作用近似なし
void lens_non_approx(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//レンズの位相変換配列作成関数
void lens(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//シリンドリカルレンズ,縦方向のみ集光
void cylin_lens_y(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//シリンドリカルレンズ,横方向のみ集光
void cylin_lens_x(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//シリンドリカルレンズ,横方向のみ拡散
void cylin_lens_x_concave(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//ランダム位相拡散板配列関数
void diffusionplate_random(double* Re, double* Im, int x, int y);

//レンズ拡散板配列関数(正方形の画像のみ)
void diffusionplate_lens(double* Re, double* Im, int im_size, int lens_size, double d, double lam, double f, bool approx);