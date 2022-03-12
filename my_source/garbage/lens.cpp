#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <windows.h>
#include <fftw3.h>
#include<stddef.h>
#include "lens.h"
#include <stdlib.h>
#include <time.h>

#pragma warning(disable:4996)
#define sqr(x) ((x)*(x))

//レンズの位相変換作用近似なし
void lens_non_approx(double* Relens, double* Imlens, int x, int y, double d, double lam, double f) {
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];


	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			double dx, dy;
			dx = ((double)j - (x / 2)) * d;
			dy = ((double)i - (y / 2)) * d;

			Retmp[i * x + j] = cos((-2 * M_PI / lam) * sqrt(sqr(dx) + sqr(dy) + sqr(f)));
			Imtmp[i * x + j] = sin((-2 * M_PI / lam) * sqrt(sqr(dx) + sqr(dy) + sqr(f)));
		}
	}

	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Relens[i * x + j] = Retmp[i * x + j];
			Imlens[i * x + j] = Imtmp[i * x + j];
		}
	}
	delete[]Retmp;
	delete[]Imtmp;
}

//レンズの位相変換配列作成関数
void lens(double* Relens,double* Imlens,int x,int y,double d,double lam,double f) {
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];


	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			double dx, dy;
			dx = ((double)j - (x / 2)) * d;
			dy = ((double)i - (y / 2)) * d;

			Retmp[i * x + j] = cos((-2 * M_PI / lam) * (sqr(dx) + sqr(dy)) / (2 * f));
			Imtmp[i * x + j] = sin((-2 * M_PI / lam) * (sqr(dx) + sqr(dy)) / (2 * f));
		}
	}

	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Relens[i * x + j] = Retmp[i * x + j];
			Imlens[i * x + j] = Imtmp[i * x + j];
		}
	}
	delete[]Retmp;
	delete[]Imtmp;
}

//シリンドリカルレンズ,縦方向のみ集光
void cylin_lens_y(double* Relens, double* Imlens, int x, int y, double d, double lam, double f) {
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];


	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			double  dy;
			
			dy = ((double)i - (double)(y / 2)) * d;

			Retmp[i * x + j] = cos((-2 * M_PI / lam) * (sqr(dy)) / (2 * f));
			Imtmp[i * x + j] = sin((-2 * M_PI / lam) * (sqr(dy)) / (2 * f));
		}
	}
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Relens[i * x + j] = Retmp[i * x + j];
			Imlens[i * x + j] = Imtmp[i * x + j];
		}
	}

	delete[]Retmp;
	delete[]Imtmp;
}

//シリンドリカルレンズ,横方向のみ集光
void cylin_lens_x(double* Relens, double* Imlens, int x, int y, double d, double lam, double f) {
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];


	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			double  dx;

			dx = ((double)j - (x / 2)) * d;

			Retmp[i * x + j] = cos((-2 * M_PI / lam) * (sqr(dx)) / (2 * f));
			Imtmp[i * x + j] = sin((-2 * M_PI / lam) * (sqr(dx)) / (2 * f));
		}
	}
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Relens[i * x + j] = Retmp[i * x + j];
			Imlens[i * x + j] = Imtmp[i * x + j];
		}
	}

	delete[]Retmp;
	delete[]Imtmp;
}

//シリンドリカルレンズ,横方向のみ拡散
void cylin_lens_x_concave(double* Relens, double* Imlens, int x, int y, double d, double lam, double f) {
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];


	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			double  dx;

			dx = ((double)j - (x / 2)) * d;

			Retmp[i * x + j] = cos((2 * M_PI / lam) * (sqr(dx)) / (2 * f));
			Imtmp[i * x + j] = sin((2 * M_PI / lam) * (sqr(dx)) / (2 * f));
		}
	}
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Relens[i * x + j] = Retmp[i * x + j];
			Imlens[i * x + j] = Imtmp[i * x + j];
		}
	}

	delete[]Retmp;
	delete[]Imtmp;
}

//ランダム位相拡散板配列関数
void diffusionplate_random(double*Re,double*Im,int x,int y) {
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];
	srand(1);

	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			double random;
			random = rand();
			Retmp[i * x + j] = cos(((double)random / RAND_MAX) * 2 * M_PI);
			Imtmp[i * x + j] = sin(((double)random / RAND_MAX) * 2 * M_PI);
		}
	}
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Re[i * x + j] = Retmp[i * x + j];
			Im[i * x + j] = Imtmp[i * x + j];
		}
	}
	delete[]Retmp;
	delete[]Imtmp;
}

//レンズ拡散板配列関数(正方形の画像のみ)
void diffusionplate_lens(double* Re, double* Im, int im_size, int lens_size, double d, double lam, double f, bool approx) {
	int num_lens;
	div_t num_lens_div;

	//商と余りを求める
	num_lens_div = div(im_size, lens_size);
	num_lens = num_lens_div.quot;

	if (num_lens_div.rem > 0) {
		num_lens = num_lens + 1;
	}
	
	int tmp_size;
	tmp_size = num_lens * lens_size;
	double* Retmp, * Imtmp, * Relens, * Imlens;
	Retmp = new double[tmp_size * tmp_size];
	Imtmp = new double[tmp_size * tmp_size];
	Relens = new double[lens_size * lens_size];
	Imlens = new double[lens_size * lens_size];
	
	if (approx) {
		//近似する場合
		lens(Relens, Imlens, lens_size, lens_size, d, lam, f);
	}
	else {
		//近似しないとき
		lens_non_approx(Relens, Imlens, lens_size, lens_size, d, lam, f);
	}


	for (int i = 0; i < tmp_size; i++) {
		for (int j = 0; j < tmp_size; j++) {
			Retmp[i * tmp_size + j] = Relens[(i % lens_size) * lens_size + (j % lens_size)];
			Imtmp[i * tmp_size + j] = Imlens[(i % lens_size) * lens_size + (j % lens_size)];
		}
	}
	delete[]Relens;
	delete[]Imlens;

	for (int i = 0; i < im_size; i++) {
		for (int j = 0; j < im_size; j++) {
			Re[i * im_size + j] = Retmp[i * tmp_size + j];
			Im[i * im_size + j] = Imtmp[i * tmp_size + j];
		}
	}

	delete[]Retmp;
	delete[]Imtmp;
}