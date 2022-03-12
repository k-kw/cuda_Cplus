#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <windows.h>
#include <fftw3.h>
#include <stddef.h>
#include "diffring_cal.h"
#include "FFT.h"
#include "BMP.h"

#pragma warning(disable:4996)

#define sqr(x) ((x)*(x))

//振幅もしくは実部配列を正規化後、位相情報へ変換
void to_phase(double* Re, double* Im, int size, double* orig){
	double* tmp;
	tmp = new double[size];

	//正規化
	normali_2(tmp, size, orig);

	//位相情報へ
	for (int i = 0; i < size; i++) {
		Re[i] = cos(2 * M_PI * tmp[i]);
		Im[i] = sin(2 * M_PI * tmp[i]);

	}

}

//ポインタ配列の真ん中１列を取り出す
void get_mid_line(double* one_dim, double* two_dim, int Y, int X) {
	for (int i = 0; i < X; i++) {
		one_dim[i] = two_dim[(Y / 2) * X + i];
	}
}

//複素振幅計算関数
void power_complex(double* pow, int size, double* Re, double* Im) {
	double* tmp;

	tmp = new double[size];

	for (int i = 0; i < size; i++) {
		tmp[i] = sqrt(sqr(Re[i]) + sqr(Im[i]));
	}

	for (int i = 0; i < size; i++) {
		pow[i] = tmp[i];
	}

	delete[]tmp;
}

//複素振幅の2乗計算関数
void power_2_complex(double* pow, int size, double* Re, double* Im) {
	double* tmp;

	tmp = new double[size];

	for (int i = 0; i < size; i++) {
		tmp[i] = sqr(Re[i]) + sqr(Im[i]);
	}

	for (int i = 0; i < size; i++) {
		pow[i] = tmp[i];
	}

	delete[]tmp;
}

//複素数配列乗算関数
void mul_complex(int size, double* Re_in1, double* Im_in1, double* Re_in2, double* Im_in2, double* Re_out, double* Im_out) {
	double* Retmp, * Imtmp;
	Retmp = new double[size];
	Imtmp = new double[size];

	for (int i = 0; i < size; i++) {
		Retmp[i] = Re_in1[i] * Re_in2[i] - Im_in1[i] * Im_in2[i];
		Imtmp[i] = Re_in1[i] * Im_in2[i] + Im_in1[i] * Re_in2[i];
	}

	for (int i = 0; i < size; i++) {
		Re_out[i] = Retmp[i];
		Im_out[i] = Imtmp[i];
	}
	delete[]Retmp;
	delete[]Imtmp;
};

//角スペクトル法のHを直接計算する関数
void H_kaku(double* ReH, double* ImH,double lam, double z, double d, int x, int y) {
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];
	double u = 1 / ((double)x * d), v = 1 / ((double)y * d);
	//H計算
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Retmp[i * x + j] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((double)j - x / 2)) - sqr(v * ((double)i - y / 2))));
			Imtmp[i * x + j] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((double)j - x / 2)) - sqr(v * ((double)i - y / 2))));
		}
	}
	//Hシフト
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			if (j < x/ 2 && i < y / 2) {
				ReH[i * x + j] = Retmp[(i + y / 2) * x + (j + x/ 2)];
				ReH[(i + y / 2) * x + (j + x / 2)] = Retmp[i * x + j];
				ImH[i * x + j] = Imtmp[(i + y / 2) * x + (j + x / 2)];
				ImH[(i + y / 2) * x + (j + x / 2)] = Imtmp[i * x + j];
			}
			else if (j >= x / 2 && i < y / 2) {
				ReH[i * x + j] = Retmp[(i + y / 2) * x + (j - x / 2)];
				ReH[(i + y/ 2) * x + (j -x / 2)] = Retmp[i * x+ j];
				ImH[i *x + j] = Imtmp[(i + y / 2) * x + (j - x / 2)];
				ImH[(i + y / 2) *x + (j - x / 2)] = Imtmp[i * x+ j];
			}
		}
	}


	delete[]Retmp;
	delete[]Imtmp;
};

//角スペクトル法(入出力サイズは全てx・y、0埋めされていない画像データを入力、一番左はifft後0pad部を取り除いたものを出力、その右はifft→振幅計算→正規化→0pad排除したものを出力(基本使わない))
void kaku(double* Re, double* Im,double *POWER,int x ,int y,double lam,double z,double d, double* ReG, double* ImG) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//入力された画像データを０埋めして倍の大きさの画像にする
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);

	//Gをfft
	fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//Hを直接計算
	double* ReHtmp, * ImHtmp;
	ReHtmp = new double[X * Y];
	ImHtmp = new double[X * Y];

	H_kaku(ReHtmp, ImHtmp, lam, z, d, X, Y);


	//GXHを計算
	mul_complex(Y * X, ReGtmp, ImGtmp, ReHtmp, ImHtmp, ReGtmp, ImGtmp);

	//GXHをifft
	ifft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);
	
	//出力１
	//0埋め部分を省く
	double* Retmp, * Imtmp;
	Retmp = new double[x*y];
	Imtmp = new double[x*y];

	elim_0(Retmp, X, Y, ReGtmp);
	elim_0(Imtmp, X, Y, ImGtmp);

	//出力１書き出し
	for (int i = 0; i < x * y; i++) {
		Re[i]=Retmp[i] ;
		Im[i]=Imtmp[i] ;
	}

	//出力2
	//振幅計算
	power_complex(ReGtmp, X * Y, ReGtmp, ImGtmp);
	//正規化
	normali_2(ReGtmp, X * Y, ReGtmp);
	//0埋め部分を省く
	double* Ptmp;
	Ptmp = new double[x * y];
	elim_0(Ptmp, X, Y, ReGtmp);

	for (int i = 0; i < x*y; i++) {
		POWER[i] = Ptmp[i];
	}

	delete[]ReGtmp;
	delete[]ImGtmp;
	delete[]ReHtmp;
	delete[]ImHtmp;
	delete[]Retmp;
	delete[]Imtmp;
	delete[]Ptmp;
}

//角スペクトル法ver2(入出力サイズは全てx・y、0埋めされていない画像データを入力、一番左はifft後0pad部を取り除いたものを出力、その右はifft→0pad排除→振幅計算→正規化出力(基本使わない))
void kaku_2(double* Re, double* Im, double* POWER, int x, int y, double lam, double z, double d, double* ReG, double* ImG) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//入力された画像データを０埋めして倍の大きさの画像にする
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);

	//Gをfft
	fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//Hを直接計算
	double* ReHtmp, * ImHtmp;
	ReHtmp = new double[X * Y];
	ImHtmp = new double[X * Y];

	H_kaku(ReHtmp, ImHtmp, lam, z, d, X, Y);


	//GXHを計算
	mul_complex(Y * X, ReGtmp, ImGtmp, ReHtmp, ImHtmp, ReGtmp, ImGtmp);

	//GXHをifft
	ifft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//出力１
	//0埋め部分を省く
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];

	elim_0(Retmp, X, Y, ReGtmp);
	elim_0(Imtmp, X, Y, ImGtmp);

	//出力１書き出し
	for (int i = 0; i < x * y; i++) {
		Re[i] = Retmp[i];
		Im[i] = Imtmp[i];
	}

	//出力2
	//振幅計算
	power_complex(Retmp, x * y, Retmp, Imtmp);
	//正規化
	normali_2(Retmp, x * y, Retmp);

	for (int i = 0; i < x * y; i++) {
		POWER[i] = Retmp[i];
	}

	delete[]ReGtmp;
	delete[]ImGtmp;
	delete[]ReHtmp;
	delete[]ImHtmp;
	delete[]Retmp;
	delete[]Imtmp;

}

//角スペクトル法ver3(振幅を出力しない)
void kaku_3(double* Re, double* Im, int x, int y, double lam, double z, double d, double* ReG, double* ImG) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//入力された画像データを０埋めして倍の大きさの画像にする
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);

	//Gをfft
	fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//Hを直接計算
	double* ReHtmp, * ImHtmp;
	ReHtmp = new double[X * Y];
	ImHtmp = new double[X * Y];

	H_kaku(ReHtmp, ImHtmp, lam, z, d, X, Y);


	//GXHを計算
	mul_complex(Y * X, ReGtmp, ImGtmp, ReHtmp, ImHtmp, ReGtmp, ImGtmp);

	//GXHをifft
	ifft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//出力１
	//0埋め部分を省く
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];

	elim_0(Retmp, X, Y, ReGtmp);
	elim_0(Imtmp, X, Y, ImGtmp);

	//出力１書き出し
	for (int i = 0; i < x * y; i++) {
		Re[i] = Retmp[i];
		Im[i] = Imtmp[i];
	}
	

	delete[]ReGtmp;
	delete[]ImGtmp;
	delete[]ReHtmp;
	delete[]ImHtmp;
	delete[]Retmp;
	delete[]Imtmp;
}

//角スペクトル法ver4(H配列を引数として扱う)
void kaku_4(double* Re, double* Im, int x, int y, double lam, double d, double* ReG, double* ImG, double* ReH, double* ImH) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//入力された画像データを0埋めして倍の大きさの画像にする
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);

	//Gをfft
	//fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);
	fft_2D_ver2(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);


	//GXHを計算
	mul_complex(Y * X, ReGtmp, ImGtmp, ReH, ImH, ReGtmp, ImGtmp);

	//GXHをifft
	ifft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//出力１
	//0埋め部分を省く
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];

	elim_0(Retmp, X, Y, ReGtmp);
	elim_0(Imtmp, X, Y, ImGtmp);

	//出力１書き出し
	for (int i = 0; i < x * y; i++) {
		Re[i] = Retmp[i];
		Im[i] = Imtmp[i];
	}

	delete[]ReGtmp;
	delete[]ImGtmp;
	delete[]Retmp;
	delete[]Imtmp;
}

////フレネル回折のhを計算
//void fresnel_h(double*Re,double*Im,double lam,double z,double d,int x,int y) {
//	double* Retmp, * Imtmp;
//	Retmp = new double[x * y];
//	Imtmp = new double[x * y];
//	for (int i = 0; i < y; i++) {
//		for (int j = 0; j < x; j++) {
//			double dx, dy;
//			dx = (j - x / 2) * d;
//			dy = (i - y / 2) * d;
//
//			Retmp[i * x + j] = cos(M_PI * (sqr(dx) + sqr(dy)) / (lam * z));
//			Imtmp[i * x + j] = sin(M_PI * (sqr(dx) + sqr(dy)) / (lam * z));
//		}
//	}
//
//	//シフトして代入
//	for (int i = 0; i < y; i++) {
//		for (int j = 0; j < x; j++) {
//			if (j < x / 2 && i < y / 2) {
//				Re[i * x + j] = Retmp[(i + y / 2) * x + (j + x / 2)];
//				Re[(i + y / 2) * x + (j + x / 2)] = Retmp[i * x + j];
//				Im[i * x + j] = Imtmp[(i + y / 2) * x + (j + x / 2)];
//				Im[(i + y / 2) * x + (j + x / 2)] = Imtmp[i * x + j];
//			}
//			else if (j >= x / 2 && i < y / 2) {
//				Re[i * x + j] = Retmp[(i + y / 2) * x + (j - x / 2)];
//				Re[(i + y / 2) * x + (j - x / 2)] = Retmp[i * x + j];
//				Im[i * x + j] = Imtmp[(i + y / 2) * x + (j - x / 2)];
//				Im[(i + y / 2) * x + (j - x / 2)] = Imtmp[i * x + j];
//			}
//		}
//	}
//	delete[]Retmp;
//	delete[]Imtmp;
//
//}

