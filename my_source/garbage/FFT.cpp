#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <windows.h>
#include <fftw3.h>
#include<stddef.h>
#include "FFT.h"

#pragma comment(lib, "libfftw3-3.lib")
#pragma comment(lib, "libfftw3f-3.lib")
#pragma comment(lib, "libfftw3l-3.lib")
#pragma warning(disable:4996)

//FFT�֐�ver2
//�������̊m�ۂ�(size_t)�ɕϊ�
void fft_2D_ver2(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in) {
	fftw_complex* in, * out;
	fftw_plan p;
	in = (fftw_complex*)fftw_malloc((size_t)y * (size_t)x * sizeof(fftw_complex));
	out = (fftw_complex*)fftw_malloc((size_t)y * (size_t)x * sizeof(fftw_complex));
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			in[i * x + j][0] = Re_in[i * x + j];
			in[i * x + j][1] = Im_in[i * x + j];
		}
	}
	p = fftw_plan_dft_2d(y, x, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Re_out[i * x + j] = out[i * x + j][0];
			Im_out[i * x + j] = out[i * x + j][1];
		}
	}
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
};

//FFT�֐�
void fft_2D(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in) {
	fftw_complex* in, * out;
	fftw_plan p;
	in = (fftw_complex*)fftw_malloc(y * x * sizeof(fftw_complex));
	out = (fftw_complex*)fftw_malloc(y * x * sizeof(fftw_complex));
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			in[i * x + j][0] = Re_in[i * x + j];
			in[i * x + j][1] = Im_in[i * x + j];
		}
	}
	p = fftw_plan_dft_2d(y, x, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Re_out[i * x + j] = out[i * x + j][0];
			Im_out[i * x + j] = out[i * x + j][1];
		}
	}
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
};

//IFFT�֐�
void ifft_2D(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in) {
	fftw_complex* in, * out;
	fftw_plan p;
	in = (fftw_complex*)fftw_malloc((size_t)y * (size_t)x * sizeof(fftw_complex));
	out = (fftw_complex*)fftw_malloc((size_t)y * (size_t)x * sizeof(fftw_complex));
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			in[i * x + j][0] = Re_in[i * x + j];
			in[i * x + j][1] = Im_in[i * x + j];
		}
	}
	p = fftw_plan_dft_2d(y, x, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Re_out[i * x + j] = out[i * x + j][0];
			Im_out[i * x + j] = out[i * x + j][1];
		}
	}
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
};

//2D�摜��0pad�֐�(�c�����ꂼ��Q�{�ɂ���0���߁Ain��out�̓T�C�Y�Ⴄ)
void Opad(double*img_out,int in_x,int in_y,double*img_in) {
	int x, y, X, Y;
	x = in_x;
	y = in_y;
	X = 2 * x;
	Y = 2 * y;

	double* img_tmp;
	img_tmp = new double[X * Y];

	for (int i = 0; i < X * Y; i++) {
		img_tmp[i] = 0;
	}

	//���͂��ꂽ�摜�f�[�^���O���߂��Ĕ{�̑傫���̉摜�ɂ���
	for (int i = Y / 4; i < y + Y / 4; i++) {
		for (int j = X / 4; j < x + X / 4; j++) {
			img_tmp[i * X + j] = img_in[(i - Y / 4) * x + (j - X / 4)];
		}
	}

	for (int i = 0; i < X * Y; i++) {
		img_out[i] = img_tmp[i];
	}

	delete[]img_tmp;

}

//2D�摜��0pad��������菜���֐�(�c�����ꂼ��1/2�{�ɂ��Đ^�񒆂��擾�Ain��out�̓T�C�Y�Ⴄ)
void elim_0(double* img_out, int in_x, int in_y, double* img_in) {
	int x, y, X, Y;
	X = in_x;
	Y = in_y;
	x = X/2;
	y = Y/2;

	double* tmp;
	tmp = new double[x * y];
	for (int i = 0; i < y * x; i++) {
		tmp[i] = 0;
	}

	for (int i = Y / 4; i < y + Y / 4; i++) {
		for (int j = X / 4; j < x + X / 4; j++) {
			tmp[(i - Y / 4) * x + (j - X / 4)] = img_in[i * X + j];
		}
	}

	for (int i = 0; i < x * y; i++) {
		img_out[i] = tmp[i];
	}

	delete[]tmp;
}