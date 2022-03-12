#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include "my_all.h"
#include "Bmp_class_dll.h"
#include "complex_array_class_dll.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif 

#define sqr(x) ((x)*(x))
#define SX 512
#define SY 512

//bmp��lens�N���X�𓮓I�����N(�ÖٓI�����N)
#pragma comment(lib, "Dll_bmp_class.lib")

char imgname[] = "./lena512x512.bmp";
char writename[] = "./restore.bmp";

//���f���z���Z�֐�
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

//2D�摜��0pad�֐�(�c�����ꂼ��Q�{�ɂ���0���߁Ain��out�̓T�C�Y�Ⴄ)
void Opad(double* img_out, int in_x, int in_y, double* img_in) {
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
	x = X / 2;
	y = Y / 2;

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

//�p�X�y�N�g���@��H�𒼐ڌv�Z����֐�
void H_kaku(double* ReH, double* ImH, double lam, double z, double d, int x, int y) {
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];
	double u = 1 / ((double)x * d), v = 1 / ((double)y * d);
	//H�v�Z
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Retmp[i * x + j] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((double)j - x / 2)) - sqr(v * ((double)i - y / 2))));
			Imtmp[i * x + j] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((double)j - x / 2)) - sqr(v * ((double)i - y / 2))));
		}
	}
	//H�V�t�g
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			if (j < x / 2 && i < y / 2) {
				ReH[i * x + j] = Retmp[(i + y / 2) * x + (j + x / 2)];
				ReH[(i + y / 2) * x + (j + x / 2)] = Retmp[i * x + j];
				ImH[i * x + j] = Imtmp[(i + y / 2) * x + (j + x / 2)];
				ImH[(i + y / 2) * x + (j + x / 2)] = Imtmp[i * x + j];
			}
			else if (j >= x / 2 && i < y / 2) {
				ReH[i * x + j] = Retmp[(i + y / 2) * x + (j - x / 2)];
				ReH[(i + y / 2) * x + (j - x / 2)] = Retmp[i * x + j];
				ImH[i * x + j] = Imtmp[(i + y / 2) * x + (j - x / 2)];
				ImH[(i + y / 2) * x + (j - x / 2)] = Imtmp[i * x + j];
			}
		}
	}


	delete[]Retmp;
	delete[]Imtmp;
};


void set_cufftcomplex(cufftComplex* cuconp, double* Re, double* Im, int size) {
	for (int i = 0; i < size; i++) {
		cuconp[i] = make_cuComplex((float)Re[i], (float)Im[i]);
	}
}

void set_Re_cufftcomplex(cufftComplex* cuconp, double* Re, int size) {
	for (int i = 0; i < size; i++) {
		cuconp[i] = make_cuComplex((float)Re[i], 0.0f);
	}
}

//CUDA_FFT2D
void fft_2D_cuda(int x, int y, cufftComplex* host) {
	cufftHandle plan;
	
	cufftComplex* dev;

	cudaMalloc((void**)&dev, sizeof(cufftComplex) * x * y);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * x * y, cudaMemcpyHostToDevice);

	//�t�[���G�ϊ�
	cufftPlan2d(&plan, x, y, CUFFT_C2C);
	cufftExecC2C(plan, dev, dev, CUFFT_FORWARD);

	cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);


	cudaFree(dev);
	cufftDestroy(plan);
}

//CUDA_IFFT2D
void ifft_2D_cuda(int x, int y, cufftComplex* host) {
	cufftHandle plan;

	cufftComplex* dev;

	cudaMalloc((void**)&dev, sizeof(cufftComplex) * x * y);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * x * y, cudaMemcpyHostToDevice);

	//�t�[���G�ϊ�
	cufftPlan2d(&plan, x, y, CUFFT_C2C);
	cufftExecC2C(plan, dev, dev, CUFFT_INVERSE);

	cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);

	cudaFree(dev);
	cufftDestroy(plan);
}





//CUDA_�p�X�y�N�g���@
void kaku_cuda(double* Re, double* Im, int x, int y, double lam, double d, double* ReG, double* ImG, double* ReH, double* ImH) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//���͂��ꂽ�摜�f�[�^��0���߂��Ĕ{�̑傫���̉摜�ɂ���
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);


	cufftComplex* host;
	host = (cufftComplex*)malloc(sizeof(cufftComplex) * X * Y);

	set_cufftcomplex(host, ReGtmp, ImGtmp, X * Y);


	//G��fft
	//fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);
	fft_2D_cuda(X, Y, host);

	for (int i = 0; i < X * Y; i++) {
		ReGtmp[i] = (double)cuCrealf(host[i]);
		ImGtmp[i] = (double)cuCimagf(host[i]);

	}


	//GXH���v�Z
	mul_complex(Y * X, ReGtmp, ImGtmp, ReH, ImH, ReGtmp, ImGtmp);


	set_cufftcomplex(host, ReGtmp, ImGtmp, X * Y);

	//GXH��ifft
	ifft_2D_cuda(X, Y, host);


	for (int i = 0; i < X * Y; i++) {
		ReGtmp[i] = (double)cuCrealf(host[i]);
		ImGtmp[i] = (double)cuCimagf(host[i]);

	}


	//�o�͂P
	//0���ߕ������Ȃ�
	double* Retmp, * Imtmp;
	Retmp = new double[x * y];
	Imtmp = new double[x * y];

	elim_0(Retmp, X, Y, ReGtmp);
	elim_0(Imtmp, X, Y, ImGtmp);

	//�o�͂P�����o��
	for (int i = 0; i < x * y; i++) {
		Re[i] = Retmp[i];
		Im[i] = Imtmp[i];
	}

	delete[]ReGtmp;
	delete[]ImGtmp;
	delete[]Retmp;
	delete[]Imtmp;

	free(host);
}


int main() {
	//cufftHandle plan;
	//cufftHandle plan2;
	cufftComplex* host;
	//cufftComplex* dev;
	
	host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
    
	double* img;
	img = new double[SX * SY];
	bmp_gray_256_read(img, SX, SY, imgname);

	/*for (int i = 0; i < SX * SY; i++) {
		host[i] = make_cuComplex((float)img[i], 0.0f);
	}*/

	delete[]img;

	/*cudaMalloc((void**)&dev, sizeof(cufftComplex) * SX * SY);
	cudaMemcpy(dev, host, sizeof(cufftComplex) * SX * SY, cudaMemcpyHostToDevice);*/

	////�t�[���G�ϊ�
	//cufftPlan2d(&plan, SX, SY, CUFFT_C2C);
	//cufftExecC2C(plan, dev, dev, CUFFT_FORWARD);

	//�t�t�[���G�ϊ�
	//cufftPlan2d(&plan2, SX, SY, CUFFT_C2C);
	//cufftExecC2C(plan2, dev, dev, CUFFT_INVERSE);

	/*cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);*/

	//cufftDestroy(plan2);

	double* write;
	write = new double[SX * SY];

	for (int i = 0; i < SX * SY; i++) {
		write[i] = (double)sqrt(sqr(cuCrealf(host[i])) + sqr(cuCimagf(host[i])));
	}

	normali_2(write, SX * SY, write);

	bmp_gray_256_write(writename, SX, SY, write);

	delete[]write;

	//cudaFree(dev);
	free(host);
}