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

//�U���������͎����z��𐳋K����A�ʑ����֕ϊ�
void to_phase(double* Re, double* Im, int size, double* orig){
	double* tmp;
	tmp = new double[size];

	//���K��
	normali_2(tmp, size, orig);

	//�ʑ�����
	for (int i = 0; i < size; i++) {
		Re[i] = cos(2 * M_PI * tmp[i]);
		Im[i] = sin(2 * M_PI * tmp[i]);

	}

}

//�|�C���^�z��̐^�񒆂P������o��
void get_mid_line(double* one_dim, double* two_dim, int Y, int X) {
	for (int i = 0; i < X; i++) {
		one_dim[i] = two_dim[(Y / 2) * X + i];
	}
}

//���f�U���v�Z�֐�
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

//���f�U����2��v�Z�֐�
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

//�p�X�y�N�g���@��H�𒼐ڌv�Z����֐�
void H_kaku(double* ReH, double* ImH,double lam, double z, double d, int x, int y) {
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

//�p�X�y�N�g���@(���o�̓T�C�Y�͑S��x�Ey�A0���߂���Ă��Ȃ��摜�f�[�^����́A��ԍ���ifft��0pad������菜�������̂��o�́A���̉E��ifft���U���v�Z�����K����0pad�r���������̂��o��(��{�g��Ȃ�))
void kaku(double* Re, double* Im,double *POWER,int x ,int y,double lam,double z,double d, double* ReG, double* ImG) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//���͂��ꂽ�摜�f�[�^���O���߂��Ĕ{�̑傫���̉摜�ɂ���
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);

	//G��fft
	fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//H�𒼐ڌv�Z
	double* ReHtmp, * ImHtmp;
	ReHtmp = new double[X * Y];
	ImHtmp = new double[X * Y];

	H_kaku(ReHtmp, ImHtmp, lam, z, d, X, Y);


	//GXH���v�Z
	mul_complex(Y * X, ReGtmp, ImGtmp, ReHtmp, ImHtmp, ReGtmp, ImGtmp);

	//GXH��ifft
	ifft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);
	
	//�o�͂P
	//0���ߕ������Ȃ�
	double* Retmp, * Imtmp;
	Retmp = new double[x*y];
	Imtmp = new double[x*y];

	elim_0(Retmp, X, Y, ReGtmp);
	elim_0(Imtmp, X, Y, ImGtmp);

	//�o�͂P�����o��
	for (int i = 0; i < x * y; i++) {
		Re[i]=Retmp[i] ;
		Im[i]=Imtmp[i] ;
	}

	//�o��2
	//�U���v�Z
	power_complex(ReGtmp, X * Y, ReGtmp, ImGtmp);
	//���K��
	normali_2(ReGtmp, X * Y, ReGtmp);
	//0���ߕ������Ȃ�
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

//�p�X�y�N�g���@ver2(���o�̓T�C�Y�͑S��x�Ey�A0���߂���Ă��Ȃ��摜�f�[�^����́A��ԍ���ifft��0pad������菜�������̂��o�́A���̉E��ifft��0pad�r�����U���v�Z�����K���o��(��{�g��Ȃ�))
void kaku_2(double* Re, double* Im, double* POWER, int x, int y, double lam, double z, double d, double* ReG, double* ImG) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//���͂��ꂽ�摜�f�[�^���O���߂��Ĕ{�̑傫���̉摜�ɂ���
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);

	//G��fft
	fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//H�𒼐ڌv�Z
	double* ReHtmp, * ImHtmp;
	ReHtmp = new double[X * Y];
	ImHtmp = new double[X * Y];

	H_kaku(ReHtmp, ImHtmp, lam, z, d, X, Y);


	//GXH���v�Z
	mul_complex(Y * X, ReGtmp, ImGtmp, ReHtmp, ImHtmp, ReGtmp, ImGtmp);

	//GXH��ifft
	ifft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

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

	//�o��2
	//�U���v�Z
	power_complex(Retmp, x * y, Retmp, Imtmp);
	//���K��
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

//�p�X�y�N�g���@ver3(�U�����o�͂��Ȃ�)
void kaku_3(double* Re, double* Im, int x, int y, double lam, double z, double d, double* ReG, double* ImG) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//���͂��ꂽ�摜�f�[�^���O���߂��Ĕ{�̑傫���̉摜�ɂ���
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);

	//G��fft
	fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

	//H�𒼐ڌv�Z
	double* ReHtmp, * ImHtmp;
	ReHtmp = new double[X * Y];
	ImHtmp = new double[X * Y];

	H_kaku(ReHtmp, ImHtmp, lam, z, d, X, Y);


	//GXH���v�Z
	mul_complex(Y * X, ReGtmp, ImGtmp, ReHtmp, ImHtmp, ReGtmp, ImGtmp);

	//GXH��ifft
	ifft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

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
	delete[]ReHtmp;
	delete[]ImHtmp;
	delete[]Retmp;
	delete[]Imtmp;
}

//�p�X�y�N�g���@ver4(H�z��������Ƃ��Ĉ���)
void kaku_4(double* Re, double* Im, int x, int y, double lam, double d, double* ReG, double* ImG, double* ReH, double* ImH) {
	int X, Y;
	X = 2 * x;
	Y = 2 * y;

	double* ReGtmp, * ImGtmp;
	ReGtmp = new double[X * Y];
	ImGtmp = new double[X * Y];

	//���͂��ꂽ�摜�f�[�^��0���߂��Ĕ{�̑傫���̉摜�ɂ���
	Opad(ReGtmp, x, y, ReG);
	Opad(ImGtmp, x, y, ImG);

	//G��fft
	//fft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);
	fft_2D_ver2(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);


	//GXH���v�Z
	mul_complex(Y * X, ReGtmp, ImGtmp, ReH, ImH, ReGtmp, ImGtmp);

	//GXH��ifft
	ifft_2D(ReGtmp, ImGtmp, Y, X, ReGtmp, ImGtmp);

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
}

////�t���l����܂�h���v�Z
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
//	//�V�t�g���đ��
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

