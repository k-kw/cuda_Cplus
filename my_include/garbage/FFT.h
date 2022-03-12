#pragma once

//FFT�֐�ver2
void fft_2D_ver2(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in);

//FFT�֐�
void fft_2D(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in);
//IFFT�֐�
void ifft_2D(double* Re_out, double* Im_out, int y, int x, double* Re_in, double* Im_in);

//2D�摜��0pad�֐�(�c�����ꂼ��Q�{�ɂ���0���߁Ain��out�̓T�C�Y�Ⴄ)
void Opad(double* img_out, int in_x, int in_y, double* img_in);
//2D�摜��0pad��������菜���֐�(�c�����ꂼ��1/2�{�ɂ��Đ^�񒆂��擾�Ain��out�̓T�C�Y�Ⴄ)
void elim_0(double* img_out, int in_x, int in_y, double* img_in);

