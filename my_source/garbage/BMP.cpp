#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <windows.h>
#include <stddef.h>
#include "BMP.h"

#pragma warning(disable:4996)

//bmp�t�@�C���������݊֐��i0�`1�̃C���[�W�f�[�^����́A256�K�����j
void bmp_gray_256_write(char* imgname,int x, int y, double* img) {
	BITMAPFILEHEADER BmpFileHeader;
	BITMAPINFOHEADER BmpInfoHeader;
	RGBQUAD			 RGBQuad[256];
	BmpFileHeader = { 19778, 14 + 40 + 1024 + (DWORD)(x * y), 0, 0, 14 + 40 + 1024 };
	BmpInfoHeader = { 40, x, y, 1, 8, 0L, 0L, 0L, 0L, 0L, 0L };
	for (int i = 0; i < 256; i++) {
		RGBQuad[i].rgbBlue = i;
		RGBQuad[i].rgbGreen = i;
		RGBQuad[i].rgbRed = i;
		RGBQuad[i].rgbReserved = 0;
	}
	unsigned char* tmp;
	tmp = new unsigned char[x * y];
	
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			tmp[i * x + j] = (unsigned char)(img[i * x + j] * 255) ;
		}
	}
	FILE* fp;
	fp = fopen(imgname, "wb");
	fwrite(&BmpFileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
	fwrite(&BmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, fp);
	fwrite(&RGBQuad, sizeof(RGBQuad), 1, fp);
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			fwrite(&tmp[i * x + j], 1, 1, fp);
		}
	}
	fclose(fp);
	delete[] tmp;
	printf("%s �������݊���!\n",imgname);
};

//bmp�t�@�C���ǂݍ��݊֐��i�摜�f�[�^�̂݊i�[�A�J���[��256X�S��z��j
double bmp_gray_256_read(double* img, int x, int y,  char* imgname) {
	BITMAPFILEHEADER BmpFileHeader;
	BITMAPINFOHEADER BmpInfoHeader;


	int color[1024];

	unsigned char* tmp;
	tmp = new unsigned char[x* y];

	FILE* fp;
	fp = fopen(imgname, "rb");

	if (fp == NULL) {
		printf("�t�@�C�����I�[�v���o���܂���ł����B\n");
		return 0;
	}
	else {
		printf("�t�@�C�����I�[�v���o���܂����B�摜�ǂݎ�蒆...\n");
	}

	fread(&BmpFileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
	fread(&BmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, fp);
	fread(&color, 1024, 1, fp);
	for (int i = 0; i < y; i++) {
		for (int j = 0; j <x; j++) {
			fread(&tmp[i * x + j], 1, 1, fp);
			img[i * x + j] = (double)tmp[i *x+ j];
		}
	}
	fclose(fp);

	delete[] tmp;
	printf("%s �ǂݎ�芮���I\n",imgname);
	return img[0];
}

//���K���֐��@
void normali_1(double* out,int size, double* in) {
	double* tmp;
	double max, min;
	tmp = new double[size];
	for (int i = 0; i < size; i++) {
		tmp[i] = in[i];
	}
	max = tmp[0];
	min = tmp[0];
	for (int i = 0; i < size; i++) {
		if (max < tmp[i]) {
			max = tmp[i];
		}
	}
	for (int i = 0; i < size; i++) {
		out[i] = tmp[i] / max;
	}
	delete[]tmp;
};

//���K���֐��A
void normali_2(double* out,int size, double* in) {
	double* tmp, max, min;
	tmp = new double[size];
	for (int i = 0; i < size; i++) {
		tmp[i] = in[i];
	}
	max = tmp[0];
	min = tmp[0];
	for (int i = 0; i < size; i++) {
		if (max < tmp[i]) {
			max = tmp[i];
		}
		else if (min > tmp[i]) {
			min = tmp[i];
		}
	}
	for (int i = 0; i < size; i++) {
		out[i] = (tmp[i] - min) / (max - min);
	}
	delete[]tmp;
};
