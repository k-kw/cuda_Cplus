#pragma once
//bmp�t�@�C���������݊֐��i0�`1�̃C���[�W�f�[�^����́A256�K�����j
void bmp_gray_256_write(char* imgname, int x, int y, double* img) ;

//bmp�t�@�C���ǂݍ��݊֐��i�摜�f�[�^�̂݊i�[�A�J���[��256X�S��z��j
double bmp_gray_256_read(double* img, int x, int y, char* imgname);

//���K���֐��@
void normali_1(double* out, int size, double* in) ;

//���K���֐��A
void normali_2(double* out, int size, double* in) ;