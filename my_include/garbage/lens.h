#pragma once

//�����Y�̈ʑ��ϊ���p�ߎ��Ȃ�
void lens_non_approx(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//�����Y�̈ʑ��ϊ��z��쐬�֐�
void lens(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//�V�����h���J�������Y,�c�����̂ݏW��
void cylin_lens_y(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//�V�����h���J�������Y,�������̂ݏW��
void cylin_lens_x(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//�V�����h���J�������Y,�������̂݊g�U
void cylin_lens_x_concave(double* Relens, double* Imlens, int x, int y, double d, double lam, double f);

//�����_���ʑ��g�U�z��֐�
void diffusionplate_random(double* Re, double* Im, int x, int y);

//�����Y�g�U�z��֐�(�����`�̉摜�̂�)
void diffusionplate_lens(double* Re, double* Im, int im_size, int lens_size, double d, double lam, double f, bool approx);