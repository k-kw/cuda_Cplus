#pragma once

//�U���������͎����z��𐳋K����A�ʑ����֕ϊ�
void to_phase(double* Re, double* Im, int size, double* orig);

//�|�C���^�z��̐^�񒆂P������o��
void get_mid_line(double* one_dim, double* two_dim, int Y, int X);

//���f�U���v�Z�֐�
void power_complex(double* pow, int size, double* Re, double* Im);

//���f�U����2��v�Z�֐�
void power_2_complex(double* pow, int size, double* Re, double* Im);

//���f���z���Z�֐�
void mul_complex(int size, double* Re_in1, double* Im_in1, double* Re_in2, double* Im_in2, double* Re_out, double* Im_out);

//�p�X�y�N�g���@��H�𒼐ڌv�Z����֐�
void H_kaku(double* ReH, double* ImH, double lam, double z, double d, int x, int y);

//�p�X�y�N�g���@(���o�̓T�C�Y�͑S��x�Ey�A0���߂���Ă��Ȃ��摜�f�[�^����́A��ԍ���ifft��0pad������菜�������̂��o�́A���̉E��ifft���U���v�Z�����K����0pad�r���������̂��o��)
void kaku(double* Re, double* Im, double* POWER, int x, int y, double lam, double z, double d, double* ReG, double* ImG);

//�p�X�y�N�g���@(���o�̓T�C�Y�͑S��x�Ey�A0���߂���Ă��Ȃ��摜�f�[�^����́A��ԍ���ifft��0pad������菜�������̂��o�́A���̉E��ifft���U���v�Z�����K����0pad�r���������̂��o��)
void kaku_2(double* Re, double* Im, double* POWER, int x, int y, double lam, double z, double d, double* ReG, double* ImG);

//�p�X�y�N�g���@ver3(�U�����o�͂��Ȃ�)
void kaku_3(double* Re, double* Im, int x, int y, double lam, double z, double d, double* ReG, double* ImG);

//�p�X�y�N�g���@ver4(H�z��������Ƃ��Ĉ���)
void kaku_4(double* Re, double* Im, int x, int y, double lam, double d, double* ReG, double* ImG, double* ReH, double* ImH);

////�t���l����܂�h���v�Z
//void fresnel_h(double* Re, double* Im, double lam, double z, double d, int x, int y);