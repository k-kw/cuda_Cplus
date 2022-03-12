#pragma once

#ifdef DLLCOMARRAY_EXPORTS
#define DLLCOMARRAY_API __declspec(dllexport)
#else
#define DLLCOMARRAY_API __declspec(dllimport)
#endif

//FFT�N���X
class DLLCOMARRAY_API My_Fft
{
private:
	int x;
	int y;
public:
	double* Re_in, * Im_in, * Re_out, * Im_out;

	My_Fft(int sx, int sy);  //�R���X�g���N�^

	//interface
	void data_to_in(double* Re, double* Im);  //double�f�[�^��Re_in��Im_in�Ɋi�[
	void out_to_data(double* Re, double* Im); //Re_out��Im_out��double�Ɋi�[

	//FFT
	void fft2d();  //2����FFT
	void ifft2d(); //2����IFFT

	~My_Fft(); //�f�X�g���N�^
};




//���f�z��N���X
class DLLCOMARRAY_API My_Complex_Array
{
private:
	
public:
	//�T�C�Y
	int s;
	//�z��|�C���^
	double* Re;
	double* Im;

	//�R���X�g���N�^
	My_Complex_Array(int s);
	//�f�X�g���N�^
	~My_Complex_Array();

	//�z��Ɋi�[
	//���d��`
	void data_to_ReIm(double* Rein, double* Imin);
	void data_to_ReIm(int* Rein, int* Imin);
	void data_to_ReIm(unsigned char* Rein, unsigned char* Imin);
	void data_to_ReIm(double* Rein);
	void data_to_ReIm(int* Rein);
	void data_to_ReIm(unsigned char* Rein);


	//�U���o��
	void power(double* pow);
	
	//���f��Z���d��`
	//�i�[����Ă��镡�f�z��Ǝw�肵�����f�z��I�u�W�F�N�g�̏�Z���ʂ��A�ʂ̕��f�z��Ɋi�[
	void mul_complex(My_Complex_Array* opponent, My_Complex_Array* out);

	//�i�[����Ă��镡�f�z��Ǝw�肵�����f�z��I�u�W�F�N�g�̏�Z���ʂ��A���̃I�u�W�F�N�g�Ɋi�[
	void mul_complex(My_Complex_Array* opponent);

	//�i�[����Ă��镡�f�z��Ǝw�肵�����f�z��̏�Z���ʂ��A���̃I�u�W�F�N�g�Ɋi�[
	void mul_complex(double* Re2, double* Im2);


	//�U��(����)���in�𐳋K����A�ʑ����ɕϊ�
	void to_phase(double* in);

	//�U��(����)���in��255�Ŋ����āA�ʑ����ɕϊ�
	void to256_phase(double* in);


	//0���߂��Ċg��Ains = inx * iny�Aoutx > inx�Aouty > iny
	void zeropad(My_Complex_Array* out, int outx, int outy, int inx, int iny);

	//���S�����o���ďk���Ains = inx * iny�Aoutx < inx�Aouty < iny
	void extract_center(My_Complex_Array* out, int outx, int outy, int inx, int iny);

};


//���f�z��N���X���p�������Q�������f�z��N���X
class DLLCOMARRAY_API My_ComArray_2D : public My_Complex_Array
{
private:

public:
	//�c�� (s = x * y)
	int x, y;

	//�R���X�g���N�^
	My_ComArray_2D(int s, int x, int y) :My_Complex_Array(s), x(x), y(y) {};

	//�f�X�g���N�^�͊��N���X�̂��̂������Ŏg����

	//0���߂��Ċg��Ains = inx * iny�Aoutx > inx�Aouty > iny
	void zeropad(My_ComArray_2D* out);

	//���S�����o���ďk���Ains = inx * iny�Aoutx < inx�Aouty < iny
	void extract_center(My_ComArray_2D* out);


	//�p�X�y�N�g���@��H���i�[�A�摜�̏c���{�̑傫���ŗp��
	void H_kaku(double lam, double z, double d);

	//���݊i�[���ꂢ�Ă�H���g���āA�w�肵��in�̊p�X�y�N�g���@�����s��A���ʂ�out�Ɋi�[
	//in��out�́AH�̏c�������B���̓f�[�^�����̂܂�in�ɂ����OK�A
	void kaku(My_ComArray_2D* out, My_ComArray_2D* in);


	
	//�����_���g�U��
	void diffuser_Random(int rand_seed);                                               
	

};


//2�������f�z��N���X���p�����������Y�z��N���X
class DLLCOMARRAY_API My_LensArray :public My_ComArray_2D
{
private:

public:
	bool approx;    //�ߎ�
	double f;       //�œ_����
	double lamda;   //�g��
	double d;       //��f�s�b�`


	//�R���X�g���N�^
	My_LensArray(int s, int x, int y, bool approx, double f, double lamda, double d)
		:My_ComArray_2D(s, x, y), approx(approx), f(f), lamda(lamda), d(d) {};

	//�f�X�g���N�^����

	void Lens();                           //�P�ꃌ���Y

	void diffuser_Lensarray(int ls);      //�����Y�A���C�g�U��

};
