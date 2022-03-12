#pragma once

#ifdef DLLBMPCLASS_EXPORTS
#define BMP_CLASS_DLL_API __declspec(dllexport)
#else
#define BMP_CLASS_DLL_API __declspec(dllimport)
#endif

#include <string>

class BMP_CLASS_DLL_API My_Bmp {

private:

	//�C���[�W�T�C�Y
	int im_x;
	int im_y;
	
	
	
public:
	//�C���[�W�f�[�^�̐擪�|�C���^
	unsigned char* img;

	My_Bmp(int sx, int sy);   //�R���X�g���N�^
	void img_read(std::string imgpath);           //BMP�ǂݍ���
	void ucimg_to_double(double* data_out);       //�ǂݍ���unsigned char��double�ɂ��Ċi�[
	
	//�������݂����f�[�^��256�K�����Aunsigned char�ɕϊ���img�Ɋi�[
	//���d��`
	void data_to_ucimg(double* data_in);
	void data_to_ucimg(int* data_in);
	void data_to_ucimg(float* data_in);
	
	void uc_to_img(unsigned char* data_in);       //unsigned char��img�Ɋi�[
	void img_write(std::string imgpath);          //BMP��������
	~My_Bmp();                                    //�f�R���X�g���N�^

};

