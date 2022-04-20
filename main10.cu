#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>

#include "my_all.h"
#include "Bmp_class_dll.h"
#include "complex_array_class_dll.h"
#include "dvcfnc.cuh"

#include <opencv2//opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

//copy
#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//�������C�u�����C���N���[�h
#include <curand.h>
#include <curand_kernel.h>
//copy

//�ǉ��̈ˑ��t�@�C���ݒ�̑���
//opencv��DLL��PATH��ʂ��ē��I�����N���C�u����(�ÖٓI�����N)�Ƃ���
#pragma comment(lib, "opencv_world454.lib")
#pragma comment(lib, "opencv_world454d.lib")


//bmp�N���X�𓮓I�����N(�ÖٓI�����N)
#pragma comment(lib, "Dll_bmp_class.lib")
//���f�z��N���X�𓮓I�����N(�ÖٓI�����N)
#pragma comment(lib, "DllComArray.lib")

//�]���Ȍx���폜
#pragma warning(disable:4996)

using namespace std;
using namespace cv;

//�p�����[�^
#define BX 28       //bindat��
#define BY 28       //bindat�̏c

//SLM�ɍ��킹���ق��������H

//#define SX 4096     //SLM�ł̉���f��(4�Ŋ���鐮���Ɍ���)
//#define SY 2400     //SLM�ł̏c��f��(4�Ŋ���鐮���Ɍ���)
//float d = 3.74e-06;
//#define short 2400    //�Z��

//SLM�𑜓x
#define SLMX 4096     //SLM�ł̉���f��(4�Ŋ���鐮���Ɍ���)
#define SLMY 2400     //SLM�ł̏c��f��(4�Ŋ���鐮���Ɍ���)
#define short 2400    //�Z��

//�V�~�����[�V�����z��T�C�Y
#define SX 8192
#define SY 4800

//SX,SY�̉�f�s�b�`
float d = 1.87e-06;

//0���ߌ�摜�T�C�Y
#define SX2 (2*SX)
#define SY2 (2*SY)
#define SIZE (SX*SY)      //�p�f�B���O�O�T�C�Y
#define PADSIZE (SX2*SY2) //�p�f�B���O��T�C�Y

#define N 6       //�摜�̖���
#define CHECK_NUM N  //�V�~�����[�V�����摜���`�F�b�N����ԍ�

//#define lam 532e-09  //�g��
//#define d 1.496e-05 //��f�s�b�`
//#define a 0.1 //�`������1
//#define b 0.03 //�`������2
//#define f 0.03 //�œ_����

//�g��
float lamda = 532e-09;

//�����Y�g�U�ł̐��@��SLM���猈�߂�
//#define LENS_SIZE 32 //�g�U�����Y�̃����Y�T�C�Y
//1mm(�����Y�p)/d(SLM�s�b�`)=267���
#define LENS_SIZE 512

//�`�������Əœ_����
float a = 0.0066;
//float b = 0.03;
float b = 0.0033;
//float f = 0.001;
//�t���C�A�C�����Y�̃f�[�^�V�[�g���
float f = 0.0033;

////NEW
////SLM�𑜓x�ɑ΂���A�J�����̉𑜓x�̊���
//#define SC 0.5
////�J�����̉𑜓x
//#define CAMX (int)(SX*SC)
//#define CAMY (int)(SY*SC)
////NEW

#define resolution pow(2, 8) //�𑜓x
#define approx false    //�����Y�̎��̋ߎ�
#define sqr(x) ((x)*(x))

//copy
//CUDA
#ifndef __CUDACC__
#define __CUDACC__
#endif 
//copy


//1�����̃O���b�h�ƃu���b�N
//���X���b�h��
// PADSIZE�ɓ���
//#define Nthread SX2*SY2
// 
//�u���b�N���̃X���b�h��1=<BS=<1024
#define BS 1024



//�񎟌��̃O���b�h�ƃu���b�N
//�u���b�N������̃X���b�h���͍��v1024�܂łȂ̂ŁAblock(32,32)��葝�₹�Ȃ�
//grid�͏���Ȃ��H
//SX,SY=512,512
//dim3 grid(32, 32), block(32, 32), grid2(16, 16);


//���̂Ƃ���2�̊K��̐����`�ł����ł��Ȃ�,�f�o�b�O�̕K�v����H
//SX,SY=4096,4096
//dim3 grid(256, 256), block(32, 32), grid2(128, 128);

#define blockx 32
#define blocky 32

dim3 grid((SX2 + blockx - 1) / blockx, (SY2 + blocky - 1) / blocky), block(blockx, blocky), grid2((SX + blockx - 1) / blockx, (SY + blocky - 1) / blocky);




//�e���v���[�g�֐������ʂɂ���Ƃ��܂������Ȃ�
template <class Type>
__global__ void cunormali(Type* devin, Type* devout, Type max, Type min, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        devout[idx] = (devin[idx] - min) / (max - min);

    }
}




//�t�@�C���p�X
string binpath = "../../../../dat/bindat/1byte/fm_28_1.dat";
string simpath = "../../../../dat/simdat/SLM_phase/1byte/lsd/test_sim.dat";
string oriimg = "./test.bmp";
string simimg = "./testsim.bmp";
string t = "exp.bmp";

int main() {
    clock_t start, lap;
    start = clock();

    //�ǂݍ��݃o�C�g�m�F
    int byte_num;
    do {
        cout << "\n�o�C�i���f�[�^��4�o�C�g�œǂݍ��݁F4�����\t1�o�C�g�œǂݍ��݁F1�����\n";
        cout << " 1 or 4: "; cin >> byte_num;

    } while (byte_num != 4 && byte_num != 1);


    //�������݃o�C�g�m�F
    int byte_numw;
    do {
        cout << "\n�o�C�i���f�[�^��4�o�C�g�ŏ������݁F4�����\t1�o�C�g�ŏ������݁F1�����\n";
        cout << " 1 or 4: "; cin >> byte_numw;

    } while (byte_numw != 4 && byte_numw != 1);



    //�摜�f�[�^��U�����(����)�Ƃ��邩�A�ʑ��ɕϊ����邩�m�F
    int ampl_or_phase;
    do {
        cout << "\n�摜�f�[�^��U�����(����)�Ƃ���ꍇ�F0�����\t���K����A�ʑ����Ƃ���ꍇ�F1�����\n";
        cout << " 0 or 1 : "; cin >> ampl_or_phase;

    } while (ampl_or_phase != 0 && ampl_or_phase != 1);



    //�����_���ʑ��������Y�A���C���m�F
    int rand_or_lsd;
    do {
        cout << "\n�g�U�������_���ʑ��Ƃ���ꍇ�F0�����\t�g�U����������Y�A���C�Ƃ���ꍇ�F1�����\n";
        cout << " 0 or 1 : "; cin >> rand_or_lsd;

    } while (rand_or_lsd != 0 && rand_or_lsd != 1);

    //�t�@�C�����́E�o�C�i���X�g���[���I�[�v��
    ifstream ifs(binpath, ios::binary /*| ios::in*/);
    //�t�@�C���o�́E�o�C�i���X�g���[���I�[�v��
    ofstream ofs(simpath, ios::binary/* | ios::out*/);



    //�����I�[�v���ł������m�F
    if ((ifs) && (ofs)) {

        My_LensArray* Lens;
        Lens = new My_LensArray(SX * SY, SX, SY, approx, (double)f, (double)lamda, (double)d);

        if (rand_or_lsd == 0) {
            //�����_���g�U��
            Lens->diffuser_Random(0);

        }
        else {
            //�����Y�A���C�g�U��
            Lens->diffuser_Lensarray(LENS_SIZE);

        }


        //�����Y�̔z����f�o�C�X�֑���
        //double* ReL, * ImL;
        //cudaMalloc((void**)&ReL, sizeof(double) * SX * SY);
        //cudaMalloc((void**)&ImL, sizeof(double) * SX * SY);
        //cudaMemcpy(ReL, Lens->Re, sizeof(double) * SX * SY, cudaMemcpyHostToDevice);
        //cudaMemcpy(ImL, Lens->Im, sizeof(double) * SX * SY, cudaMemcpyHostToDevice);
        //cuComplex* Lhost, * Ldev;
        //cudaMallocHost((void**)&Lhost, sizeof(cuComplex) * SX * SY);
        //cudaMalloc((void**)&Ldev, sizeof(cuComplex) * SX * SY);
        //set_cufftcomplex(Lhost, Lens->Re, Lens->Im, SX * SY);

        //�f�o�C�X�Adouble ������
        double* dvbfd, * dvbfd2;
        cudaMalloc((void**)&dvbfd, sizeof(double) * SIZE);
        cudaMalloc((void**)&dvbfd2, sizeof(double) * SIZE);

        
        cudaMemcpy(dvbfd, Lens->Re, sizeof(cuComplex) * SX * SY, cudaMemcpyHostToDevice);
        cudaMemcpy(dvbfd2, Lens->Im, sizeof(cuComplex) * SX * SY, cudaMemcpyHostToDevice);

        cuComplex* Ldev;
        cudaMalloc((void**)&Ldev, sizeof(cuComplex) * SIZE);
        cusetcufftcomplex<<<(SIZE + BS - 1) / BS, BS >>>(Ldev, dvbfd, dvbfd2, SIZE);

        //cudaMemcpy(Ldev, Lhost, sizeof(cuComplex) * SX * SY, cudaMemcpyHostToDevice);
        //�摜�f�[�^���i�[����host
        //�y�[�W�Œ�ł�OK
        //cufftComplex* host;
        //cudaMallocHost((void**)&host, sizeof(cufftComplex)* SX* SY);
        ////host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
        ////host���R�s�[����f�o�C�X���̃������m��
        //cufftComplex* devbuf_cufc;
        //cudaMalloc((void**)&devbuf_cufc, sizeof(cufftComplex) * SX * SY);


        //�f�o�C�X,cufftComplex������
        cufftComplex* dvbffc;
        cudaMalloc((void**)&dvbffc, sizeof(cufftComplex) * SIZE);


        //�f�o�C�X,cufftComplex,PADSIZE������
        cufftComplex* dvbffcpd;
        cudaMalloc((void**)&dvbffcpd, sizeof(cufftComplex)* PADSIZE);

        //H�z����f�o�C�X���ō쐬
        //float* ReHa, * ImHa;
        //cudaMalloc((void**)&ReHa, sizeof(float) * SX * SY * 4);
        //cudaMalloc((void**)&ImHa, sizeof(float) * SX * SY * 4);
        ////�u���b�N������̃X���b�h���͍��v1024�܂łȂ̂ŁAblock(32,32)��葝�₹�Ȃ�
        ////grid�͏���Ȃ��H
        //Hcudaf_shiftf(ReHa, ImHa, 2 * SX, 2 * SY, d, a, lamda, grid, block);
        ////Hnotgpushift(ReHa, ImHa, 2 * SX, 2 * SY, d, a, lamda, grid, block);
        //float* ReHb, * ImHb;
        //cudaMalloc((void**)&ReHb, sizeof(float) * SX * SY * 4);
        //cudaMalloc((void**)&ImHb, sizeof(float) * SX * SY * 4);
        ////�u���b�N������̃X���b�h���͍��v1024�܂łȂ̂ŁAblock(32,32)��葝�₹�Ȃ�
        ////grid�͏���Ȃ��H
        //Hcudaf_shiftf(ReHb, ImHb, 2 * SX, 2 * SY, d, b, lamda, grid, block);
        ////Hnotgpushift(ReHb, ImHb, 2 * SX, 2 * SY, d, b, lamda, grid, block);


        cuComplex* Ha;
        cudaMalloc((void**)&Ha, sizeof(cuComplex)* SX * SY * 4);
        Hcudashiftcom(Ha, SX2, SY2, a, d, lamda, grid, block);
        cuComplex* Hb;
        cudaMalloc((void**)&Hb, sizeof(cuComplex) * SX * SY * 4);
        Hcudashiftcom(Hb, SX2, SY2, b, d, lamda, grid, block);



        ////�|���Z�̏o�̓������m��
        //cufftComplex* dvbf_cfc_pad;
        //cudaMalloc((void**)&dvbf_cfc_pad, sizeof(cufftComplex) * SX * SY * 4);
        ////�����Y�̊|���Z�o�̓������m��
        //cufftComplex* devbuf_cufc_2;
        //cudaMalloc((void**)&devbuf_cufc_2, sizeof(cufftComplex)* SX* SY);
        ////�U���i�[�z��
        //double* devbuf_db;
        //cudaMalloc((void**)&devbuf_db, sizeof(double) * SIZE);
        //double* devRe;

        


        for (int k = 0; k < N; k++) {
            //�i���󋵕\��
            if (k == 0) {
                cout << "\n\n\n-------------------------------�o�̓t�@�C���쐬��---------------------------------\n\n\n";
            }

            //�o�C�i���ǂݍ��ݔz��|�C���^
            unsigned char* chRe;
            int* intRe;
            chRe = new unsigned char[BX * BY];
            intRe = new int[BX * BY];


            //data�ǂݎ��
            if (byte_num == 1) {
                //1byte�ňꖇ���ǂݍ���
                ifs.read((char*)chRe, sizeof(unsigned char) * BX * BY);
                //�㉺���]
                invert_img<unsigned char>(chRe, chRe, BX, BY);

            }
            else {
                //4byte�ňꖇ���ǂݍ���
                ifs.read((char*)intRe, sizeof(int) * BX * BY);
                //�㉺���]
                invert_img<int>(intRe, intRe, BX, BY);


            }


            //�摜�f�[�^�m�F
            if (k == N - 1) {

                My_Bmp* check;
                check = new My_Bmp(BX, BY);

                if (byte_num == 1) {

                    check->uc_to_img(chRe);
                    check->img_write(oriimg);
                }
                else {
                    check->data_to_ucimg(intRe);
                    check->img_write(oriimg);

                }



                delete check;
            }
            delete[]intRe;

            //�摜�f�[�^���g�傷��Ƃ�CV_8U�ł��
            //�摜�f�[�^��cv::Mat�ɃR�s�[
            Mat bin_mat(BY, BX, CV_8U);
            memcpy(bin_mat.data, chRe, BX * BY * sizeof(unsigned char));
            /*imshow("View", bin_mat);
            waitKey(0);*/
            delete[]chRe;

            //�g��
            Mat bin_mat_res(short, short, CV_8U);
            resize(bin_mat, bin_mat_res, Size(short, short));
            bin_mat.release();
            /*string resizeimg = "resize.bmp";
            imwrite(resizeimg, bin_mat_res);
            imshow("View", bin_mat_res);
            waitKey(0);*/

            //�[�����߂��č��킹��
            Mat bin_mat_pjr(SLMY, SLMX, CV_8U);
            copyMakeBorder(bin_mat_res, bin_mat_pjr, (int)(SLMY - short) / 2, (int)(SLMY - short) / 2, (int)(SLMX - short) / 2, (int)(SLMX - short) / 2, BORDER_CONSTANT, 0);
            bin_mat_res.release();
            /*string padimg = "pad.bmp";
            imwrite(padimg, bin_mat_pjr);
            imshow("View", bin_mat_pjr);
            waitKey(0);*/

            unsigned char* padRe;
            padRe = new unsigned char[SLMX * SLMY];


            //�g�債��cv::Mat��padRe�ɃR�s�[
            memcpy(padRe, bin_mat_pjr.data, SLMX * SLMY * sizeof(unsigned char));
            bin_mat_pjr.release();



            //�摜�f�[�^�m�F
            if (k == N - 1) {

                My_Bmp* check;
                check = new My_Bmp(SLMX, SLMY);

                check->uc_to_img(padRe);
                check->img_write(t);

                delete check;

            }

            My_ComArray_2D* Complex, *tmp;
            Complex = new My_ComArray_2D(SX * SY, SX, SY);
            tmp = new My_ComArray_2D(SLMX * SLMY, SLMX, SLMY);

            tmp->data_to_ReIm(padRe);

            //tmp��Complex�Ɋg��,�i�[



            delete[]tmp;

            
            cudaMemcpy(dvbfd, Complex->Re, sizeof(double) * SIZE, cudaMemcpyHostToDevice);
            cudaMemcpy(dvbfd2, Complex->Im, sizeof(double) * SIZE, cudaMemcpyHostToDevice);

            if (ampl_or_phase == 0) {
                //�U���ϒ�
                cusetcufftcomplex<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbfd, dvbfd2, SIZE);

            }
            else {
                //�ʑ��ϒ�
                double* Remax, * Remin;
                Remax = new double;
                Remin = new double;
                *Remax = get_max<double>(Complex->Re, SIZE);
                *Remin = get_min<double>(Complex->Re, SIZE);
                
                
                cunormali<double><<<(SIZE + BS - 1) / BS, BS >>>(dvbfd, dvbfd2, *Remax, *Remin, SIZE);
                cunormaliphase<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbfd2, SIZE);
                delete Remax; delete Remin;
            }

            delete[]padRe;

            //OLD
            //if (ampl_or_phase == 1) {
            //    //�ʑ����ɂ���
            //    Complex->to_phase(Complex->Re);
            //}
            ////CUDA�ɂ��V�~�����[�V����
            //
            //set_cufftcomplex(host, Complex->Re, Complex->Im, SX * SY);
            //cudaMemcpy(dev, host, sizeof(cufftComplex) * SX * SY, cudaMemcpyHostToDevice);
            //OLD

            //�p�X�y�N�g��
            cudaMemset(dvbffcpd, 0, sizeof(cufftComplex) * 4 * SX * SY);
            pad_cufftcom2cufftcom<<<grid2, block >>>(dvbffcpd, 2 * SX, 2 * SY, dvbffc, SX, SY);

            ////�f�o�b�O
            //cufftComplex* deb;
            //deb = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY * 4);
            //cudaMemcpy(deb, devpad, sizeof(cufftComplex)* SX* SY * 4, cudaMemcpyDeviceToHost);
            //My_ComArray_2D* de;
            //de = new My_ComArray_2D(SX * SY * 4, SX2, SY2);
            //cufftcom2mycom(de, deb, SX* SY * 4);
            //de->power(de->Re);
            //My_Bmp* debug;
            //debug = new My_Bmp(SX2, SY2);
            //debug->data_to_ucimg(de->Re);
            //string debugimg = "./pad.bmp";
            //debug->img_write(debugimg);



            fft_2D_cuda_dev(SX2, SY2, dvbffcpd);

            ////�f�o�b�O
            //cudaMemcpy(deb, devpad, sizeof(cufftComplex)* SX* SY * 4, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(de, deb, SX* SY * 4);
            //de->power(de->Re);
            //debug->data_to_ucimg(de->Re);
            //string fftimg = "./fft1.bmp";
            //debug->img_write(fftimg);

            //normfft<<<(Nthread + BS - 1) / BS, BS >>>(devpad, 2 * SX, 2 * SY);
            
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Ha, SX2 * SY2);



            ifft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            //devicein��0elim
            elimpad<<<grid2, block >>>(dvbffc, SX, SY, dvbffcpd, 2 * SX, 2 * SY);

            ////�f�o�b�O
            //cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(Complex, host, SX * SY);
            //Complex->power(Complex->Re);
            //My_Bmp* debug2;
            //debug2 = new My_Bmp(SX, SY);
            //debug2->data_to_ucimg(Complex->Re);
            //string one = "./kaku1-1.bmp";
            //debug2->img_write(one);
            ////�f�o�b�O
            //cudaMemcpy(deb, devpad, sizeof(cufftComplex) * SX * SY * 4, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(de, deb, SX * SY * 4);
            //My_ComArray_2D H(4 * SX * SY, 2 * SX, 2 * SY);
            //H.H_kaku((double)lamda, (double)a, (double)d);
            //H.mul_complex(de);
            //set_cufftcomplex(deb, H.Re, H.Im, 4 * SX * SY);
            //cudaMemcpy(mul, deb, sizeof(cufftComplex) * 4 * SX * SY, cudaMemcpyHostToDevice);
            //ifft_2D_cuda_dev(SX2, SY2, mul);
            ////devicein��0elim
            //elimpad << <grid2, block >> > (dev, SX, SY, mul, 2 * SX, 2 * SY);
            ////�f�o�b�O
            //ifft_2D_cuda_dev(SX2, SY2, devpad);
            //elimpad << <grid2, block >> > (dev, SX, SY, devpad, 2 * SX, 2 * SY);
            ////�f�o�b�O
            //cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(Complex, host, SX * SY);
            //Complex->power(Complex->Re);
            //debug2->data_to_ucimg(Complex->Re);
            //string one2 = "./kaku1-2.bmp";
            //debug2->img_write(one2);

            
            Cmulfft<<<(SX * SY + BS - 1) / BS, BS >>>(dvbffc, dvbffc, Ldev, SX * SY);

            //�p�X�y�N�g��
            cudaMemset(dvbffcpd, 0, sizeof(cufftComplex) * 4 * SX * SY);
            pad_cufftcom2cufftcom<<<grid2, block >>>(dvbffcpd, 2 * SX, 2 * SY, dvbffc, SX, SY);
            fft_2D_cuda_dev(2 * SX, 2 * SY, dvbffcpd);
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Hb, SX2 * SY2);
            ifft_2D_cuda_dev(2 * SX, 2 * SY, dvbffcpd);
            elimpad<<<grid2, block >>>(dvbffc, SX, SY, dvbffcpd, 2 * SX, 2 * SY);
            cucompower<<<(SIZE + BS - 1) / BS, BS >>>(dvbfd, dvbffc, SIZE);



            //cudaMemcpy(host, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            //cufftcom2mycom(Complex, host, SX * SY);
            //
            ////�U���v�Z
            //Complex->power(Complex->Re);
            //if (k == CHECK_NUM - 1) {
            //    My_Bmp* check;
            //    check = new My_Bmp(SX, SY);
            //    check->data_to_ucimg(Complex->Re);
            //    check->img_write(simimg);
            //    delete check;
            //}

            cudaMemcpy(Complex->Re, dvbfd, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);

            if (k == CHECK_NUM - 1) {

                My_Bmp* check;
                check = new My_Bmp(SX, SY);

                check->data_to_ucimg(Complex->Re);
                check->img_write(simimg);
                delete check;

            }


            double* Pline;
            Pline = new double[SX];

            mid_line<double>(Complex->Re, SX, SY, Pline);
            delete Complex;

            //�������ݔz��
            int* intw;
            unsigned char* chw;
            intw = new int[SX];
            chw = new unsigned char[SX];

            norm_reso_n<double>(Pline, intw, (int)(resolution - 1), SX);
            delete[]Pline;
            to_uch(intw, SX, chw);


            //��������
            if (byte_numw == 1) {
                ofs.write((char*)chw, sizeof(unsigned char) * SX);

            }
            else {
                ofs.write((char*)intw, sizeof(int) * SX);

            }


            delete[]intw;
            delete[]chw;


            if ((k + 1) % 100 == 0) {
                cout << "-----------------------------------" << k + 1 << "--------------------------------------\n";
                lap = clock();
                cout << setprecision(4) << (double)(lap - start) / CLOCKS_PER_SEC / 60 << "���o��\n\n";

            }
        }
        delete Lens;

        
        cudaFree(dvbffc);
        cudaFree(dvbfd);
        cudaFree(dvbfd2);
        cudaFree(dvbffcpd);
        cudaFree(Ldev);
        cudaFree(Ha);
        cudaFree(Hb);
        //cudaFree(dvbf_cfc_pad);
        //cudaFree(devbuf_cufc_2);
        //cudaFree(devbuf_db);
        //cudaFree(devRe); cudaFree(devIm);
        //cudaFree(Lhost);
        //cudaFree(host);
        //cudaFree(devbuf_cufc);
        //cudaFree(devpad);
        //cudaFree(ReL);
        //cudaFree(ImL);
        //cudaFree(ReHa);
        //cudaFree(ImHa);
        //cudaFree(ReHb);
        //cudaFree(ImHb);
    }

    else {
        cout << "�f�[�^�t�@�C�����J���܂���ł���\n�I�����܂��B";

    }

    return 0;
}