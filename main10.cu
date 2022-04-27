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
#define SLMSIZE (SLMX*SLMY)

//�V�~�����[�V�����z��T�C�Y
#define SX 8192
#define SY 4800
#define SIZE (SX*SY)      //�p�f�B���O�O�T�C�Y

//SX,SY�̉�f�s�b�`
float d = 1.87e-06;

//0���ߌ�摜�T�C�Y
#define SX2 (2*SX)
#define SY2 (2*SY)
#define PADSIZE (SX2*SY2) //�p�f�B���O��T�C�Y

#define N 200       //�摜�̖���
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
float a = 0.04;
//float b = 0.03;
float b = 0.04;
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

//shared memory��1�u���b�N��16KB, float�Ȃ�4096��, double�Ȃ炻�̔���


//�e���v���[�g�֐������ʂɂ���Ƃ��܂������Ȃ�
template <class Type>
__global__ void cunormali(Type* devin, Type* devout, Type max, Type min, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        devout[idx] = (devin[idx] - min) / (max - min);

    }
}

bool samevalue_sclup(My_ComArray_2D *out, My_ComArray_2D *in) {
    int xml, yml, inx, iny, outx, outy;
    inx = in->x;
    iny = in->y;
    outx = out->x;
    outy = out->y;

    if ((outx <= inx) || (outy <= iny)) {
        cout << "�o�͔z��̕��ƍ����͂���������͂��傫�����Ă�������" << endl;
        return false;
    }

    
    xml = (outx + inx - 1) / inx;
    yml = (outy + iny - 1) / iny;
    

    //cout << xml << yml << endl;

    for (int i = 0; i < outy; i++) {
        for (int j = 0; j < outx; j++) {
            out->Re[i * outx + j] = in->Re[(int)(i / yml) * inx + (int)(j / xml)];
            out->Im[i * outx + j] = in->Im[(int)(i / yml) * inx + (int)(j / xml)];

        }
    }
    return true;
}

__global__ void samevl_sclup_cuda(double* out, int outx, int outy, double* in, int inx, int iny) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int xml, yml, tmpy, tmpx;
    
    xml = (outx + inx - 1) / inx;
    yml = (outy + iny - 1) / iny;
    tmpy = (int)idy / yml;
    tmpx = (int)idx / xml;

    if (idx < outx && idy < outy) {
        out[idy * outx + idx] = in[tmpy * inx + tmpx];

    }
}

//�e���v���[�g
template <class Type>
__global__ void samevl_sclup_cuda_anytype2double(double* out, int outx, int outy, Type* in, int inx, int iny) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int xml, yml, tmpy, tmpx;

    xml = (outx + inx - 1) / inx;
    yml = (outy + iny - 1) / iny;
    tmpy = (int)idy / yml;
    tmpx = (int)idx / xml;

    if (idx < outx && idy < outy) {
        out[idy * outx + idx] = (double)in[tmpy * inx + tmpx];

    }
}

__global__ void samevl_sclup_cuda_uc2double(double* out, int outx, int outy, unsigned char* in, int inx, int iny) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int xml, yml, tmpy, tmpx;

    xml = (outx + inx - 1) / inx;
    yml = (outy + iny - 1) / iny;
    tmpy = (int)idy / yml;
    tmpx = (int)idx / xml;

    if (idx < outx && idy < outy) {
        out[idy * outx + idx] = (double)in[tmpy * inx + tmpx];

    }
}

//sx:lx��sy:ly�������䗦�Ɍ���
void sum_scldown(double* out, int sx, int sy, double* in, int lx, int ly) {
    int mul;
    mul = (lx + sx - 1) / sx;

    //������
    for (int i = 0; i < sy; i++) {
        for (int j = 0; j < sx; j++) {
            out[i * sx + j] = 0;
        }
    }
    
    for (int i = 0; i < ly; i++) {
        for (int j = 0; j < lx; j++) {
            out[(i / mul) * sx + (j / mul)] += in[i * lx + j];
        }
    }
}

//CUDA�ł��Əo�͂���������
//�o�̓�������cudaMemset�łO�ɂ��Ă����ׂ�
__global__ void sum_scldwn_cuda(double* out, int sx, int sy, double* in, int lx, int ly) {
    int mul;
    mul = (lx + sx - 1) / sx;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    //int tmpx, tmpy;

    //__shared__ double tmpsum;

    if (idx < lx && idy < ly) {
        /*tmpx = idx / mul;
        tmpy = idy / mul;*/
        out[(idy / mul) * sx + (idx / mul)] += in[idy * lx + idx];
    }

}


//�t�@�C���p�X
string binpath = "../../../../dat/bindat/1byte/m_28_1.dat";
string simpath = "../../../../dat/simdat/SLM_phase/1byte/lsd/test_sim.dat";
string oriimg = "./test.bmp";
string simimg = "./testsim_last2.bmp";
string scaledown = "./scdwn_last2.bmp";
string t = "exp.bmp";

int main() {
    clock_t start, lap;
    start = clock();

    ////�ǂݍ��݃o�C�g�m�F
    //int byte_num;
    //do {
    //    cout << "\n�o�C�i���f�[�^��4�o�C�g�œǂݍ��݁F4�����\t1�o�C�g�œǂݍ��݁F1�����\n";
    //    cout << " 1 or 4: "; cin >> byte_num;
    //} while (byte_num != 4 && byte_num != 1);
    ////�������݃o�C�g�m�F
    //int byte_numw;
    //do {
    //    cout << "\n�o�C�i���f�[�^��4�o�C�g�ŏ������݁F4�����\t1�o�C�g�ŏ������݁F1�����\n";
    //    cout << " 1 or 4: "; cin >> byte_numw;
    //} while (byte_numw != 4 && byte_numw != 1);


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
        My_ComArray_2D* Lenspad;
        Lens = new My_LensArray(SIZE, SX, SY, approx, (double)f, (double)lamda, (double)d);
        Lenspad = new My_ComArray_2D(PADSIZE, SX2, SY2);

        if (rand_or_lsd == 0) {
            //�����_���g�U��
            Lens->diffuser_Random(0);

        }
        else {
            //�����Y�A���C�g�U��
            Lens->diffuser_Lensarray(LENS_SIZE);

        }
        Lens->zeropad(Lenspad);
        delete Lens;

        //LENS
        double* dvbfdpd, * dvbfdpd2;
        cudaMalloc((void**)&dvbfdpd, sizeof(double) * PADSIZE);
        cudaMalloc((void**)&dvbfdpd2, sizeof(double) * PADSIZE);
        cudaMemcpy(dvbfdpd, Lenspad->Re, sizeof(double) * PADSIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(dvbfdpd2, Lenspad->Im, sizeof(double) * PADSIZE, cudaMemcpyHostToDevice);
        delete Lenspad;

        cuComplex* Ldev;
        cudaMalloc((void**)&Ldev, sizeof(cuComplex) * PADSIZE);
        cusetcucomplex<<<(PADSIZE + BS - 1) / BS, BS >>>(Ldev, dvbfdpd, dvbfdpd2, PADSIZE);

        cudaFree(dvbfdpd);cudaFree(dvbfdpd2);
        //LENS


        unsigned char* dvbfucq;
        cudaMalloc((void**)&dvbfucq, sizeof(unsigned char) * SLMSIZE);


        /*double* dvbfdq;
        cudaMalloc((void**)&dvbfdq, sizeof(double) * SLMSIZE);*/
        //cudaMalloc((void**)&dvbfdq2, sizeof(double) * SLMSIZE);

        //�f�o�C�X�Adouble ������
        double* dvbfd, * dvbfd2;
        cudaMalloc((void**)&dvbfd, sizeof(double) * SIZE);
        cudaMalloc((void**)&dvbfd2, sizeof(double) * SIZE);

        //�f�o�C�X,cufftComplex������
        cufftComplex* dvbffc;
        cudaMalloc((void**)&dvbffc, sizeof(cufftComplex) * SIZE);


        //�f�o�C�X,cufftComplex,PADSIZE������
        cufftComplex* dvbffcpd;
        cudaMalloc((void**)&dvbffcpd, sizeof(cufftComplex)* PADSIZE);

        //H������
        cuComplex* Ha;
        cudaMalloc((void**)&Ha, sizeof(cuComplex) * PADSIZE);
        Hcudashiftcom(Ha, SX2, SY2, a, d, lamda, grid, block);
        cuComplex* Hb;
        cudaMalloc((void**)&Hb, sizeof(cuComplex) * PADSIZE);
        Hcudashiftcom(Hb, SX2, SY2, b, d, lamda, grid, block);


        //�z�X�g���y�[�W�Œ胁����
        double* hostbfd;
        cudaMallocHost((void**)&hostbfd, sizeof(double) * SIZE);

        unsigned char* hostbfuc;
        cudaMallocHost((void**)&hostbfuc, sizeof(unsigned char) * SLMSIZE);

        
        //�z�X�g���ʏ탁����
        unsigned char* chRe;
        chRe = new unsigned char[BX * BY];

        double* scldwn, * Pline;
        scldwn = new double[SLMSIZE];
        Pline = new double[SLMX];

        int* intw;
        unsigned char* chw;
        intw = new int[SLMX];
        chw = new unsigned char[SLMX];

        for (int k = 0; k < N; k++) {
            //�i���󋵕\��
            if (k == 0) {
                cout << "\n\n\n-------------------------------�o�̓t�@�C���쐬��---------------------------------\n\n\n";
            }

            //�o�C�i���ǂݍ��ݔz��|�C���^
            


            //data�ǂݎ��
            //1byte�ňꖇ���ǂݍ���
            ifs.read((char*)chRe, sizeof(unsigned char) * BX * BY);
            //�㉺���]
            invert_img<unsigned char>(chRe, chRe, BX, BY);

            //if (byte_num == 1) {
            //    //1byte�ňꖇ���ǂݍ���
            //    ifs.read((char*)chRe, sizeof(unsigned char) * BX * BY);
            //    //�㉺���]
            //    invert_img<unsigned char>(chRe, chRe, BX, BY);
            //}
            //else {
            //    //4byte�ňꖇ���ǂݍ���
            //    ifs.read((char*)intRe, sizeof(int) * BX * BY);
            //    //�㉺���]
            //    invert_img<int>(intRe, intRe, BX, BY);
            //}


            //�摜�f�[�^�m�F
            if (k == N - 1) {

                My_Bmp* check;
                check = new My_Bmp(BX, BY);
                check->uc_to_img(chRe);
                check->img_write(oriimg);
                /*if (byte_num == 1) {
                    check->uc_to_img(chRe);
                    check->img_write(oriimg);
                }
                else {
                    check->data_to_ucimg(intRe);
                    check->img_write(oriimg);
                }*/
                delete check;
            }

            //�摜�f�[�^���g�傷��Ƃ�CV_8U�ł��
            //�摜�f�[�^��cv::Mat�ɃR�s�[
            Mat bin_mat(BY, BX, CV_8U);
            memcpy(bin_mat.data, chRe, BX * BY * sizeof(unsigned char));
            /*imshow("View", bin_mat);
            waitKey(0);*/

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

            //�g�債��cv::Mat��padRe�ɃR�s�[
            memcpy(hostbfuc, bin_mat_pjr.data, SLMSIZE * sizeof(unsigned char));
            bin_mat_pjr.release();

            //�摜�f�[�^�m�F
            if (k == N - 1) {

                My_Bmp* check;
                check = new My_Bmp(SLMX, SLMY);

                check->uc_to_img(hostbfuc);
                check->img_write(t);

                delete check;

            }

            
            //tmp->data_to_ReIm(padRe);

            //NEW
            //cudaMemcpy(dvbfdq, tmp->Re, sizeof(double) * SLMSIZE, cudaMemcpyHostToDevice);
            //cudaMemcpy(dvbfdq2, tmp->Im, sizeof(double) * SLMSIZE, cudaMemcpyHostToDevice);
            cudaMemcpy(dvbfucq, hostbfuc, sizeof(unsigned char) * SLMSIZE, cudaMemcpyHostToDevice);
            
            ////�f�o�b�O
            //My_ComArray_2D* tmp1;
            //tmp1 = new My_ComArray_2D(SLMSIZE, SLMX, SLMY);
            /*cudaMemcpy(tmp1->Re, dvbfdq, sizeof(double) * SLMSIZE, cudaMemcpyDeviceToHost);
            cudaMemcpy(tmp1->Im, dvbfdq2, sizeof(double) * SLMSIZE, cudaMemcpyDeviceToHost);
            if (k == CHECK_NUM - 1) {
                My_Bmp* check;
                check = new My_Bmp(SLMX, SLMY);
                check->data_to_ucimg(tmp1->Re);
                string debug = "debug.bmp";
                check->img_write(debug);
                delete check;

            }*/


            //samevl_sclup_cuda<<<grid2, block >>>(dvbfd, SX, SY, dvbfdq, SLMX, SLMY);
            samevl_sclup_cuda_anytype2double<unsigned char><<<grid2, block >>>(dvbfd, SX, SY, dvbfucq, SLMX, SLMY);
            cudaMemset(dvbfd2, 0, sizeof(double)* SIZE);
            //samevl_sclup_cuda<<<grid2, block >>>(dvbfd2, SX, SY, dvbfdq2, SLMX, SLMY);
            
            //NEW
            /*cudaMemcpy(Complex->Re, dvbfd, sizeof(double)* SIZE, cudaMemcpyDeviceToHost);
            cudaMemcpy(Complex->Im, dvbfd2, sizeof(double)* SIZE, cudaMemcpyDeviceToHost);
            if (k == CHECK_NUM - 1) {
                My_Bmp* check;
                check = new My_Bmp(SX, SY);
                check->data_to_ucimg(Complex ->Re);
                string debug = "debug.bmp";
                check->img_write(debug);
                delete check;

            }*/
            ////�f�o�b�O
            //if (k == CHECK_NUM - 1) {
            //    My_Bmp* check;
            //    check = new My_Bmp(SLMX, SLMY);
            //    check->data_to_ucimg(tmp->Re);
            //    string tmp = "tmp.bmp";
            //    check->img_write(tmp);
            //    delete check;
            //}
            ////tmp��Complex�Ɋg��,�i�[
            //bool cf;
            //cf = samevalue_sclup(Complex, tmp);
            //if (cf == false){
            //    cout << "SLM�̉𑜓x�ɍ��킹���z����A�v�Z����z��̃T�C�Y���������ł��B" << endl;
            //    return 0;
            //
            //}
            ////�f�o�b�O
            //if (k == CHECK_NUM - 1) {
            //    My_Bmp* check;
            //    check = new My_Bmp(SX, SY);
            //    check->data_to_ucimg(Complex->Re);
            //    string upscl = "upscl.bmp";
            //    check->img_write(upscl);
            //    delete check;
            //}
            //
            //cudaMemcpy(dvbfd, Complex->Re, sizeof(double) * SIZE, cudaMemcpyHostToDevice);
            //cudaMemcpy(dvbfd2, Complex->Im, sizeof(double) * SIZE, cudaMemcpyHostToDevice);

            if (ampl_or_phase == 0) {
                //�U���ϒ�
                cusetcucomplex<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbfd, dvbfd2, SIZE);

            }
            else {
                //�ʑ��ϒ�
                double* Remax, * Remin;
                Remax = new double;
                Remin = new double;
                //*Remax = get_max<double>(tmp->Re, SLMSIZE);
                //*Remin = get_min<double>(tmp->Re, SLMSIZE);
                *Remax = (double)get_max<unsigned char>(hostbfuc, SLMSIZE);
                *Remin = (double)get_min<unsigned char>(hostbfuc, SLMSIZE);

                
                cunormali<double><<<(SIZE + BS - 1) / BS, BS >>>(dvbfd, dvbfd2, *Remax, *Remin, SIZE);
                cunormaliphase<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbfd2, SIZE);
                delete Remax; delete Remin;
            }


            //�p�X�y�N�g��
            cudaMemset(dvbffcpd, 0, sizeof(cufftComplex) * PADSIZE);
            pad_cufftcom2cufftcom<<<grid2, block >>>(dvbffcpd, SX2, SY2, dvbffc, SX, SY);
            fft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Ha, PADSIZE);
            ifft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            normfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, SX2, SY2);

            //�f�o�b�O
            if (k == CHECK_NUM - 1) {
                elimpadcucompower<<<grid2, block >>>(dvbfd, SX, SY, dvbffcpd, SX2, SY2);
                cudaMemcpy(hostbfd, dvbfd, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);
                My_Bmp* check;
                check = new My_Bmp(SX, SY);

                check->data_to_ucimg(hostbfd);
                string dbg = "bfrlens.bmp";
                check->img_write(dbg);
                delete check;

            }
            

            //OLD
            ////devicein��0elim
            ////elimpad<<<grid2, block >>>(dvbffc, SX, SY, dvbffcpd, SX2, SY2);
            ////Cmulfft<<<(SIZE + BS - 1) / BS, BS >>>(dvbffc, dvbffc, Ldev, SIZE);
            //elimpad2Cmulfft<<<grid2, block >>>(dvbffc, Ldev, SX, SY, dvbffcpd, SX2, SY2);
            //OLD

            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Ldev, PADSIZE);


            //�p�X�y�N�g��
            
            //OLD
            //cudaMemset(dvbffcpd, 0, sizeof(cufftComplex) * PADSIZE);
            //pad_cufftcom2cufftcom<<<grid2, block >>>(dvbffcpd, SX2, SY2, dvbffc, SX, SY);
            //OLD

            fft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            Cmulfft<<<(PADSIZE + BS - 1) / BS, BS >>>(dvbffcpd, dvbffcpd, Hb, PADSIZE);
            ifft_2D_cuda_dev(SX2, SY2, dvbffcpd);
            normfft << <(PADSIZE + BS - 1) / BS, BS >> > (dvbffcpd, SX2, SY2);

            /*elimpad<<<grid2, block >>>(dvbffc, SX, SY, dvbffcpd, SX2, SY2);
            cucompower<<<(SIZE + BS - 1) / BS, BS >>>(dvbfd, dvbffc, SIZE);*/

            elimpadcucompower<<<grid2, block >>>(dvbfd, SX, SY, dvbffcpd, SX2, SY2);

            ////NEW
            ////�O���{���ő��a���Ƃ��āA�T�C�Y�_�E��
            //double* powdwn;
            //cudaMalloc((void**)&powdwn, sizeof(double) * SLMSIZE);
            //cudaMemset(powdwn, 0, sizeof(double) * SLMSIZE);
            //sum_scldwn_cuda<<<grid2, block >>>(powdwn, SLMX, SLMY, dvbfd, SX, SY);
            //cudaMemcpy(scldwn, powdwn, sizeof(double) * SLMSIZE, cudaMemcpyDeviceToHost);
            ////�f�o�b�O
            //if (k == CHECK_NUM - 1) {
            //    My_Bmp* check;
            //    check = new My_Bmp(SLMX, SLMY);
            //    check->data_to_ucimg(scldwn);
            //    string dwncuda = "scldwncuda.bmp";
            //    check->img_write(dwncuda);
            //    delete check;
            //}
            ////NEW

            cudaMemcpy(hostbfd, dvbfd, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);

            if (k == CHECK_NUM - 1) {

                My_Bmp* check;
                check = new My_Bmp(SX, SY);

                check->data_to_ucimg(hostbfd);
                check->img_write(simimg);
                delete check;

            }

            //CPU�ŏo�͐U�����J�����̉𑜓x���炢�܂ŗ��Ƃ�
            if ((int)(SX / SLMX) != (int)(SY / SLMY)) {
                //�����䗦�łȂ��Ȃ�I��
                cout << "SLM�𑜓x�ƃV�~�����[�V�����z��͏c�������䗦�ɂ��Ă��������B\n";
                return 0;
            }
            memset(scldwn, 0, sizeof(double) * SLMSIZE);
            sum_scldown(scldwn, SLMX, SLMY, hostbfd, SX, SY);
            //�f�o�b�O
            if (k == CHECK_NUM - 1) {

                My_Bmp* check;
                check = new My_Bmp(SLMX, SLMY);

                check->data_to_ucimg(scldwn);
                check->img_write(scaledown);
                delete check;

            }           
            mid_line<double>(scldwn, SLMX, SLMY, Pline);

            //�������ݔz��
            
            /*norm_reso_n<double>(Pline, intw, (int)(resolution - 1), SX);*/
            norm_reso_n<double>(Pline, intw, (int)(resolution - 1), SLMX);
            
            //to_uch(intw, SX, chw);
            to_uch(intw, SLMX, chw);

            //��������
            ofs.write((char*)chw, sizeof(unsigned char)* SLMX);

            //if (byte_numw == 1) {
            //    //ofs.write((char*)chw, sizeof(unsigned char) * SX);
            //    ofs.write((char*)chw, sizeof(unsigned char) * SLMX);
            //}
            //else {
            //    //ofs.write((char*)intw, sizeof(int) * SX);
            //    ofs.write((char*)intw, sizeof(int) * SLMX);
            //}

            

            if ((k + 1) % 100 == 0) {
                cout << "-----------------------------------" << k + 1 << "--------------------------------------\n";
                lap = clock();
                cout << setprecision(4) << (double)(lap - start) / CLOCKS_PER_SEC / 60 << "���o��\n\n";

            }
        }
        //delete[]intRe;
        delete[]chRe;
        //delete tmp;
        //delete Complex;
        delete[]scldwn;
        delete[]Pline;
        delete[]intw;
        delete[]chw;
        //delete[]padRe;
        cudaFree(hostbfd);
        cudaFree(hostbfuc);
        //cudaFree(dvbfdq);
        cudaFree(dvbfucq);
        cudaFree(dvbffc);
        cudaFree(dvbfd);
        cudaFree(dvbfd2);
        cudaFree(dvbffcpd);
        cudaFree(Ldev);
        cudaFree(Ha);
        cudaFree(Hb);
        
    }

    else {
        cout << "�f�[�^�t�@�C�����J���܂���ł���\n�I�����܂��B";

    }

    return 0;
}