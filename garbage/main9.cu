#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>

#include "my_all.h"
#include "Bmp_class_dll.h"
#include "complex_array_class_dll.h"

#include <opencv2//opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//�������C�u�����C���N���[�h
#include <curand.h>
#include <curand_kernel.h>

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

//�p�����[�^1
#define BX 28       //bindat��
#define BY 28       //bindat�̏c

#define SX 512     //SLM�ł̉���f��(4�Ŋ���鐮���Ɍ���)
#define SY 512     //SLM�ł̏c��f��(4�Ŋ���鐮���Ɍ���)

#define short 512     //PJRSY��PJRSX�̒Z��

#define N 1       //�摜�̖���
#define LENS_SIZE 32 //�g�U�����Y�̃����Y�T�C�Y

#define CHECK_NUM N  //�V�~�����[�V�����摜���`�F�b�N����ԍ�
//#define lam 532e-09  //�g��
//#define d 1.496e-05 //��f�s�b�`
//#define a 0.1 //�`������1
//#define b 0.03 //�`������2
//#define f 0.03 //�œ_����

float lamda = 532e0 - 9;
float d = 1.496e-05;
float a = 0.1;
float b = 0.03;
float f = 0.03;


#define resolution pow(2, 8) //�𑜓x
#define approx true    //�����Y�̎��̋ߎ�


//CUDA
#define sqr(x) ((x)*(x))

#ifndef __CUDACC__
#define __CUDACC__
#endif 

//0���ߌ�摜�T�C�Y
#define SX2 2*SX
#define SY2 2*SY


//���X���b�h��
#define Nthread SX2*SY2

//�u���b�N���̃X���b�h��1=<BS=<1024
#define BS 1024

void set_cufftcomplex(cufftComplex* cuconp, double* Re, double* Im, int size) {
    for (int i = 0; i < size; i++) {
        cuconp[i] = make_cuComplex((float)Re[i], (float)Im[i]);
    }
}


void fft_2D_cuda_dev(int x, int y, cufftComplex* dev)
{
    cufftHandle plan;
    cufftPlan2d(&plan, x, y, CUFFT_C2C);
    cufftExecC2C(plan, dev, dev, CUFFT_FORWARD);
    cufftDestroy(plan);


}


void ifft_2D_cuda_dev(int x, int y, cufftComplex* dev)
{
    cufftHandle plan;
    cufftPlan2d(&plan, x, y, CUFFT_C2C);
    cufftExecC2C(plan, dev, dev, CUFFT_INVERSE);
    cufftDestroy(plan);
}

void cufftcom_to_mycom(My_ComArray_2D* out, cufftComplex* in, int s) {
    for (int i = 0; i < s; i++) {
        out->Re[i] = (double)cuCrealf(in[i]);
        out->Im[i] = (double)cuCimagf(in[i]);

    }
}



__global__ void Hcudaf(float* Re, float* Im, int x, int y, float u, float v, float z, float lam)
{



    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idy < y && idx < x) {
        Re[idy * x + idx] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2))));
        Im[idy * x + idx] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr(u * ((float)idx - x / 2)) - sqr(v * ((float)idy - y / 2))));
    }


}

__global__ void  shiftf(float* ore, float* oim, float* re, float* im, int x, int y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idy < y && idx < x) {

        if (idx < x / 2 && idy < y / 2) {
            ore[idy * x + idx] = re[(idy + y / 2) * x + (idx + x / 2)];
            ore[(idy + y / 2) * x + (idx + x / 2)] = re[idy * x + idx];
            oim[idy * x + idx] = im[(idy + y / 2) * x + (idx + x / 2)];
            oim[(idy + y / 2) * x + (idx + x / 2)] = im[idy * x + idx];
        }
        else if (idx >= x / 2 && idy < y / 2) {
            ore[idy * x + idx] = re[(idy + y / 2) * x + (idx - x / 2)];
            ore[(idy + y / 2) * x + (idx - x / 2)] = re[idy * x + idx];
            oim[idy * x + idx] = im[(idy + y / 2) * x + (idx - x / 2)];
            oim[(idy + y / 2) * x + (idx - x / 2)] = im[idy * x + idx];
        }



    }
}

//floatXcufftCom
__global__ void mulcomcufftcom(cufftComplex* out, float* re, float* im, cufftComplex* in, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        out[idx] = make_cuComplex(re[idx] * cuCrealf(in[idx]) - im[idx] * cuCimagf(in[idx]),
            re[idx] * cuCimagf(in[idx]) + im[idx] * cuCrealf(in[idx]));

    }
}

__global__ void mulcomcufft_mycom(cufftComplex* out, My_LensArray* in2, cufftComplex* in, int s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < s) {

        out[idx] = make_cuComplex((float)in2->Re[idx] * cuCrealf(in[idx]) - (float)in2->Im[idx] * cuCimagf(in[idx]),
            (float)in2->Re[idx] * cuCimagf(in[idx]) + (float)in2->Im[idx] * cuCrealf(in[idx]));

    }
}

__global__ void pad_cufftcom2cufftcom(cufftComplex* out, int lx, int ly, cufftComplex* in, int sx, int sy)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < sx && idy < sy) {
        out[(idy + ly / 4) * lx + (idx + lx / 4)] = in[idy * sx + idx];
    }

    /*for (int i = 0; i < sy; i++) {
        for (int j = 0; j < sx; j++) {
            out[(i + ly / 4) * lx + (j + lx / 4)] = in[i * sx + j];
        }
        
    }*/

}


__global__ void elimpad(cufftComplex* out, int sx, int sy, cufftComplex* in, int lx, int ly)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < sx && idy < sy) {
        out[idy * sx + idx] = in[(idy + ly / 4) * lx + (idx + lx / 4)];
    }
}



void kaku_cuda(cufftComplex* devicein, float* ReHs, float* ImHs, int x, int y, dim3 grid, dim3 block, int mulgrid, int mulblock) {

    cufftComplex* devpad;
    cudaMalloc((void**)&devpad, sizeof(cufftComplex) * 4 * x * y);
    cudaMemset(devpad, 0, sizeof(cufftComplex) * 4 * x * y);

    //�f�o�b�O
    pad_cufftcom2cufftcom << <grid, block >> > (devpad, 2 * x, 2 * y, devicein, x, y);


    //�f�o�b�O
    cufftComplex* debug;
    debug = (cufftComplex*)malloc(sizeof(cufftComplex) * 4 * x * y);
    cudaMemcpy(debug, devpad, sizeof(cufftComplex) * 4 * x * y, cudaMemcpyDeviceToHost);
    My_ComArray_2D* test;
    test = new My_ComArray_2D(4 * x * y, 2 * x, 2 * y);
    cufftcom_to_mycom(test, debug, 4 * x * y);
    My_Bmp* timg;
    timg = new My_Bmp(2 * x, 2 * y);
    timg->data_to_ucimg(test->Re);
    string wp = "./pad.bmp";
    timg->img_write(wp);


    fft_2D_cuda_dev(2 * x, 2 * y, devpad);


    //�|���Z
    cufftComplex* rslt;
    cudaMalloc((void**)&rslt, sizeof(cufftComplex) * x * y * 4);
    mulcomcufftcom << <mulgrid, mulblock >> > (rslt, ReHs, ImHs, devpad, 4 * x * y);


    ifft_2D_cuda_dev(2 * x, 2 * y, rslt);



    //devicein��0elim
    elimpad << <grid, block >> > (devicein, x, y, rslt, 2 * x, 2 * y);



    cudaFree(devpad);
    cudaFree(rslt);

}


void Hcudaf_shiftf(float* devReH, float* devImH, int x, int y, float d, float z, float lamda, dim3 grid, dim3 block) {
    float* ReH, * ImH;
    cudaMalloc((void**)&ReH, sizeof(float) * x * y);
    cudaMalloc((void**)&ImH, sizeof(float) * x * y);

    float u = 1 / (x * d), v = 1 / (y * d);

    Hcudaf << <grid, block >> > (ReH, ImH, x, y, u, v, z, lamda);
    shiftf << <grid, block >> > (devReH, devImH, ReH, ImH, x, y);

    cudaFree(ReH);
    cudaFree(ImH);
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
        Lens = new My_LensArray(SX * SY, SX, SY, approx, f, (double)lamda, d);

        if (rand_or_lsd == 0) {
            //�����_���g�U��
            Lens->diffuser_Random(0);

        }
        else {
            //�����Y�A���C�g�U��
            Lens->diffuser_Lensarray(LENS_SIZE);


        }


        ////H�z�񒼐ڌv�Z
        ////��ʑ̂���g�U��
        //My_ComArray_2D* Ha, * Hb;
        //Ha = new My_ComArray_2D(4 * SX * SY, 2 * SX, 2 * SY);

        //Ha->H_kaku(lam, a, d);


        ////��ʑ̂���Z���T
        //Hb = new My_ComArray_2D(4 * SX * SY, 2 * SX, 2 * SY);

        //Hb->H_kaku(lam, b, d);



        for (int k = 0; k < N; k++) {
            //�i���󋵕\��
            if (k == 0) {
                cout << "\n\n\n-------------------------------simdata�t�@�C���쐬��---------------------------------\n\n\n";
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
            Mat bin_mat_pjr(SY, SX, CV_8U);
            copyMakeBorder(bin_mat_res, bin_mat_pjr, (int)(SY - short) / 2, (int)(SY - short) / 2, (int)(SX - short) / 2, (int)(SX - short) / 2, BORDER_CONSTANT, 0);
            bin_mat_res.release();
            /*string padimg = "pad.bmp";
            imwrite(padimg, bin_mat_pjr);
            imshow("View", bin_mat_pjr);
            waitKey(0);*/

            unsigned char* padRe;
            padRe = new unsigned char[SX * SY];


            //�g�債��cv::Mat��padRe�ɃR�s�[
            memcpy(padRe, bin_mat_pjr.data, SX * SY * sizeof(unsigned char));
            bin_mat_pjr.release();



            //�摜�f�[�^�m�F
            if (k == N - 1) {


                My_Bmp* check;
                check = new My_Bmp(SX, SY);

                check->uc_to_img(padRe);
                check->img_write(t);

                delete check;

            }



            My_ComArray_2D* Complex;
            Complex = new My_ComArray_2D(SX * SY, SX, SY);

            Complex->data_to_ReIm(padRe);



            delete[]padRe;


            if (ampl_or_phase == 1) {
                //�ʑ����ɂ���
                Complex->to_phase(Complex->Re);
            }




            //CUDA�ɂ��V�~�����[�V����
            dim3 grid(32, 32), block(32, 32), grid2(16, 16);

            //�y�[�W�Œ�ł�OK
            cufftComplex* host;
            //cudaMallocHost((void**)&host, sizeof(cufftComplex)* SX* SY);
            host = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
            set_cufftcomplex(host, Complex->Re, Complex->Im, SX* SY);

            cufftComplex* dev;
            cudaMalloc((void**)&dev, sizeof(cufftComplex)* SX* SY);
            cudaMemcpy(dev, host, sizeof(cufftComplex)* SX* SY, cudaMemcpyHostToDevice);

            //�f�o�b�O
            cufftComplex* devpad;
            cudaMalloc((void**)&devpad, sizeof(cufftComplex)* SX* SY * 4);
            pad_cufftcom2cufftcom << <grid2, block >> > (devpad, 2 * SX, 2 * SY, dev, SX, SY);
            cufftComplex* hostde;
            //cudaMallocHost((void**)&host, sizeof(cufftComplex)* SX* SY);
            hostde = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY * 4);
            cudaMemcpy(hostde, devpad, sizeof(cufftComplex)* SX* SY*4, cudaMemcpyDeviceToHost);
            My_ComArray_2D* de;
            de = new My_ComArray_2D(4 * SX * SY, 2 * SX, 2 * SY);
            cufftcom_to_mycom(de, hostde, 4*SX* SY);
            if (k == CHECK_NUM - 1) {
                //�U���v�Z
                de->power(de->Re);
                My_Bmp* check;
                check = new My_Bmp(2*SX, 2*SY);

                check->data_to_ucimg(de->Re);
                string test2 = "padmain.bmp";
                check->img_write(test2);

                delete check;
            }



            //�f�o�b�O
            cufftComplex* debug;
            /*cudaMallocHost((void**)&debug, sizeof(cufftComplex)* SX* SY);
            */
            debug = (cufftComplex*)malloc(sizeof(cufftComplex) * SX * SY);
            //cudaMemcpy(debug, dev, sizeof(cufftComplex)* SX* SY, cudaMemcpyDeviceToHost);
            //cufftcom_to_mycom(Complex, debug, SX* SY);
            //if (k == CHECK_NUM - 1) {
            //    //�U���v�Z
            //    Complex->power(Complex->Re);
            //    My_Bmp* check;
            //    check = new My_Bmp(SX, SY);

            //    check->data_to_ucimg(Complex->Re);
            //    string test = "memcpy.bmp";
            //    check->img_write(test);

            //    delete check;

            //}


            //H���f�o�C�X�Ōv�Z
            float* ReHs, * ImHs;
            cudaMalloc((void**)&ReHs, sizeof(float)* SX* SY * 4);
            cudaMalloc((void**)&ImHs, sizeof(float)* SX* SY * 4);

            //�u���b�N������̃X���b�h���͍��v1024�܂łȂ̂ŁAblock(32,32)��葝�₹�Ȃ�
            //grid�͏���Ȃ��H
            Hcudaf_shiftf(ReHs, ImHs, 2 * SX, 2 * SY, d, a, lamda, grid, block);

            //�p�X�y�N�g��
            kaku_cuda(dev, ReHs, ImHs, SX, SY, grid2, block, (Nthread + BS - 1) / BS, BS);

            //�f�o�b�O
            cudaMemcpy(debug, dev, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);
            cufftcom_to_mycom(Complex, debug, SX * SY);
            if (k == CHECK_NUM - 1) {
                //�U���v�Z
                Complex->power(Complex->Re);
                My_Bmp* check;
                check = new My_Bmp(SX, SY);

                check->data_to_ucimg(Complex->Re);
                string test = "memcpy.bmp";
                check->img_write(test);

                delete check;
            }



            //�����Y���|���Z
            cufftComplex* rslt;
            cudaMalloc((void**)&rslt, sizeof(cufftComplex)* SX* SY);
            mulcomcufft_mycom<<<(512*512 + BS - 1) / BS, BS>>>(rslt, Lens, dev, SX * SY);
            cudaFree(dev);

            Hcudaf_shiftf(ReHs, ImHs, 2 * SX, 2 * SY, d, b, lamda, grid, block);

            //�p�X�y�N�g��
            kaku_cuda(rslt, ReHs, ImHs, SX, SY, grid2, block, (Nthread + BS - 1) / BS, BS);

            cudaMemcpy(host, rslt, sizeof(cufftComplex) * SX * SY, cudaMemcpyDeviceToHost);

            cufftcom_to_mycom(Complex, host, SX * SY);



            ////�g�U�܂ł̓`���v�Z
            //Ha->kaku(Complex, Complex);

            ////�g�U��X�摜
            //Complex->mul_complex(Lens);

            ////���C���Z���T�܂œ`���v�Z
            //Hb->kaku(Complex, Complex);


            //�U���v�Z
            Complex->power(Complex->Re);

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
                cout << k + 1 << "�܂Ŋ���----------------------------------------------\n";
                lap = clock();
                cout << setprecision(4) << (double)(lap - start) / CLOCKS_PER_SEC / 60 << "���o��\n\n";

            }
        }
        delete Lens;

        /*delete Ha;
        delete Hb;*/

    }

    else {
        cout << "�f�[�^�t�@�C�����J���܂���ł���\n�I�����܂��B";

    }

    return 0;
}