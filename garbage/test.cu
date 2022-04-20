#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>

#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//#include "my_all.h"
#include "Bmp_class_dll.h"
#include "complex_array_class_dll.h"

#include <opencv2//opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

#ifndef __CUDACC__
#define __CUDACC__
#endif 

#define sqr(x) ((x)*(x))
#define SX 512
#define SY 512


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

__global__ void assign2D(int* d_a, int w, int h, int value)
{
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	d_a[idy * w + idx] = value;
}

int main() {
	int w = 1024, h = 1024;
	int* h_a;
	h_a = new int[w * h];

	int* d_a;

	cudaMalloc((void**)&d_a, sizeof(int) * w * h);
	cudaMemcpy(d_a, h_a, sizeof(int) * w * h, cudaMemcpyHostToDevice);

	assign2D << <dim3(64, 64), dim3(16, 16) >> > (d_a, w, h, 5);

	cudaMemcpy(h_a, d_a, sizeof(int) * w * h, cudaMemcpyDeviceToHost);

	cout << h_a[0] << "\n" << h_a[1];


	delete[]h_a;
	cudaFree(d_a);

	return 0;


}