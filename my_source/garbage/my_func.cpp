#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include<stddef.h>
#include "my_func.h"
#include <stdlib.h>
#include <time.h>

#pragma warning(disable:4996)

//���s�҂ɋ������
//cf���ɒl������Aone�͑I�����P�̖��́Athe_other�͂Q�̖��́Avalue�͑I������I�ԂƂ����͂����鐔��
void forced_two_select(int* cf, char one[], char the_other[], int one_value, int other_value) {
    
    do {
        printf("\n%s�F%d�����\t%s�F%d�����\n", one, one_value, the_other, other_value);
        printf("Press 0 or 1 and enterkey : "); (void)scanf("%d", cf);
    } while (*cf != one_value && *cf != other_value);
}

//���s�҂ɋ������
//�ԋp�l�ɒl������Aone�͑I�����P�̖��́Athe_other�͂Q�̖��́Avalue�͑I������I�ԂƂ����͂����鐔��
int forced_two_select_ver2(char title[], char one[], char the_other[], int one_value, int other_value) {
    int res = 0;
    do {
        printf("\n%s\n%s�F%d�����\t%s�F%d�����\n",title, one, one_value, the_other, other_value);
        printf("Press 0 or 1 and enterkey : "); (void)scanf("%d", &res);
    } while (res != one_value && res != other_value);
    return res;
}