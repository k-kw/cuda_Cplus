#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include<stddef.h>
#include "my_func.h"
#include <stdlib.h>
#include <time.h>

#pragma warning(disable:4996)

//実行者に強制二択
//cfがに値が入る、oneは選択肢１の名称、the_otherは２の名称、valueは選択肢を選ぶとき入力させる数字
void forced_two_select(int* cf, char one[], char the_other[], int one_value, int other_value) {
    
    do {
        printf("\n%s：%dを入力\t%s：%dを入力\n", one, one_value, the_other, other_value);
        printf("Press 0 or 1 and enterkey : "); (void)scanf("%d", cf);
    } while (*cf != one_value && *cf != other_value);
}

//実行者に強制二択
//返却値に値が入る、oneは選択肢１の名称、the_otherは２の名称、valueは選択肢を選ぶとき入力させる数字
int forced_two_select_ver2(char title[], char one[], char the_other[], int one_value, int other_value) {
    int res = 0;
    do {
        printf("\n%s\n%s：%dを入力\t%s：%dを入力\n",title, one, one_value, the_other, other_value);
        printf("Press 0 or 1 and enterkey : "); (void)scanf("%d", &res);
    } while (res != one_value && res != other_value);
    return res;
}