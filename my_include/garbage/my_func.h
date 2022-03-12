#pragma once

//実行者に強制二択
//cfがに値が入る、oneは選択肢１の名称、the_otherは２の名称、valueは選択肢を選ぶとき入力させる数字
void forced_two_select(int* cf, char one[], char the_other[], int one_value, int other_value);



//実行者に強制二択
//返却値に値が入る、oneは選択肢１の名称、the_otherは２の名称、valueは選択肢を選ぶとき入力させる数字
int forced_two_select_ver2(char title[] ,char one[], char the_other[], int one_value, int other_value);