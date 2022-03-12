#pragma once


//ヘッダ内に直接,外部結合を持った関数を定義してはいけない。宣言して、他のソースファイルで定義すべき
//内部結合を持つインライン関数等はヘッダで定義しないといけない。

//inline
//最大値
template <class Type>
inline Type get_max(Type* data, int size)
{
	Type max = data[0];
	for (int i = 0; i < size; i++) {
		if (max < data[i]) {
			max = data[i];
		}
	}

	return max;
}

//最小値
template <class Type>
inline Type get_min(Type* data, int size)
{
	Type min = data[0];
	for (int i = 0; i < size; i++) {
		if (min > data[i]) {
			min = data[i];
		}
	}

	return min;
}

//最大値で正規化
template <class Type>
inline void norm_max(Type* data_in, int size, double* data_out)
{
	Type max;
	max = get_max(data_in, size);

	for (int i = 0; i < size; i++) {
		data_out[i] = (double)((double)data_in[i] / (double)max);
	}

}

//最大値と最小値で正規化(確実に0~1)
template <class Type>
inline void norm_max_min(Type* data_in, int size, double* data_out)
{
	Type max;
	max = get_max(data_in, size);
	Type min;
	min = get_min(data_in, size);

	double* tmp;
	tmp = new double[size];

	for (int i = 0; i < size; i++) {
		tmp[i] = (double)((data_in[i] - min) / (max - min));
	}

	for (int i = 0; i < size; i++) {
		data_out[i] = tmp[i];
	}


	delete[]tmp;
}

//正規化後,n階調化,int型出力
template <typename Type>
inline void norm_reso_n(Type* data_in, int* data_out, int n, int size)
{
	double* tmp;
	tmp = new double[size];
	norm_max_min(data_in, size, tmp);

	for (int i = 0; i < size; i++) {
		data_out[i] = (int)(round(tmp[i] * n));
	}

	delete[]tmp;
}


//intデータをunsigned charに変換後、格納
//0~255のデータに限る
inline void to_uch(int* data_in, int size, unsigned char* data_img)
{
	for (int i = 0; i < size; i++) {
		data_img[i] = (unsigned char)data_in[i];
	}
}


//1次元画像の上下反転
template <class Type>
inline void invert_img(Type* data_in, Type* data_out, int im_x, int im_y)
{
	Type* tmp;
	tmp = new Type[im_x * im_y];

	for (int i = 0; i < im_y; i++) {
		for (int j = 0; j < im_x; j++) {
			tmp[i * im_x + j] = data_in[((im_y - 1) - i) * im_x + j];

		}
	}

	for (int i = 0; i < im_y; i++) {
		for (int j = 0; j < im_x; j++) {
			data_out[i * im_x + j] = tmp[i * im_x + j];

		}
	}



	delete[]tmp;
}


//中心1列取り出す
template <class Type>
inline void mid_line(Type* data_in, int sx, int sy, Type* data_out) {
	for (int i = 0; i < sx; i++) {
		data_out[i] = data_in[(sy / 2) * sx + i];
	}
}



inline void Opad(double* img_out, int in_x, int in_y, double* img_in) {
	int x, y, X, Y;
	x = in_x;
	y = in_y;
	X = 2 * x;
	Y = 2 * y;

	double* img_tmp;
	img_tmp = new double[X * Y];

	for (int i = 0; i < X * Y; i++) {
		img_tmp[i] = 0;
	}

	//入力された画像データを０埋めして倍の大きさの画像にする
	for (int i = Y / 4; i < y + Y / 4; i++) {
		for (int j = X / 4; j < x + X / 4; j++) {
			img_tmp[i * X + j] = img_in[(i - Y / 4) * x + (j - X / 4)];
		}
	}

	for (int i = 0; i < X * Y; i++) {
		img_out[i] = img_tmp[i];
	}

	delete[]img_tmp;

}

//2D画像の0pad部分を取り除く関数(縦横それぞれ1/2倍にして真ん中を取得、inとoutはサイズ違う)
inline void elim_0(double* img_out, int in_x, int in_y, double* img_in) {
	int x, y, X, Y;
	X = in_x;
	Y = in_y;
	x = X / 2;
	y = Y / 2;

	double* tmp;
	tmp = new double[x * y];
	for (int i = 0; i < y * x; i++) {
		tmp[i] = 0;
	}

	for (int i = Y / 4; i < y + Y / 4; i++) {
		for (int j = X / 4; j < x + X / 4; j++) {
			tmp[(i - Y / 4) * x + (j - X / 4)] = img_in[i * X + j];
		}
	}

	for (int i = 0; i < x * y; i++) {
		img_out[i] = tmp[i];
	}

	delete[]tmp;
}






