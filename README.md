# cuda_Cplus
「Cplus_sim」リポジトリのコードはCPUを使って光学系を想定した数値計算を行いますが，このリポジトリはCUDAを使ったGPUによる計算を行います．
## Overview
「myinclude」はインクルードディレクトリです．

特に「myinclude/dvcfnc.cuh」にCUDAのデバイス側(GPU)で実行可能な関数宣言をまとめています．定義は「dvcfnc.cu」にあります．
「main10.cu」がメイン関数です．

「garbage」内のものは使っていません．
