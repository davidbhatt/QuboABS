# QuboABS
QUBO solver using ABS algorithm
This code solves the NP hard problem of QUBO (Quadratic unconstrained Binary Optimization) 
It follows the Adaptive Bulk Search Algorithm described in the paper 
Ryota Yasudo, Koji Nakano, Yasuaki Ito, Masaru Tatekawa, Ryota Katsuki, Takashi Yazane, and Yoko Inaba. 2020. 
Adaptive Bulk Search: Solving Quadratic Unconstrained Binary Optimization Problems on Multiple GPUs using CUDA C++. 
In 49th International Conference on Parallel Processing - ICPP (ICPP '20). Association for Computing Machinery, New York, NY, USA, Article 62, 1–11.
 https://doi.org/10.1145/3404397.3404423
This is implemented in CUDA C++
to compile in linux use ./compile.sh after changing necessary compile flags
the options to be provided are
-i filename #compulsory options filename containing the qubo matrix in qubo format
#optional options include
-it  no of iterations ofr local search
-nsol no of parallel instances of solutions to generate

to run use 
./a.out -i qubo_filename
sample qubo file 16.txt is provided
