


Small test program to illustrate the usage of "staged concurrent copy and execute"
Try to make it work with zero-copy

uncomment the "#define DEBUG" to have some output to stdout



Compilattion
$ nvcc -std=c++11 -Xcompiler -Wall staged.cu -o staged

Profile the program:
$ nvprof -o staged.nvvp  ./staged



#####
CEPP Atos
paul.karlshoefer@atos.net

