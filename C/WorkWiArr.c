#include "hedNN.h"
#include "utilMacr.h"

void copy_vector(float *src, float *dest, int n) {
    for (int i = 0; i < n; i++) dest[i] = src[i];
}

void copy_matrixAsRibons(float *src, float *dest, int rows, int cols) {
    for (int row = 0; row < rows; row++)
        for (int elem = 0; elem < cols; elem++) dest[row * cols + elem] = src[row * cols + elem];
}

void print_deb_matrixAsRibon(float *vec, int rows, int cols, char * label) {
    printf("%s:\n", label);
    lbr;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            printf("%f ", vec[i * cols + j]);
    printf("\n");
    rbr;
}

void print_deb_vector(float *vec, int cols, char * label) {
    printf("%s:\n", label);
    lbr;
    for (int i = 0; i < cols; i++)
        printf("%f ", vec[i]);
    rbr;
    nl;
}

void print_deb_vectorInt(int *vec, int cols, char * label) {
    printf("%s:\n", label);
    lbr;
    for (int i = 0; i < cols; i++)
        printf("%d ", vec[i]);
    rbr;
    nl;
}

int copy_matrixAsStaticSquare_toRibon(float src[][max_in_nn], float *dest, int in, int out) {
    for (int row = 0; row < out; row++)
        for (int elem = 0; elem < in; elem++)
            dest[row * in + elem] = src[row][elem];
    _0_("copy_matrix_as_vec");
}

int len_of_vecInt(int * vec){
    return sizeof(vec) / sizeof (int);
}
