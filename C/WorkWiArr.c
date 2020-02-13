void copy_vector(float *src, float *dest, int n) {
    for (int i = 0; i < n; i++) dest[i] = src[i];
}

void copy_matrix(float *src, float *dest, int rows, int cols) {
    for (int row = 0; row < rows; row++)
        for (int elem = 0; elem < cols; elem++) dest[row * cols + elem] = src[row * cols + elem];
}

void print_deb_matrix(float *vec, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) printf("%f", vec[i * cols + j]), printf("\n");
}
