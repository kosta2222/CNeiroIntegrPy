#include "hedNN.h"
#include "hedPy.h"
#include "utilMacr.h"
//---------------------[Fit Инс]--------------------------
void
fit(float *X, float *Y, int rows, int cols_train, int cols_teach, int eps, float lr, int debug) {
    NN->lr = lr;
    float mse_after_oneVector = 0;
    int epocha = 0;
    // итерации,обучение
    // временные вектора для процесса обучения
    float tmp_vec_x[NN->inputNeurons];
    float tmp_vec_y[NN->outputNeurons];
    while (epocha < eps)$
        printf("epoch: %d\n", epocha);
    for (int row = 0; row < rows; row++)$
        for (int elem = 0; elem < NN->inputNeurons; elem++) tmp_vec_x[elem] = X[row * cols_train + elem];
    for (int elem = 0; elem < NN->outputNeurons; elem++) tmp_vec_y[elem] = Y[row * cols_teach + elem];
    //    train(tmp_vec_x, tmp_vec_y, debug);
    c(train(tmp_vec_x, tmp_vec_y, debug), "train(tmp_vec_x, tmp_vec_y, debug)")
    mse_after_oneVector = getMinimalSquareError(getHidden(&NN->list[NN->nlCount - 1]), Y, NN->outputNeurons);
    printf("mse: %f\n", mse_after_oneVector);
    if (mse_after_oneVector == 0) goto out_bach;
    $$
    /*
     *  Все векторы из пакета отдали, запишем последнюю ошибку
     */
    object_mse[epocha] = mse_after_oneVector;
    epochs[epocha] = epocha;
    epocha++;
    $$
    out_bach :;

}
//---------------------[Fit Инс]--------------------------
