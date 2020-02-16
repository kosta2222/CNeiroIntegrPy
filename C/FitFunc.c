#include "hedNN.h"
#include "hedPy.h"
#include "utilMacr.h"
#include <stdbool.h>
extern whole_NN_params NN[1];
extern int epochs[max_am_epoch];
extern float object_mse[max_am_objMse];
//---------------------[Fit Инс]--------------------------
void
fit(float *X, float *Y, int rows, int cols_train, int cols_teach, int eps, float lr, int debug) {
    NN->lr = lr;
    float mse_after_oneVector = 0;
    int epocha = 0;
    // итерации,обучение
    // временные вектора для процесса обучения
    float tmp_vec_x[max_in_nn];
    float tmp_vec_y[max_rows_orOut];
    while (epocha < eps)$
        printf("epoch: %d\n", epocha);
        for (int row = 0; row < rows; row++)$
             for (int elem = 0; elem < NN->inputNeurons; elem++)
                 tmp_vec_x[elem] = X[row * cols_train + elem];
             for (int elem = 0; elem < NN->outputNeurons; elem++)
                 tmp_vec_y[elem] = Y[row * cols_teach + elem];
             train(tmp_vec_x, tmp_vec_y, debug);
             print_deb_vector(getHidden(&NN->list[NN->nlCount - 1]),NN->outputNeurons,"in fit outNN vector");
             print_deb_vector(Y,NN->outputNeurons,"in fit Y");
             mse_after_oneVector = getMinimalSquareError(getHidden(&NN->list[NN->nlCount - 1]), Y, NN->outputNeurons);
             printf("mse: %f\n", mse_after_oneVector);
             $$
        /*
         *  Все векторы из пакета отдали, запишем последнюю ошибку
        */
//        if (mse_after_oneVector == 0) goto out_bach;             
        object_mse[epocha] = mse_after_oneVector;
        epochs[epocha] = epocha;
        epocha++;
        $$
    out_bach :;

}
int
fit_viaCV(float *X, float *Y, float *X_test, float * Y_test, int max_acc, int rows, int cols_train, int cols_teach, float lr, int debug) {
    NN->lr = lr;
    float mse_after_oneVector = 0;
    int epocha = 0;
    int tmp_acc=0;
    // итерации,обучение
    // временные вектора для процесса обучения
    float tmp_vec_x[max_in_nn];
    float tmp_vec_y[max_rows_orOut];
    while (true)$
        printf("epoch: %d\n", epocha);
        for (int row = 0; row < rows; row++)$
             for (int elem = 0; elem < NN->inputNeurons; elem++)
                 tmp_vec_x[elem] = X[row * cols_train + elem];
             for (int elem = 0; elem < NN->outputNeurons; elem++)
                 tmp_vec_y[elem] = Y[row * cols_teach + elem];
             train(tmp_vec_x, tmp_vec_y, debug);
             tmp_acc=(float )cross_validation(X_test,Y_test,rows,cols_train,cols_teach);
             // Если точность утверждения достигута-оканчиваем обучение
             if(tmp_acc==max_acc)
                 goto out_bach;
             print_deb_vector(getHidden(&NN->list[NN->nlCount - 1]),NN->outputNeurons,"in fit outNN vector");
             print_deb_vector(Y,NN->outputNeurons,"in fit Y");
             mse_after_oneVector = getMinimalSquareError(getHidden(&NN->list[NN->nlCount - 1]), Y, NN->outputNeurons);
             printf("mse: %f\n", mse_after_oneVector);
             $$
        /*
         *  Все векторы из пакета отдали, запишем последнюю ошибку
        */
//        if (mse_after_oneVector == 0) goto out_bach;             
        object_mse[epocha] = mse_after_oneVector;
        epochs[epocha] = epocha;
        epocha++;
        $$
    out_bach :;
    
    _0_("fit_viaCV");
    
}
//---------------------[Fit Инс]--------------------------
