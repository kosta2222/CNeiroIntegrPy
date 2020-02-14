#include "hedNN.h"
#include "utilMacr.h"
extern whole_NN_params NN[1];

int check_2oneHotVecs(float *out_NN, float *vec_y_test, int vec_size);
float calc_accur( int *scores, int rows);

int cross_validation(float * X_test, float *Y_test, int rows, int cols_X_test, int cols_Y_test) {
    float tmp_vec_x_test[max_in_nn];
    float tmp_vec_y_test[max_rows_orOut];
    int scores[max_validSet_rows];
    int index_row = 0;
    int res=0;
    for (int row = 0; row < rows; row++)$
        for (int elem = 0; elem < NN->inputNeurons; elem++)
            tmp_vec_x_test[elem] = X_test[row * cols_X_test + elem];
        for (int elem = 0; elem < NN->outputNeurons; elem++)
            tmp_vec_y_test[elem] = Y_test[row * cols_Y_test + elem];
        predict_direct(tmp_vec_x_test, debug);
        res=check_2oneHotVecs(getHidden(&NN->list[NN->nlCount - 1]), tmp_vec_y_test, NN->outputNeurons);
        scores[index_row] =res; 
        printf("in cross_val res:%d\n",res);
        index_row++;
    $$
    print_deb_vector(scores,rows,"in cross val scores");
    printf("Accuracy:%f %\n",calc_accur(scores,rows));
    _0_("cross_validation");
}

/*
 *  Возвращает 1 если вектора равны и 0 если не равны
 */
int check_2oneHotVecs(float *out_NN, float *vec_y_test, int vec_size) {
    float tmp_elemOf_outNN_asHot = 0;
    for (int col = 0; col < vec_size; col++)$
        tmp_elemOf_outNN_asHot = (out_NN[col] > 0.5) ? 1 : 0;
        if (tmp_elemOf_outNN_asHot == vec_y_test[col]) continue;
           else return 0;
    $$
    printf("in check_2oneHotVecs tmp_elemOf_outNN_asHot %f\n",tmp_elemOf_outNN_asHot);
    return 1;
}

float calc_accur( int *scores, int rows) {
    print_deb_vector(scores,rows,"in calc_accur scores");
    float accuracy = 0;
    int sum = 0;
    // Посчитаем аккуратность
    for (int col = 0; col < rows; col++) sum += scores[col];
    accuracy = (float) (sum / rows) * 100; // выразим в процентах
    return accuracy;
}
