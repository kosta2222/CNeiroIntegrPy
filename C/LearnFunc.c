#include "hedNN.h"
#include "hedPy.h"
#include "utilMacr.h"
//-----------------[Основные функции обучения]------------

void initiate_layers(int *network_map, int size) {
    int in = 0;
    int out = 0;
    NN->nlCount = size - 1;
    NN->inputNeurons = network_map[0];
    NN->outputNeurons = network_map[NN->nlCount];
    //    setIO(&NN->list[0], network_map[0], network_map[1]);
    setIO(&NN->list[0], network_map[0], network_map[1]);
    for (int i = 2; i <= NN->nlCount; i++)
        in = network_map[i - 1], out = network_map[i], setIO(&NN->list[i - 1], in, out), printf("in: %d \t out:%d\n", in, out);

}

void
backPropagate() {

    /* Вычисление ошибки */calcOutError(&NN->list[NN->nlCount - 1], NN->targets);

    calcHidError(&NN->list[NN->nlCount - 1], getEssentialGradients(&NN->list[NN->nlCount - 1]), getCostSignals(&NN->list[NN->nlCount - 1 ]));

    for (int i = NN->nlCount - 2; i > 0; i--) calcHidError(&NN->list[i], getEssentialGradients(&NN->list[i + 1]), getCostSignals(&NN->list[i - 1]));
    /* Последнему слою не нужны входа т.к. у них нет функции активации */
    calcHidZeroLay(&NN->list[0], getEssentialGradients(&NN->list[1]));
    /* Обновление весов */for (int i = NN->nlCount - 1; i > 0; i--)updMatrix(&NN->list[i], getCostSignals(&NN->list[i - 1]));
    updMatrix(&NN->list[0], NN->inputs);
}

void
train(float *in, float *targ, int debug) {
    /*
     *  Работает с рядом из матриц X и Y
     */
    copy_vector(in, NN->inputs, max_in_nn);
    copy_vector(targ, NN->targets, max_rows_orOut);
    feedForwarding(false, debug);


}

void
predict(float* in, int debug) {
    /*
     *  Работает с одним вектором
     */
    copy_vector(in, NN->inputs, max_in_nn);
    feedForwarding(true, debug);
}

void
feedForwarding(bool ok, int debug) {
    // если ok = true - обучаемся, перед этим выполним один проход по сети
    makeHidden(&NN->list[0], NN->inputs, debug);
            // для данного слоя получить то что отдал пред-слой
            // получаем отдачу слоя и передаем ее следующему  справа как аргумент
    for (int i = 1; i < NN->nlCount; i++) makeHidden(&NN->list[i], getHidden(&NN->list[i - 1]), debug);
    if (ok) for (int out = 0; out < NN->outputNeurons; out++) printf("%d item val %f;", out + 1, NN->list[NN->nlCount - 1].hidden[out] * koef_to_predict);
    else backPropagate();
    }

int
getInCount(nnLay *curLay) {

    return curLay->in;
}

int
getOutCount(nnLay *curLay) {

    return curLay->out;
}

//float *
//getMatrix(nnLay *curLay) {
//        float matrix[(curLay->out)*(curLay->in)];
//	copy_matrix(matrix,curLay->matrix,curLay->out,curLay->in);
//	return matrix;
//}

void
updMatrix(nnLay *curLay, float *enteredVal) {
    for (int row = 0; row < curLay->out; row++)
        for (int elem = 0; elem < curLay->in; elem++, curLay->matrix[row][elem] -= NN->lr * curLay->errors[elem] * enteredVal[elem]);
}

void
setIO(nnLay *curLay, int inputs, int outputs) {
    /* сенсоры - входа*/curLay->in = inputs + 1;
    /* данный ряд нейронов */curLay->out = outputs;
    for (int row = 0; row < curLay->out; row++)for (int elem = 0; elem < curLay->in; elem++)printf("operations\n"), curLay->matrix[row][elem] = operations(INIT_W_HE, curLay->in, 0, 0, 0, "");

}

void
makeHidden(nnLay *curLay, float *inputs, int debug) {
    float tmpS = 0;
    float val = 0;
    for (int row = 0; row < curLay->out; row++) {
        for (int elem = 0; elem < curLay->in; elem++)
            if (elem == 0) tmpS += curLay->matrix[row][0];
            else tmpS += curLay->matrix[row][elem] * inputs[elem];
        curLay->cost_signals[row] = tmpS, val = relu(tmpS), curLay->hidden[row] = val, operations(debug, curLay->cost_signals[row], 0, 0, 0, "cost signals");
        tmpS = 0;
    }
    operations(debug, 0, 0, 0, 0, "make hidden made");
}

float *
getCostSignals(nnLay * curLay) {

    return curLay->cost_signals;
}

float*
getHidden(nnLay *curLay) {

    return curLay->hidden;
}

void
calcOutError(nnLay *curLay, float *targets) {
    for (int row = 0; row < curLay->out; row++, curLay->errors[row] = (curLay->hidden[row] - targets[row]) * derivateRelu(curLay->cost_signals[row]));
}

void
calcHidError(nnLay *curLay, float *essential_gradients, float *enteredVals) {
    for (int elem = 0; elem < curLay->in; elem++) for (int row = 0; row < curLay->out; row++, curLay->errors[elem] += essential_gradients[row] * curLay->matrix[row][elem] * derivateRelu(enteredVals[elem]));
}

void
calcHidZeroLay(nnLay* zeroLay, float * essential_gradients) {
    for (int elem = 0; elem < zeroLay->in; elem++) for (int row = 0; row < zeroLay->out; row++) zeroLay->errors[elem] += essential_gradients[row] * zeroLay->matrix[row][elem];
}

float*
getEssentialGradients(nnLay *curLay) {
    return curLay->errors;
}

float
getMinimalSquareError(float *out_nn, float* teacher_answ, int size_vec) {
    float sum = 0;
    float square = 0;
    float mean = 0;
    for (int row = 0, sum = 0; row < size_vec; row++) sum += out_nn[row] - teacher_answ[row];
    square = pow(sum, 2);
    mean = square / size_vec;
    return mean;
}

float
sigmoida(float val) {
    float res = (1.0 / (1.0 + exp(val)));
    return res;
}

float
sigmoidasDerivate(float val) {
    float res = exp(-val) / (pow((1 + exp(-val)), 2));
    return res;
}

float relu(float x) {
    if (x < 0)
        return 0;
    else
        return x;
}

float derivateRelu(float x) {
    if (x < 0)
        return 0;
    else
        return 1;
}
//-----------------[/Основные функции обучения]--------------