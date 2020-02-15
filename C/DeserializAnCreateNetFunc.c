#include "hedNN.h"
#include "utilMacr.h"
extern whole_NN_params NN[1];
int make_kernel_f(nnLay *list, int lay_pos, float *matrix_el_st, int * ops_st, int sp_op) {
    int out = ops_st[sp_op];
    int in = ops_st[sp_op - 1];
    list[lay_pos].out = out;
    list[lay_pos].in = in;
    for (int row = 0; row < out; row++)
        for (int elem = 0; elem < in; elem++)
            list[lay_pos].matrix[row][elem] = matrix_el_st[row * elem];
    _0_(" make_kernel");
}

int vm(nnLay *list, u_char *bin_buf) {
    float matrix_el_st[max_stack_matrEl];
    int ops_st[max_stack_otherOp];
    int ip = 0;
    int sp_ma = -1;
    int sp_op = -1;
    u_char op = -1;
    float arg = 0;
    int n_lay = 0;
    op = bin_buf[ip];
    while (op != stop) {
        switch (op) {
            // загружаем на стек количество входов и выходов ядра
            case push_i:
            {
                ops_st[++sp_op] = bin_buf[++ip];
                break;
            }
                // загружаем на стек элементы матриц
            case push_fl:
            {
                arg = *((float *) (&bin_buf[ip + 1]));
                matrix_el_st[++sp_ma] = arg;
                ip += 4;
                break;
            }
                // создаем одно ядро в массиве
            case make_kernel:
            {
                make_kernel_f(list, n_lay, matrix_el_st, ops_st, sp_op);
                // переходим к следующему индексу ядра
                n_lay++;
                // зачищаем стеки
                sp_op = -1;
                sp_ma = -1;
                break;
            }
        }
        // показываем на следующую инструкцию
        ip++;
        op = bin_buf[ip];
    }
    // также подсчитаем сколько у наc ядер
    NN->nlCount = n_lay;
    // находим количество входов
    NN->inputNeurons=(NN->list[0].in)-1;// -1 зависит от биасов
    // находим количество выходов когда образовали сеть
    NN->outputNeurons=NN->list[NN->nlCount-1].out;
    _0_("vm");
}

int vm_deserializ(nnLay * list, char* f_name) {
    u_char bin_buf[bin_kernel_bufLen];
    FILE * fp;
    fp = fopen(f_name, "rb");
    is_null_ptrErr(fp, "vm_deserializ", "fp");
    check_s(ftell(fp), bin_kernel_bufLen, "vm_deserializ", "bin_kernel_bufLen");
    while (fread(bin_buf, 1, 256, fp) != NULL);
    fclose(fp);
    // разборка байт-кода
    vm(list, bin_buf);
    _0_("vm_deserializ");
}
