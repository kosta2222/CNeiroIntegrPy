/* 
 * File:   utilMacr.h
 * Author: papa
 *
 * Created on 3 февраля 2020 г., 21:54
 */

#ifndef UTILMACR_H
#define	UTILMACR_H
#define c(f,id){debug==DEBUG?printf("%s\n",id),f:f;} 

#define for_in(from,to)\
for(int cn=from;cn<to;cn++)
#define $ {
#define $$ }
#define lbr printf("[")
#define rbr printf("]\n")
#define nl printf("\n")
#define _0_(func) \
lbr;\
printf("Exit succes\n");\
printf("in func->%s\n",func);\
rbr;\
return 0
#define _e_(func,var) \
printf("in func->%s\n",func);printf("var->%s\n",var);
#define ms0 printf ("Static memory error\n")
#define md0 printf ("Dinamic memory error\n")
#define check_d(r,func,var) if(r==0) {lbr;md0;_e_(func,var);rbr;}/*if макрос работает только на одной строке*/
#define check_s(i,max_buf,func,var) if(i>max_buf){lbr;ms0;_e_(func,var);rbr;}
#endif	/* UTILMACR_H */

