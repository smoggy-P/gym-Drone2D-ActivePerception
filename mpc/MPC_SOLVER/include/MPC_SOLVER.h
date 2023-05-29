/*
MPC_SOLVER : A fast customized optimization solver.

Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.


This software is intended for simulation and testing purposes only. 
Use of this software for any commercial purpose is prohibited.

This program is distributed in the hope that it will be useful.
EMBOTECH makes NO WARRANTIES with respect to the use of the software 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. 

EMBOTECH shall not have any liability for any damage arising from the use
of the software.

This Agreement shall exclusively be governed by and interpreted in 
accordance with the laws of Switzerland, excluding its principles
of conflict of laws. The Courts of Zurich-City shall have exclusive 
jurisdiction in case of any dispute.

*/

/* Generated by FORCESPRO v6.1.0 on Saturday, May 13, 2023 at 1:35:22 AM */
#ifndef MPC_SOLVER_H
#define MPC_SOLVER_H

#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif
#ifndef SOLVER_STRING_H
#define SOLVER_STRING_H
#include <string.h>
#endif


#ifndef SOLVER_STANDARD_TYPES
#define SOLVER_STANDARD_TYPES

typedef signed char solver_int8_signed;
typedef unsigned char solver_int8_unsigned;
typedef char solver_int8_default;
typedef signed short int solver_int16_signed;
typedef unsigned short int solver_int16_unsigned;
typedef short int solver_int16_default;
typedef signed int solver_int32_signed;
typedef unsigned int solver_int32_unsigned;
typedef int solver_int32_default;
typedef signed long long int solver_int64_signed;
typedef unsigned long long int solver_int64_unsigned;
typedef long long int solver_int64_default;

#endif


/* DATA TYPE ------------------------------------------------------------*/
typedef double MPC_SOLVER_float;
typedef double MPC_SOLVER_ldl_s_float;
typedef double MPC_SOLVER_ldl_r_float;
typedef double MPC_SOLVER_callback_float;

typedef double MPC_SOLVERinterface_float;

/* SOLVER SETTINGS ------------------------------------------------------*/

/* MISRA-C compliance */
#ifndef MISRA_C_MPC_SOLVER
#define MISRA_C_MPC_SOLVER (0)
#endif

/* restrict code */
#ifndef RESTRICT_CODE_MPC_SOLVER
#define RESTRICT_CODE_MPC_SOLVER (0)
#endif

/* print level */
#ifndef SET_PRINTLEVEL_MPC_SOLVER
#define SET_PRINTLEVEL_MPC_SOLVER    (0)
#endif

/* timing */
#ifndef SET_TIMING_MPC_SOLVER
#define SET_TIMING_MPC_SOLVER    (1)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define SET_MAXIT_MPC_SOLVER			(5000)	

/* scaling factor of line search (FTB rule) */
#define SET_FLS_SCALE_MPC_SOLVER		(MPC_SOLVER_float)(0.99)      

/* maximum number of supported elements in the filter */
#define MAX_FILTER_SIZE_MPC_SOLVER	(5000) 

/* maximum number of supported elements in the filter */
#define MAX_SOC_IT_MPC_SOLVER			(4) 

/* desired relative duality gap */
#define SET_ACC_RDGAP_MPC_SOLVER		(MPC_SOLVER_float)(0.0001)

/* desired maximum residual on equality constraints */
#define SET_ACC_RESEQ_MPC_SOLVER		(MPC_SOLVER_float)(1E-06)

/* desired maximum residual on inequality constraints */
#define SET_ACC_RESINEQ_MPC_SOLVER	(MPC_SOLVER_float)(1E-06)

/* desired maximum violation of complementarity */
#define SET_ACC_KKTCOMPL_MPC_SOLVER	(MPC_SOLVER_float)(1E-06)

/* whether callback return values should be checked */
#define EXTFUNC_RETURN_MPC_SOLVER (0)

/* SOLVER RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define OPTIMAL_MPC_SOLVER      (1)

/* maximum number of iterations has been reached */
#define MAXITREACHED_MPC_SOLVER (0)

/* solver has stopped due to a timeout */
#define TIMEOUT_MPC_SOLVER   (2)

/* solver stopped externally */
#define EXIT_EXTERNAL_MPC_SOLVER (3)

/* wrong number of inequalities error */
#define INVALID_NUM_INEQ_ERROR_MPC_SOLVER  (-4)

/* factorization error */
#define FACTORIZATION_ERROR_MPC_SOLVER   (-5)

/* NaN encountered in function evaluations */
#define BADFUNCEVAL_MPC_SOLVER  (-6)

/* invalid value (<= -100) returned by external function */
#define INVALIDFUNCEXIT_MPC_SOLVER (-200)

/* bad value returned by external function */
#define BADFUNCEXIT_MPC_SOLVER(status) (status > -100? status - 200 : INVALIDFUNCEXIT_MPC_SOLVER)

/* no progress in method possible */
#define NOPROGRESS_MPC_SOLVER   (-7)

/* regularization error */
#define REGULARIZATION_ERROR_MPC_SOLVER   (-9)

/* invalid values in parameters */
#define PARAM_VALUE_ERROR_MPC_SOLVER   (-11)

/* too small timeout given */
#define INVALID_TIMEOUT_MPC_SOLVER   (-12)

/* thread error */
#define THREAD_FAILURE_MPC_SOLVER  (-98)

/* locking mechanism error */
#define LOCK_FAILURE_MPC_SOLVER  (-99)

/* licensing error - solver not valid on this machine */
#define LICENSE_ERROR_MPC_SOLVER  (-100)

/* Insufficient number of internal memory instances.
 * Increase codeoptions.max_num_mem. */
#define MEMORY_INVALID_MPC_SOLVER (-101)
/* Number of threads larger than specified.
 * Increase codeoptions.nlp.max_num_threads. */
#define NUMTHREADS_INVALID_MPC_SOLVER (-102)


/* INTEGRATORS RETURN CODE ------------*/
/* Integrator ran successfully */
#define INTEGRATOR_SUCCESS (11)
/* Number of steps set by user exceeds maximum number of steps allowed */
#define INTEGRATOR_MAXSTEPS_EXCEEDED (12)


/* MEMORY STRUCT --------------------------------------------------------*/
typedef struct MPC_SOLVER_mem MPC_SOLVER_mem;
#ifdef __cplusplus
extern "C" {
#endif
/* MEMORY STRUCT --------------------------------------------------------*/
extern MPC_SOLVER_mem * MPC_SOLVER_external_mem(void * mem_ptr, solver_int32_unsigned i_mem, size_t mem_size);
extern size_t MPC_SOLVER_get_mem_size( void );
extern size_t MPC_SOLVER_get_const_size( void );
#ifdef __cplusplus
}
#endif

/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct
{
    /* vector of size 175 */
    MPC_SOLVER_float x0[175];

    /* vector of size 4 */
    MPC_SOLVER_float xinit[4];

    /* vector of size 700 */
    MPC_SOLVER_float all_parameters[700];

    /* scalar */
    solver_int32_unsigned num_of_threads;


} MPC_SOLVER_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct
{
    /* column vector of length 7 */
    MPC_SOLVER_float x01[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x02[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x03[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x04[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x05[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x06[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x07[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x08[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x09[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x10[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x11[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x12[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x13[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x14[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x15[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x16[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x17[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x18[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x19[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x20[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x21[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x22[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x23[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x24[7];

    /* column vector of length 7 */
    MPC_SOLVER_float x25[7];


} MPC_SOLVER_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct
{
    /* scalar: iteration number */
    solver_int32_default it;

    /* scalar: number of iterations needed to optimality (branch-and-bound) */
    solver_int32_default it2opt;

    /* scalar: inf-norm of equality constraint residuals */
    MPC_SOLVER_float res_eq;

    /* scalar: inf-norm of inequality constraint residuals */
    MPC_SOLVER_float res_ineq;

    /* scalar: norm of stationarity condition */
    MPC_SOLVER_float rsnorm;

    /* scalar: max of all complementarity violations */
    MPC_SOLVER_float rcompnorm;

    /* scalar: primal objective */
    MPC_SOLVER_float pobj;

    /* scalar: dual objective */
    MPC_SOLVER_float dobj;

    /* scalar: duality gap := pobj - dobj */
    MPC_SOLVER_float dgap;

    /* scalar: relative duality gap := |dgap / pobj | */
    MPC_SOLVER_float rdgap;

    /* scalar: duality measure */
    MPC_SOLVER_float mu;

    /* scalar: duality measure (after affine step) */
    MPC_SOLVER_float mu_aff;

    /* scalar: centering parameter */
    MPC_SOLVER_float sigma;

    /* scalar: number of backtracking line search steps (affine direction) */
    solver_int32_default lsit_aff;

    /* scalar: number of backtracking line search steps (combined direction) */
    solver_int32_default lsit_cc;

    /* scalar: step size (affine direction) */
    MPC_SOLVER_float step_aff;

    /* scalar: step size (combined direction) */
    MPC_SOLVER_float step_cc;

    /* scalar: total solve time */
    MPC_SOLVER_float solvetime;

    /* scalar: time spent in function evaluations */
    MPC_SOLVER_float fevalstime;

    /* column vector of length 8: solver ID of FORCESPRO solver */
    solver_int32_default solver_id[8];




} MPC_SOLVER_info;







/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* Time of Solver Generation: (UTC) Saturday, May 13, 2023 1:35:24 AM */
/* User License expires on: (UTC) Tuesday, September 26, 2023 10:00:00 PM (approx.) (at the time of code generation) */
/* Solver Static License expires on: (UTC) Tuesday, September 26, 2023 10:00:00 PM (approx.) */
/* Solver Id: 5a962d07-9537-48c2-b096-da9033560c15 */
/* examine exitflag before using the result! */
#ifdef __cplusplus
extern "C" {
#endif		

typedef solver_int32_default (*MPC_SOLVER_extfunc)(MPC_SOLVER_float* x, MPC_SOLVER_float* y, MPC_SOLVER_float* lambda, MPC_SOLVER_float* params, MPC_SOLVER_float* pobj, MPC_SOLVER_float* g, MPC_SOLVER_float* c, MPC_SOLVER_float* Jeq, MPC_SOLVER_float* h, MPC_SOLVER_float* Jineq, MPC_SOLVER_float* H, solver_int32_default stage, solver_int32_default iterations, solver_int32_default threadID);

extern solver_int32_default MPC_SOLVER_solve(MPC_SOLVER_params *params, MPC_SOLVER_output *output, MPC_SOLVER_info *info, MPC_SOLVER_mem *mem, FILE *fs, MPC_SOLVER_extfunc evalextfunctions_MPC_SOLVER);











#ifdef __cplusplus
}
#endif

#endif
