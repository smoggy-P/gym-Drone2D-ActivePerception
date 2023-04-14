/*
 * AD tool to FORCESPRO Template - missing information to be filled in by createADTool.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif
    
#include "include/MPC_SOLVER.h"

#ifndef NULL
#define NULL ((void *) 0)
#endif

#include "MPC_SOLVER_model.h"



/* copies data from sparse matrix into a dense one */
static void MPC_SOLVER_sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, MPC_SOLVER_callback_float *data, MPC_SOLVER_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for(j=colidx[i]; j<colidx[i+1]; j++)
        {
            out[i*nrow + row[j]] = ((MPC_SOLVER_float) data[j]);
        }
    }
}




/* AD tool to FORCESPRO interface */
extern solver_int32_default MPC_SOLVER_adtool2forces(MPC_SOLVER_float *x,        /* primal vars                                         */
                                 MPC_SOLVER_float *y,        /* eq. constraint multiplers                           */
                                 MPC_SOLVER_float *l,        /* ineq. constraint multipliers                        */
                                 MPC_SOLVER_float *p,        /* parameters                                          */
                                 MPC_SOLVER_float *f,        /* objective function (scalar)                         */
                                 MPC_SOLVER_float *nabla_f,  /* gradient of objective function                      */
                                 MPC_SOLVER_float *c,        /* dynamics                                            */
                                 MPC_SOLVER_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 MPC_SOLVER_float *h,        /* inequality constraints                              */
                                 MPC_SOLVER_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 MPC_SOLVER_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                           */
								 solver_int32_default iteration, /* iteration number of solver                         */
								 solver_int32_default threadID   /* Id of caller thread                                */)
{
    /* AD tool input and output arrays */
    const MPC_SOLVER_callback_float *in[4];
    MPC_SOLVER_callback_float *out[7];
    

    /* Allocate working arrays for AD tool */
    
    MPC_SOLVER_callback_float w[46];
	
    /* temporary storage for AD tool sparse output */
    MPC_SOLVER_callback_float this_f = (MPC_SOLVER_callback_float) 0.0;
    MPC_SOLVER_float nabla_f_sparse[5];
    MPC_SOLVER_float h_sparse[6];
    MPC_SOLVER_float nabla_h_sparse[17];
    MPC_SOLVER_float c_sparse[4];
    MPC_SOLVER_float nabla_c_sparse[10];
            
    
    /* pointers to row and column info for 
     * column compressed format used by AD tool */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for AD tool */
    in[0] = x;
    in[1] = p;
    in[2] = l;
    in[3] = y;

	if ((0 <= stage && stage <= 13))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		MPC_SOLVER_objective_0(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = MPC_SOLVER_objective_0_sparsity_out(1)[0];
			ncol = MPC_SOLVER_objective_0_sparsity_out(1)[1];
			colind = MPC_SOLVER_objective_0_sparsity_out(1) + 2;
			row = MPC_SOLVER_objective_0_sparsity_out(1) + 2 + (ncol + 1);
			MPC_SOLVER_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = c_sparse;
		out[1] = nabla_c_sparse;
		MPC_SOLVER_dynamics_0(in, out, NULL, w, 0);
		if( c )
		{
			nrow = MPC_SOLVER_dynamics_0_sparsity_out(0)[0];
			ncol = MPC_SOLVER_dynamics_0_sparsity_out(0)[1];
			colind = MPC_SOLVER_dynamics_0_sparsity_out(0) + 2;
			row = MPC_SOLVER_dynamics_0_sparsity_out(0) + 2 + (ncol + 1);
			MPC_SOLVER_sparse2fullcopy(nrow, ncol, colind, row, c_sparse, c);
		}
		if( nabla_c )
		{
			nrow = MPC_SOLVER_dynamics_0_sparsity_out(1)[0];
			ncol = MPC_SOLVER_dynamics_0_sparsity_out(1)[1];
			colind = MPC_SOLVER_dynamics_0_sparsity_out(1) + 2;
			row = MPC_SOLVER_dynamics_0_sparsity_out(1) + 2 + (ncol + 1);
			MPC_SOLVER_sparse2fullcopy(nrow, ncol, colind, row, nabla_c_sparse, nabla_c);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		MPC_SOLVER_inequalities_0(in, out, NULL, w, 0);
		if( h )
		{
			nrow = MPC_SOLVER_inequalities_0_sparsity_out(0)[0];
			ncol = MPC_SOLVER_inequalities_0_sparsity_out(0)[1];
			colind = MPC_SOLVER_inequalities_0_sparsity_out(0) + 2;
			row = MPC_SOLVER_inequalities_0_sparsity_out(0) + 2 + (ncol + 1);
			MPC_SOLVER_sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = MPC_SOLVER_inequalities_0_sparsity_out(1)[0];
			ncol = MPC_SOLVER_inequalities_0_sparsity_out(1)[1];
			colind = MPC_SOLVER_inequalities_0_sparsity_out(1) + 2;
			row = MPC_SOLVER_inequalities_0_sparsity_out(1) + 2 + (ncol + 1);
			MPC_SOLVER_sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
	if ((14 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		MPC_SOLVER_objective_1(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = MPC_SOLVER_objective_1_sparsity_out(1)[0];
			ncol = MPC_SOLVER_objective_1_sparsity_out(1)[1];
			colind = MPC_SOLVER_objective_1_sparsity_out(1) + 2;
			row = MPC_SOLVER_objective_1_sparsity_out(1) + 2 + (ncol + 1);
			MPC_SOLVER_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		MPC_SOLVER_inequalities_1(in, out, NULL, w, 0);
		if( h )
		{
			nrow = MPC_SOLVER_inequalities_1_sparsity_out(0)[0];
			ncol = MPC_SOLVER_inequalities_1_sparsity_out(0)[1];
			colind = MPC_SOLVER_inequalities_1_sparsity_out(0) + 2;
			row = MPC_SOLVER_inequalities_1_sparsity_out(0) + 2 + (ncol + 1);
			MPC_SOLVER_sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = MPC_SOLVER_inequalities_1_sparsity_out(1)[0];
			ncol = MPC_SOLVER_inequalities_1_sparsity_out(1)[1];
			colind = MPC_SOLVER_inequalities_1_sparsity_out(1) + 2;
			row = MPC_SOLVER_inequalities_1_sparsity_out(1) + 2 + (ncol + 1);
			MPC_SOLVER_sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
    
    /* add to objective */
    if (f != NULL)
    {
        *f += ((MPC_SOLVER_float) this_f);
    }

    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
