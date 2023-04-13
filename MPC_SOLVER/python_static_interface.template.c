/* This template is used when statically linking a solver and its external
 * evaluation functions (for nonlinearties). It simply exports a function that
 * is essentially a closure around the solver function, with the last argument
 * (the pointer to the external evaluation function) fixed.
 *
 * The template is used by setting the following preprocessor macros:
 *  - MPC_SOLVER
 *  - "include/MPC_SOLVER.h"
 *  - MPC_SOLVER_adtool2forces
 *  - MPC_SOLVER_interface
 *
 * Compare also the MEX interface, which exists for a similar purpose in the
 * MATLAB client. */

#define CONCAT(x, y) x ## y
#define CONCATENATE(x, y) CONCAT(x, y)
#define SOLVER_FLOAT CONCATENATE(MPC_SOLVER, _float)
#define SOLVER_FUN_NAME CONCATENATE(MPC_SOLVER, _solve)

#include "include/MPC_SOLVER.h"

/* Header of external evaluation function */
solver_int32_default MPC_SOLVER_adtool2forces(SOLVER_FLOAT *x, SOLVER_FLOAT *y, SOLVER_FLOAT *l,
                   SOLVER_FLOAT *p, SOLVER_FLOAT *f, SOLVER_FLOAT *nabla_f,
                   SOLVER_FLOAT *c, SOLVER_FLOAT *nabla_c, SOLVER_FLOAT *h,
                   SOLVER_FLOAT *nabla_h, SOLVER_FLOAT *hess,
                   solver_int32_default stage, solver_int32_default iteration, solver_int32_default threadID);

int MPC_SOLVER_interface(void *params, void *outputs, void *info, void *mem, FILE *fp) {
    return SOLVER_FUN_NAME(params, outputs, info, mem, fp, &MPC_SOLVER_adtool2forces);
}
