import numpy
import ctypes

name = "MPC_SOLVER"
requires_callback = True
lib = "lib/libMPC_SOLVER.so"
lib_static = "lib/libMPC_SOLVER.a"
c_header = "include/MPC_SOLVER.h"
nstages = 15

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  4,   1),    4),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (105,   1),  105),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (390,   1),  390),
 ("num_of_threads"      , "dense" , "solver_int32_unsigned", ctypes.c_uint  , numpy.uint32 , (  1,   1),    1)]

# Output                | Type    | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x02"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x03"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x04"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x05"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x06"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x07"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x08"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x09"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x10"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x11"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x12"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x13"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x14"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x15"                 , ""               , ctypes.c_double, numpy.float64,     (  7,),    7)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
 ('it2opt', ctypes.c_int),
 ('res_eq', ctypes.c_double),
 ('res_ineq', ctypes.c_double),
 ('rsnorm', ctypes.c_double),
 ('rcompnorm', ctypes.c_double),
 ('pobj', ctypes.c_double),
 ('dobj', ctypes.c_double),
 ('dgap', ctypes.c_double),
 ('rdgap', ctypes.c_double),
 ('mu', ctypes.c_double),
 ('mu_aff', ctypes.c_double),
 ('sigma', ctypes.c_double),
 ('lsit_aff', ctypes.c_int),
 ('lsit_cc', ctypes.c_int),
 ('step_aff', ctypes.c_double),
 ('step_cc', ctypes.c_double),
 ('solvetime', ctypes.c_double),
 ('fevalstime', ctypes.c_double),
 ('solver_id', ctypes.c_int * 8)
]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1), 
	(7, 4, 6, 26, 5, 4, 5, 1)
]