% MPC_SOLVER : A fast customized optimization solver.
% 
% Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.
% 
% 
% This software is intended for simulation and testing purposes only. 
% Use of this software for any commercial purpose is prohibited.
% 
% This program is distributed in the hope that it will be useful.
% EMBOTECH makes NO WARRANTIES with respect to the use of the software 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
% PARTICULAR PURPOSE. 
% 
% EMBOTECH shall not have any liability for any damage arising from the use
% of the software.
% 
% This Agreement shall exclusively be governed by and interpreted in 
% accordance with the laws of Switzerland, excluding its principles
% of conflict of laws. The Courts of Zurich-City shall have exclusive 
% jurisdiction in case of any dispute.
% 
% [OUTPUTS] = MPC_SOLVER(INPUTS) solves an optimization problem where:
% Inputs:
% - xinit - matrix of size [4x1]
% - x0 - matrix of size [105x1]
% - all_parameters - matrix of size [390x1]
% - num_of_threads - scalar
% Outputs:
% - x01 - column vector of length 7
% - x02 - column vector of length 7
% - x03 - column vector of length 7
% - x04 - column vector of length 7
% - x05 - column vector of length 7
% - x06 - column vector of length 7
% - x07 - column vector of length 7
% - x08 - column vector of length 7
% - x09 - column vector of length 7
% - x10 - column vector of length 7
% - x11 - column vector of length 7
% - x12 - column vector of length 7
% - x13 - column vector of length 7
% - x14 - column vector of length 7
% - x15 - column vector of length 7
function [x01, x02, x03, x04, x05, x06, x07, x08, x09, x10, x11, x12, x13, x14, x15] = MPC_SOLVER(xinit, x0, all_parameters, num_of_threads)
    
    [output, ~, ~] = MPC_SOLVERBuildable.forcesCall(xinit, x0, all_parameters, num_of_threads);
    x01 = output.x01;
    x02 = output.x02;
    x03 = output.x03;
    x04 = output.x04;
    x05 = output.x05;
    x06 = output.x06;
    x07 = output.x07;
    x08 = output.x08;
    x09 = output.x09;
    x10 = output.x10;
    x11 = output.x11;
    x12 = output.x12;
    x13 = output.x13;
    x14 = output.x14;
    x15 = output.x15;
end
