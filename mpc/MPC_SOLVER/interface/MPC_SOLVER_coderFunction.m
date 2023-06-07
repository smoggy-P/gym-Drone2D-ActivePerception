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
% - x0 - matrix of size [175x1]
% - xinit - matrix of size [4x1]
% - all_parameters - matrix of size [700x1]
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
% - x16 - column vector of length 7
% - x17 - column vector of length 7
% - x18 - column vector of length 7
% - x19 - column vector of length 7
% - x20 - column vector of length 7
% - x21 - column vector of length 7
% - x22 - column vector of length 7
% - x23 - column vector of length 7
% - x24 - column vector of length 7
% - x25 - column vector of length 7
function [x01, x02, x03, x04, x05, x06, x07, x08, x09, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25] = MPC_SOLVER(x0, xinit, all_parameters, num_of_threads)
    
    [output, ~, ~] = MPC_SOLVERBuildable.forcesCall(x0, xinit, all_parameters, num_of_threads);
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
    x16 = output.x16;
    x17 = output.x17;
    x18 = output.x18;
    x19 = output.x19;
    x20 = output.x20;
    x21 = output.x21;
    x22 = output.x22;
    x23 = output.x23;
    x24 = output.x24;
    x25 = output.x25;
end
