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
% - outputs - column vector of length 175
function [outputs] = MPC_SOLVER(x0, xinit, all_parameters, num_of_threads)
    
    [output, ~, ~] = MPC_SOLVERBuildable.forcesCall(x0, xinit, all_parameters, num_of_threads);
    outputs = coder.nullcopy(zeros(175,1));
    outputs(1:7) = output.x01;
    outputs(8:14) = output.x02;
    outputs(15:21) = output.x03;
    outputs(22:28) = output.x04;
    outputs(29:35) = output.x05;
    outputs(36:42) = output.x06;
    outputs(43:49) = output.x07;
    outputs(50:56) = output.x08;
    outputs(57:63) = output.x09;
    outputs(64:70) = output.x10;
    outputs(71:77) = output.x11;
    outputs(78:84) = output.x12;
    outputs(85:91) = output.x13;
    outputs(92:98) = output.x14;
    outputs(99:105) = output.x15;
    outputs(106:112) = output.x16;
    outputs(113:119) = output.x17;
    outputs(120:126) = output.x18;
    outputs(127:133) = output.x19;
    outputs(134:140) = output.x20;
    outputs(141:147) = output.x21;
    outputs(148:154) = output.x22;
    outputs(155:161) = output.x23;
    outputs(162:168) = output.x24;
    outputs(169:175) = output.x25;
end
