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
classdef MPC_SOLVERBuildable < coder.ExternalDependency

    methods (Static)
        
        function name = getDescriptiveName(~)
            name = mfilename;
        end
        
        function b = isSupportedContext(context)
            b = context.isMatlabHostTarget();
        end
        
        function updateBuildInfo(buildInfo, cfg)
            buildablepath = fileparts(mfilename('fullpath'));
            [solverpath, foldername] = fileparts(buildablepath);
            [~, solvername] = fileparts(solverpath);
            % if the folder structure does not match to the interface folder, we assume it's the directory that contains the solver
            if(~strcmp(foldername, 'interface') || ~strcmp(solvername, 'MPC_SOLVER'))
                solverpath = fullfile(buildablepath, 'MPC_SOLVER');
            end
            solverInfo = struct();
            solverInfo.solvername = 'MPC_SOLVER';
            solverInfo.solverpath = solverpath;
            solverInfo.pythonClientFormat = true;
            solverInfo.useParallel = 1;
            solverInfo.isNLP = true;
            ForcesUpdateBuildInfo(buildInfo, cfg, solverInfo);
            postUpdateBuildInfoScript = [solverInfo.solvername, 'PostUpdateBuildInfo'];
            if exist(fullfile(buildablepath, [postUpdateBuildInfoScript, '.m']), 'file')
                postUpdateBuildInfo = str2func(postUpdateBuildInfoScript);
                postUpdateBuildInfo(buildInfo, cfg, solverInfo);
            end
        end
        
        function [output,exitflag,info] = forcesInitOutputsMatlab()
            infos_it = coder.nullcopy(zeros(1, 1));
            infos_it2opt = coder.nullcopy(zeros(1, 1));
            infos_res_eq = coder.nullcopy(zeros(1, 1));
            infos_res_ineq = coder.nullcopy(zeros(1, 1));
            infos_rsnorm = coder.nullcopy(zeros(1, 1));
            infos_rcompnorm = coder.nullcopy(zeros(1, 1));
            infos_pobj = coder.nullcopy(zeros(1, 1));
            infos_dobj = coder.nullcopy(zeros(1, 1));
            infos_dgap = coder.nullcopy(zeros(1, 1));
            infos_rdgap = coder.nullcopy(zeros(1, 1));
            infos_mu = coder.nullcopy(zeros(1, 1));
            infos_mu_aff = coder.nullcopy(zeros(1, 1));
            infos_sigma = coder.nullcopy(zeros(1, 1));
            infos_lsit_aff = coder.nullcopy(zeros(1, 1));
            infos_lsit_cc = coder.nullcopy(zeros(1, 1));
            infos_step_aff = coder.nullcopy(zeros(1, 1));
            infos_step_cc = coder.nullcopy(zeros(1, 1));
            infos_solvetime = coder.nullcopy(zeros(1, 1));
            infos_fevalstime = coder.nullcopy(zeros(1, 1));
            infos_solver_id = coder.nullcopy(zeros(8, 1));
            info = struct('it', infos_it,...
                          'it2opt', infos_it2opt,...
                          'res_eq', infos_res_eq,...
                          'res_ineq', infos_res_ineq,...
                          'rsnorm', infos_rsnorm,...
                          'rcompnorm', infos_rcompnorm,...
                          'pobj', infos_pobj,...
                          'dobj', infos_dobj,...
                          'dgap', infos_dgap,...
                          'rdgap', infos_rdgap,...
                          'mu', infos_mu,...
                          'mu_aff', infos_mu_aff,...
                          'sigma', infos_sigma,...
                          'lsit_aff', infos_lsit_aff,...
                          'lsit_cc', infos_lsit_cc,...
                          'step_aff', infos_step_aff,...
                          'step_cc', infos_step_cc,...
                          'solvetime', infos_solvetime,...
                          'fevalstime', infos_fevalstime,...
                          'solver_id', infos_solver_id);

            outputs_x01 = coder.nullcopy(zeros(7, 1));
            outputs_x02 = coder.nullcopy(zeros(7, 1));
            outputs_x03 = coder.nullcopy(zeros(7, 1));
            outputs_x04 = coder.nullcopy(zeros(7, 1));
            outputs_x05 = coder.nullcopy(zeros(7, 1));
            outputs_x06 = coder.nullcopy(zeros(7, 1));
            outputs_x07 = coder.nullcopy(zeros(7, 1));
            outputs_x08 = coder.nullcopy(zeros(7, 1));
            outputs_x09 = coder.nullcopy(zeros(7, 1));
            outputs_x10 = coder.nullcopy(zeros(7, 1));
            outputs_x11 = coder.nullcopy(zeros(7, 1));
            outputs_x12 = coder.nullcopy(zeros(7, 1));
            outputs_x13 = coder.nullcopy(zeros(7, 1));
            outputs_x14 = coder.nullcopy(zeros(7, 1));
            outputs_x15 = coder.nullcopy(zeros(7, 1));
            outputs_x16 = coder.nullcopy(zeros(7, 1));
            outputs_x17 = coder.nullcopy(zeros(7, 1));
            outputs_x18 = coder.nullcopy(zeros(7, 1));
            outputs_x19 = coder.nullcopy(zeros(7, 1));
            outputs_x20 = coder.nullcopy(zeros(7, 1));
            outputs_x21 = coder.nullcopy(zeros(7, 1));
            outputs_x22 = coder.nullcopy(zeros(7, 1));
            outputs_x23 = coder.nullcopy(zeros(7, 1));
            outputs_x24 = coder.nullcopy(zeros(7, 1));
            outputs_x25 = coder.nullcopy(zeros(7, 1));
            output = struct('x01', outputs_x01,...
                            'x02', outputs_x02,...
                            'x03', outputs_x03,...
                            'x04', outputs_x04,...
                            'x05', outputs_x05,...
                            'x06', outputs_x06,...
                            'x07', outputs_x07,...
                            'x08', outputs_x08,...
                            'x09', outputs_x09,...
                            'x10', outputs_x10,...
                            'x11', outputs_x11,...
                            'x12', outputs_x12,...
                            'x13', outputs_x13,...
                            'x14', outputs_x14,...
                            'x15', outputs_x15,...
                            'x16', outputs_x16,...
                            'x17', outputs_x17,...
                            'x18', outputs_x18,...
                            'x19', outputs_x19,...
                            'x20', outputs_x20,...
                            'x21', outputs_x21,...
                            'x22', outputs_x22,...
                            'x23', outputs_x23,...
                            'x24', outputs_x24,...
                            'x25', outputs_x25);
            
            exitflag = coder.nullcopy(0);
        end

        function [output,exitflag,info] = forcesCallWithParams(params)
            [output,exitflag,info] = MPC_SOLVERBuildable.forcesCall(params.x0, params.xinit, params.all_parameters, params.num_of_threads);
        end

        function [output,exitflag,info] = forcesCall(x0, xinit, all_parameters, num_of_threads)
            solvername = 'MPC_SOLVER';

            
            params = struct('x0', double(x0),...
                            'xinit', double(xinit),...
                            'all_parameters', double(all_parameters),...
                            'num_of_threads', uint32(num_of_threads));

            [output_c, exitflag_c, info_c] = MPC_SOLVERBuildable.forcesInitOutputsC();
            
            headerName = [solvername '.h'];
            coder.cinclude(headerName);
            coder.cinclude([solvername '_memory.h']);
            coder.cinclude([solvername '_adtool2forces.h']);
            % define memory pointer
            memptr = coder.opaque([solvername '_mem *'], 'HeaderFile', headerName);
            memptr = coder.ceval([solvername '_internal_mem'], uint32(0));
            % define solver input information (params, file and casadi)
            coder.cstructname(params, [solvername '_params'], 'extern', 'HeaderFile', headerName);
            fp = coder.opaque('FILE *', 'NULL', 'HeaderFile', headerName);
            % need define extern int solvername_adtool2forces(solvername_float *x, solvername_float *y, solvername_float *l, solvername_float *p, solvername_float *f, solvername_float *nabla_f, solvername_float *c, solvername_float *nabla_c, solvername_float *h, solvername_float *nabla_h, solvername_float *hess, solver_int32_default stage, solver_int32_default iteration);
            casadi = coder.opaque([solvername '_extfunc'],['&' solvername '_adtool2forces'],'HeaderFile',headerName);
            % define solver output information (output, exitflag, info)
            coder.cstructname(output_c,[solvername '_output'], 'extern', 'HeaderFile', headerName);
            coder.cstructname(info_c,[solvername '_info'], 'extern', 'HeaderFile', headerName);
            exitflag_c = coder.ceval([solvername '_solve'], coder.rref(params), ...
                                      coder.wref(output_c), coder.wref(info_c), ... 
                                      memptr, fp, casadi);
            
            [output, exitflag, info] = MPC_SOLVERBuildable.forcesInitOutputsMatlab();

            info.it = cast(info_c.it, 'like', info.it);
            info.it2opt = cast(info_c.it2opt, 'like', info.it2opt);
            info.res_eq = cast(info_c.res_eq, 'like', info.res_eq);
            info.res_ineq = cast(info_c.res_ineq, 'like', info.res_ineq);
            info.rsnorm = cast(info_c.rsnorm, 'like', info.rsnorm);
            info.rcompnorm = cast(info_c.rcompnorm, 'like', info.rcompnorm);
            info.pobj = cast(info_c.pobj, 'like', info.pobj);
            info.dobj = cast(info_c.dobj, 'like', info.dobj);
            info.dgap = cast(info_c.dgap, 'like', info.dgap);
            info.rdgap = cast(info_c.rdgap, 'like', info.rdgap);
            info.mu = cast(info_c.mu, 'like', info.mu);
            info.mu_aff = cast(info_c.mu_aff, 'like', info.mu_aff);
            info.sigma = cast(info_c.sigma, 'like', info.sigma);
            info.lsit_aff = cast(info_c.lsit_aff, 'like', info.lsit_aff);
            info.lsit_cc = cast(info_c.lsit_cc, 'like', info.lsit_cc);
            info.step_aff = cast(info_c.step_aff, 'like', info.step_aff);
            info.step_cc = cast(info_c.step_cc, 'like', info.step_cc);
            info.solvetime = cast(info_c.solvetime, 'like', info.solvetime);
            info.fevalstime = cast(info_c.fevalstime, 'like', info.fevalstime);
            info.solver_id = cast(info_c.solver_id, 'like', info.solver_id);

            output.x01 = cast(output_c.x01, 'like', output.x01);
            output.x02 = cast(output_c.x02, 'like', output.x02);
            output.x03 = cast(output_c.x03, 'like', output.x03);
            output.x04 = cast(output_c.x04, 'like', output.x04);
            output.x05 = cast(output_c.x05, 'like', output.x05);
            output.x06 = cast(output_c.x06, 'like', output.x06);
            output.x07 = cast(output_c.x07, 'like', output.x07);
            output.x08 = cast(output_c.x08, 'like', output.x08);
            output.x09 = cast(output_c.x09, 'like', output.x09);
            output.x10 = cast(output_c.x10, 'like', output.x10);
            output.x11 = cast(output_c.x11, 'like', output.x11);
            output.x12 = cast(output_c.x12, 'like', output.x12);
            output.x13 = cast(output_c.x13, 'like', output.x13);
            output.x14 = cast(output_c.x14, 'like', output.x14);
            output.x15 = cast(output_c.x15, 'like', output.x15);
            output.x16 = cast(output_c.x16, 'like', output.x16);
            output.x17 = cast(output_c.x17, 'like', output.x17);
            output.x18 = cast(output_c.x18, 'like', output.x18);
            output.x19 = cast(output_c.x19, 'like', output.x19);
            output.x20 = cast(output_c.x20, 'like', output.x20);
            output.x21 = cast(output_c.x21, 'like', output.x21);
            output.x22 = cast(output_c.x22, 'like', output.x22);
            output.x23 = cast(output_c.x23, 'like', output.x23);
            output.x24 = cast(output_c.x24, 'like', output.x24);
            output.x25 = cast(output_c.x25, 'like', output.x25);
            
            exitflag = cast(exitflag_c, 'like', exitflag);
        end
    end

    methods (Static, Access = private)
        function [output,exitflag,info] = forcesInitOutputsC()
            infos_it = coder.nullcopy(int32(zeros(1, 1)));
            infos_it2opt = coder.nullcopy(int32(zeros(1, 1)));
            infos_res_eq = coder.nullcopy(double(zeros(1, 1)));
            infos_res_ineq = coder.nullcopy(double(zeros(1, 1)));
            infos_rsnorm = coder.nullcopy(double(zeros(1, 1)));
            infos_rcompnorm = coder.nullcopy(double(zeros(1, 1)));
            infos_pobj = coder.nullcopy(double(zeros(1, 1)));
            infos_dobj = coder.nullcopy(double(zeros(1, 1)));
            infos_dgap = coder.nullcopy(double(zeros(1, 1)));
            infos_rdgap = coder.nullcopy(double(zeros(1, 1)));
            infos_mu = coder.nullcopy(double(zeros(1, 1)));
            infos_mu_aff = coder.nullcopy(double(zeros(1, 1)));
            infos_sigma = coder.nullcopy(double(zeros(1, 1)));
            infos_lsit_aff = coder.nullcopy(int32(zeros(1, 1)));
            infos_lsit_cc = coder.nullcopy(int32(zeros(1, 1)));
            infos_step_aff = coder.nullcopy(double(zeros(1, 1)));
            infos_step_cc = coder.nullcopy(double(zeros(1, 1)));
            infos_solvetime = coder.nullcopy(double(zeros(1, 1)));
            infos_fevalstime = coder.nullcopy(double(zeros(1, 1)));
            infos_solver_id = coder.nullcopy(int32(zeros(8, 1)));
            info = struct('it', infos_it,...
                          'it2opt', infos_it2opt,...
                          'res_eq', infos_res_eq,...
                          'res_ineq', infos_res_ineq,...
                          'rsnorm', infos_rsnorm,...
                          'rcompnorm', infos_rcompnorm,...
                          'pobj', infos_pobj,...
                          'dobj', infos_dobj,...
                          'dgap', infos_dgap,...
                          'rdgap', infos_rdgap,...
                          'mu', infos_mu,...
                          'mu_aff', infos_mu_aff,...
                          'sigma', infos_sigma,...
                          'lsit_aff', infos_lsit_aff,...
                          'lsit_cc', infos_lsit_cc,...
                          'step_aff', infos_step_aff,...
                          'step_cc', infos_step_cc,...
                          'solvetime', infos_solvetime,...
                          'fevalstime', infos_fevalstime,...
                          'solver_id', infos_solver_id);
                          
            outputs_x01 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x02 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x03 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x04 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x05 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x06 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x07 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x08 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x09 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x10 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x11 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x12 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x13 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x14 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x15 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x16 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x17 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x18 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x19 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x20 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x21 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x22 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x23 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x24 = coder.nullcopy(double(zeros(7, 1)));
            outputs_x25 = coder.nullcopy(double(zeros(7, 1)));
            output = struct('x01', outputs_x01,...
                            'x02', outputs_x02,...
                            'x03', outputs_x03,...
                            'x04', outputs_x04,...
                            'x05', outputs_x05,...
                            'x06', outputs_x06,...
                            'x07', outputs_x07,...
                            'x08', outputs_x08,...
                            'x09', outputs_x09,...
                            'x10', outputs_x10,...
                            'x11', outputs_x11,...
                            'x12', outputs_x12,...
                            'x13', outputs_x13,...
                            'x14', outputs_x14,...
                            'x15', outputs_x15,...
                            'x16', outputs_x16,...
                            'x17', outputs_x17,...
                            'x18', outputs_x18,...
                            'x19', outputs_x19,...
                            'x20', outputs_x20,...
                            'x21', outputs_x21,...
                            'x22', outputs_x22,...
                            'x23', outputs_x23,...
                            'x24', outputs_x24,...
                            'x25', outputs_x25);
            exitflag = coder.nullcopy(int32(0));
        end
    end

    
end
