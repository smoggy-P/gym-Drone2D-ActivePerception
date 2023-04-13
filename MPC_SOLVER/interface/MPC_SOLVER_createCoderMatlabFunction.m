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
% MPC_SOLVER_CREATECODERMATLABFUNCTION(MODELNAME, BLOCKNAME)
%
% This function generates a Simulink Model with Fixed-step solver type named MODELNAME to add the MPC_SOLVER FORCESPRO solver as 
% a MATLAB Function Simulink Block named BLOCKNAME so that it can be used for Simulink simulation and 
% Simulink Code Generation.
%
% MPC_SOLVER_CREATECODERMATLABFUNCTION(MODELNAME, BLOCKNAME, USECOMPACTBLOCK)
% can be used to generate a MATLAB Function Simulink Block with compact inputs and outputs.
%
% MPC_SOLVER_CREATECODERMATLABFUNCTION(MODELNAME, BLOCKNAME, USECOMPACTBLOCK, USEEXISTINGMODEL)
% can be used if the selected Simulink Model already exists in order to add the Simulink Block to it.
%
% MPC_SOLVER_CREATECODERMATLABFUNCTION(MODELNAME, BLOCKNAME, USECOMPACTBLOCK, USEEXISTINGMODEL, OPENSIMULINKMODEL)
% can be used to select whether to open the simulink model.
function MPC_SOLVER_createCoderMatlabFunction(modelname, blockname, useCompactBlock, useExistingModel, openSimulinkModel)

    if nargin < 1 || isempty(modelname)
        modelname = 'MPC_SOLVER_model';
    end
    if ~isvarname(modelname)
        error('Modelname must be a valid variable name');
    end
    if nargin < 2 || isempty(blockname)
        blockname = 'MPC_SOLVER_block';
    end
    if ~isvarname(blockname)
        error('Blockname must be a valid variable name');
    end
    if nargin < 3
        useCompactBlock = false;
    end
    if ~islogical(useCompactBlock)
        error('useCompactBlock must be a bool');
    end
    if nargin < 4
        useExistingModel = false;
    end
    if ~islogical(useExistingModel)
        error('useExistingModel must be a bool');
    end
    if nargin < 5
        openSimulinkModel = true;
    end
    if ~islogical(openSimulinkModel)
        error('openSimulinkModel must be a bool');
    end
    
    function_name = 'MPC_SOLVER_coderFunction.m';
    position = [170, 99, 650, 435];
    parameter_sizes = struct('xinit', '[4,1]',...
                             'x0', '[105,1]',...
                             'all_parameters', '[225,1]',...
                             'num_of_threads', '[1,1]');
    if useCompactBlock
        function_name = 'MPC_SOLVER_coderFunctioncompact.m';
        position = [170, 99, 650, 248];
        parameter_sizes = struct('xinit', '[4,1]',...
                                 'x0', '[105,1]',...
                                 'all_parameters', '[225,1]',...
                                 'num_of_threads', '[1,1]');
    end
    
    result = exist(modelname, 'file');
    if result ~= 4 && result ~= 0
        error('%s exists but is not a Simulink model. Please use a different name.', modelname);
    end
    if result == 0
        new_system(modelname);
        set_param(modelname, 'SolverType', 'Fixed-step')
    elseif ~useExistingModel
        error('Simulink Model %s already exists. Please call the script again with the parameter useExistingModel set to true to add MATLAB Function Simulink Block to existing model.', modelname);
    end
    if openSimulinkModel
        open_system(modelname);
    end

    blockpath = [modelname, '/', blockname];
    blockExists = true;
    try
        get_param(blockpath, 'ObjectParameters');
    catch
        blockExists = false;
    end
    if blockExists
        error('Blockname %s already exists in Simulink Model %s. Please select a different blockname', blockname, modelname);
    end
    add_block('simulink/User-Defined Functions/MATLAB Function', blockpath);
    set_param(blockpath, 'Position', position);

    cur_dir = fileparts(mfilename('fullpath'));
    script_filepath = fullfile(cur_dir, function_name);
    try
        blockconfig = find(Simulink.Root,'-isa','Stateflow.EMChart', 'Path', blockpath);
        blockconfig.Script = fileread(script_filepath);
        parameters = fieldnames(parameter_sizes);
        for i = 1:length(parameters)
            inputportconfig = find(Simulink.Root,'-isa','Stateflow.Data', 'Path', blockpath, 'Name', parameters{i});
            inputportconfig.Props.Array.Size = parameter_sizes.(parameters{i});
        end
        blockconfig.Locked = 1;
    catch
        error('This Simulink version does not support programmatic editing of the script of the MATLAB Function Simulink Block. Please manually create a MATLAB Function block, copy the script from ''%s'' and set the following dimensions for the input ports:\n%s.', script_filepath, printCharStruct(parameter_sizes));
    end
    save_system(modelname);
end

function message = printCharStruct(struct_var)
    message = '';
    fields = fieldnames(struct_var);
    for i = 1:length(fields)
        message = sprintf('%s''%s'': ''%s''\n', message, fields{i}, struct_var.(fields{i}));
    end
end
