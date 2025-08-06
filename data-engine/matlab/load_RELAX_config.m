function RELAX_cfg = load_RELAX_config(yamlFilePath)
% LOADRELAXCONFIG Loads RELAX pipeline configuration from a YAML file.
%
%   RELAX_cfg = load_Relax_config(yamlFilePath) reads the YAML file specified
%   by yamlFilePath, loads its contents into a MATLAB structure, and then
%   performs necessary data type conversions (e.g., from cell arrays to
%   string arrays for specific fields) to ensure the structure matches
%   the format expected by the RELAX pipeline's internal scripts.
%
%   Input:
%     yamlFilePath -f Full path to the YAML configuration file (e.g., 'C:\mydata\RELAX_config.yaml').
%
%   Output:
%     RELAX_cfg - A MATLAB structure containing the configuration parameters,
%                 formatted correctly for the RELAX pipeline.
%
%   Dependencies:
%     This function requires a YAML parser for MATLAB (e.g., 'ReadYaml' function
%     from a toolbox like YAMLmatlab) to be available on the MATLAB path.

% Check if the YAML file exists
if ~isfile(yamlFilePath)
    error('loadRelaxConfig:FileNotFound', ...
          'YAML file not found: %s. Please ensure the file is in the correct location.', yamlFilePath);
end

% Read the YAML file into a MATLAB structure
try
    RELAX_cfg = readyaml(yamlFilePath);
    disp('YAML file loaded successfully into RELAX_cfg structure.');
catch ME
    error('loadRelaxConfig:YamlReadError', ...
          'Error loading YAML file (%s): %s', yamlFilePath, ME.message);
end

% --- Conversions ---

if isfield(RELAX_cfg, 'HEOGLeftpattern') && iscell(RELAX_cfg.HEOGLeftpattern)
    RELAX_cfg.HEOGLeftpattern = string(RELAX_cfg.HEOGLeftpattern);
    disp('Converted HEOGLeftpattern to a string array.');
end

if isfield(RELAX_cfg, 'HEOGRightpattern') && iscell(RELAX_cfg.HEOGRightpattern)
    RELAX_cfg.HEOGRightpattern = string(RELAX_cfg.HEOGRightpattern);
    disp('Converted HEOGRightpattern to a string array.');
end

% --- Example of additional raw print for testing (can be removed in final version) ---
disp(' ');
disp('--- Raw Data Types and Content from RELAX_cfg for Testing (within function) ---');
fields = fieldnames(RELAX_cfg);
for i = 1:length(fields)
    fieldName = fields{i};
    disp(['Field: ', fieldName]);
    disp(RELAX_cfg.(fieldName));
    disp(['  Class: ', class(RELAX_cfg.(fieldName))]);
    if iscell(RELAX_cfg.(fieldName)) && ~isempty(RELAX_cfg.(fieldName))
        disp(['  Class of first element (if cell): ', class(RELAX_cfg.(fieldName){1})]);
    elseif isstring(RELAX_cfg.(fieldName)) && ~isempty(RELAX_cfg.(fieldName))
        disp(['  Class of first element (if string array): ', class(RELAX_cfg.(fieldName)(1))]);
    end
end
disp('------------------------------------------------------------------');

end