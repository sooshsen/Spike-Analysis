% after spike onset detection in python
% make matlab compatible file for individual experiments
% @sushmitaS16
clear;
clc;

file_loc = path/to/spikeonsets.npy;

for chan=1:384
    py_nomenclature_chan = chan-1;
    onsetpoint{chan} = readNPY(fullfile(file_loc,strcat('spikeonsets_channel_',num2str(py_nomenclature_chan),'.npy')))';
end

save fullfile(file_loc,strcat('spikeonsetpoint')) onsetpoint
