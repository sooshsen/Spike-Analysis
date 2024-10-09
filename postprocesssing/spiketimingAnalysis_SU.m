%% this script calculates relative distance from trigger onset to the single unit spikes in the experiment p2e1_1

%% trigger
% filepath_trg = '/home/ssenapat/groups/PrimNeu/Sushmita/from_system/comparativeNeurobio_hiwi/Sushimita/results_elfie_p2/matFiles/use_these_to_get_correct_results_and_plots/final_sorted_tonestarts.mat';
% load(filepath_trg);     % sorted_tone_onset

% trigger from python
filepath_trg = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/SUA/server/Elfie/p2/p2_1_1/trigger_onset_for_matlab.npy';
sorted_tone_onset = readNPY(filepath_trg);

% load the single unit spike timing file for this experiment
filepath_spikes = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/SUA/server/p2/combined123/spiketiming_perExp/spiketiming_exp_1.npy';
spiketimes = readNPY(filepath_spikes);

% arrange spiketimes in increasing order of unitID
spiketimes_unit = sortrows(spiketimes,3);

% store all units present in this experiment as a list
unitIDs = unique(spiketimes_unit(:,3));

% store spikes timepoints per experiment from each unit as a cell
for ii=1:length(unitIDs)
    onsetpoint{ii} = spiketimes_unit(spiketimes_unit(:,3) == unitIDs(ii),2)';
end



% tone onsets SORTED based on frequency
sorted_peakdiff_time = [];
tic
for id = 1:length(onsetpoint)
	unit_per_tone =  spikespertoneDetection(onsetpoint{1,id}, sorted_tone_onset); 
	sorted_peakdiff_time = [sorted_peakdiff_time, unit_per_tone];
end
toc


%% plot raster plots per unit
savehere = '/home/ssenapat/groups/PrimNeu/Final_exps_spikes/SUA/server/p2/matlab_analysis_results_p2_e1_1';

tic
for unit=1:length(unitIDs)
	clear FigHandle
	one_unit = sorted_peakdiff_time(:,unit);
	FigHandle = plotSpikeToneRel_SU(unitIDs(unit), one_unit);

	saveas(FigHandle, fullfile(savehere,strcat('unit',num2str(unitIDs(unit)),'_raster.png')));
%     saveas(FigHandle, fullfile(savehere,strcat('Channel30_py_spikeonsets_thresh4.png')));

end
toc
