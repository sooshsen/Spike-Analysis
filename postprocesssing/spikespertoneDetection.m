function onset_spikeVStone = spikespertoneDetection(channeldata, toneonset)
	% for each of the onset points of tones, we compare it to the spike onsets obtained before
    %% for single unit analysis, each channeldata corresponds to a unitdata
	% Inputs: channeldata - spike onset index data from one channel (n spikes)
	%		toneonset - tone onset index data (m tones)
	%
	% Outputs: matrix of how far each spike onset is from each tone onset (m*n array)
	%

	% for each tone onset, check spike_onset(:) - tone_onset(i) 
 	win_on = -0.5*60*1000;	%in ms
 	win_off = 1.5*60*1000;

	%convert the index to time data
	samplingRate = 30;	%kHz

	channeldata = channeldata./samplingRate;		%gives us time in milliseconds
	toneonset = toneonset./samplingRate;


	for count = 1:length(toneonset)
		
		onsetdiff_index = channeldata - toneonset(count);

 		index_window = find(onsetdiff_index>=win_on & onsetdiff_index<win_off);
 		onset_spikeVStone{count,1} = onsetdiff_index(1,index_window);


 		clear index_window

	end

	clear count

end
