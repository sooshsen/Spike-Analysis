function spikefig = plotSpikeToneRel_SU(unitNumber, unit_per_tone)
	% generates  a raster plot to show the relation between spike onset timing for each unit and the trigger tone onset
	% NOTE that tones are arranged here in increasing order of frequency

	% plotting spikes corresponding to one tone a time, i.e. n-spikes due to
    % tone1, then z-spikes due to tone2 etc. for a single unit


	spikefig = figure('visible','off');

    % get rid of any cell which is empty
%     loc=cellfun('isempty', channel_per_tone);
%     channel_per_tone(loc,:)=[];

	for i = 1:length(unit_per_tone)
        hold on;
        if isempty(unit_per_tone{i,:}) == 0 
		    fig = plot(unit_per_tone{i,:}, i*ones(1,length(unit_per_tone{i,:})), 'o');
		    fig.MarkerSize = 1;
		    fig.MarkerFaceColor = [0 0 0];
		    fig.MarkerEdgeColor = [0 0 0];
        end
	end

	xlim([-100 300]);	% tc: range of 100 milliseconds before tone onset and 300 milliseconds after tone onset
%     xlim([-3*1000 1.5*60*1000]);    % music: range of 3000ms before and 1.5s after music trigger
%     xlim([-2*1000 20*1000]);
	daspect([1 0.5 2]);     % for music : comment it
	xline(0, '--r');
	title('Unit',num2str(unitNumber));
	xlabel('time (in milliseconds)');
	ylabel('tone');

end


