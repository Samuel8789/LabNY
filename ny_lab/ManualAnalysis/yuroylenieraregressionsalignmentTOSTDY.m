% voltagetrace=[];
shift=-29;
scaling_factor=0.05625055928411633;
trace=voltagetrace;
frame_times=[]
ca_traces=xx'

shift = round(shift);

[d1, d2] = size(trace);
num_samp = max([d1, d2]);
num_chan = min([d1, d2]);

if scaling_factor > 1 || scaling_factor < 1
    % insert missing
    x = ((1:num_samp)-1)*scaling_factor+1;
    xq = 1:1:num_samp*scaling_factor;
    scaled_trace = interp1(x,trace,xq);
    if num_chan == 1
        scaled_trace = scaled_trace';
    end
elseif scaling_factor == 1
    scaled_trace = trace;
end

% now adjust for the shift
if shift < 0
    % remove voltage data
    shifted_scaled_trace = scaled_trace(1-shift:end,:);
elseif shift > 0
    % insert voltage data
    pad_val = scaled_trace(1);
    temp_padding = ones(shift,num_chan)*pad_val;
    if d1<d2
        shifted_scaled_trace = [temp_padding', scaled_trace];
    else
        shifted_scaled_trace = [temp_padding; scaled_trace];
    end
elseif shift == 0
    shifted_scaled_trace = scaled_trace;
end

figure;
plot(1:numel(shifted_scaled_trace),shifted_scaled_trace);
hold on;
plot(  ca_traces);
title('Aligned traces; Is it good? [Y/N]');
legend('DAQ voltage trace', 'Alignment Channel');


ca_traces_copy=ca_traces
frame_times=ts'*1000;
volt_data=rescale(trace,0,1);
v_times = 1:size(volt_data,1);
figure; hold on;
plot(v_times, volt_data);
plot(frame_times,ca_traces);
thresh = [.3 .8];
% 
if ~exist('thresh', 'var')
    thresh = [.3 .8];
end
%%
on_times_low = [];
on_times_high = [];
for n_fr = 2:numel(ca_traces)
    if and(ca_traces(n_fr) > thresh(1), ca_traces(n_fr-1) < thresh(1))
        on_times_low = [on_times_low; n_fr-1];
    end
    if and(ca_traces(n_fr) > thresh(2), ca_traces(n_fr-1) < thresh(2))
        on_times_high = [on_times_high; n_fr];
    end
end

while numel(on_times_low) ~= numel(on_times_high)
    figure; plot(ca_traces); axis tight;
    title(sprintf('threshhold [%.1f %.1f] failed, select manual (2 clicks)', thresh(1), thresh(2)));
    [~,thresh] = ginput(2);
    close;
    
    on_times_low = [];
    on_times_high = [];
    for n_fr = 2:numel(ca_traces)
        if and(ca_traces(n_fr) > thresh(1), ca_traces(n_fr-1) < thresh(1))
            on_times_low = [on_times_low; n_fr-1];
        end
        if and(ca_traces(n_fr) > thresh(2), ca_traces(n_fr-1) < thresh(2))
            on_times_high = [on_times_high; n_fr];
        end
    end
end

pulse_times_on = (frame_times(on_times_high) + frame_times(on_times_low))/2 - mean(diff(frame_times))/2;

%%
off_times_low = [];
off_times_high = [];
for n_fr = 2:numel(ca_traces)
    if and(ca_traces(n_fr) < thresh(2), ca_traces(n_fr-1) > thresh(2))
        off_times_low = [off_times_low; n_fr-1];
    end
    if and(ca_traces(n_fr) < thresh(1), ca_traces(n_fr-1) > thresh(1))
        off_times_high = [off_times_high; n_fr];
    end
end
pulse_times_off = (frame_times(off_times_high) + frame_times(off_times_low))/2 - mean(diff(frame_times))/2;

%%
on_times_low = [];
on_times_high = [];
for n_t = 2:numel(volt_data)
    if and(volt_data(n_t) > thresh(1), volt_data(n_t-1) < thresh(1))
        on_times_low = [on_times_low; n_t-1];
    end
    if and(volt_data(n_t) > thresh(2), volt_data(n_t-1) < thresh(2))
        on_times_high = [on_times_high; n_t];
    end
end
pulse_times_on_volt = (on_times_high + on_times_low)/2 - .5;

%%
off_times_low = [];
off_times_high = [];
for n_t = 2:numel(volt_data)
    if and(volt_data(n_t) < thresh(2), volt_data(n_t-1) > thresh(2))
        off_times_low = [off_times_low; n_t-1];
    end
    if and(volt_data(n_t) < thresh(1), volt_data(n_t-1) > thresh(1))
        off_times_high = [off_times_high; n_t];
    end
end
pulse_times_off_volt = (off_times_high + off_times_low)/2 - .5;

%%
%x_on = [ones(size(pulse_times_on_volt)), pulse_times_on_volt]\pulse_times_on;
%x_off = [ones(size(pulse_times_off_volt)), pulse_times_off_volt]\pulse_times_off;

pulse_comb = [pulse_times_on; pulse_times_off];
pulse_comb_volt = [pulse_times_on_volt; pulse_times_off_volt];
x_comb = [ones(size(pulse_comb_volt)), pulse_comb_volt]\pulse_comb;

%%
scaling_factor = x_comb(2);
shift = x_comb(1);

%%

figure; hold on;
plot(frame_times, ca_traces)
plot(pulse_times_on, zeros(size(pulse_times_on)), 'o')
plot(pulse_times_off, ones(size(pulse_times_on)), 'o')
plot((1:numel(volt_data))*x_comb(2)+x_comb(1), volt_data) % 
plot((1:numel(volt_data)), volt_data) % 

