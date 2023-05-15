function y=white_noise_band_ramp(T,fs,fl,fh,dblRampDur)

%Function to generate a band-limited white noise with a ramp up and off
%phase
%
%T: duration (s)
%fs: sampling frequency (Hz)
%fl: low frequency bound (Hz)
%fh: high frequency bound (Hz)
%dblRampDur: duration of ramp phase in (s)

N=T*fs; %number of samples
[B,A]=butter(6,[fl fh]/(fs/2));

% Generate random number for each sample
x=(rand(1,N)*2)-1;

% Apply filter
x=filtfilt(B,A,x);

vecRamps = ones(1,N);
nRamp = round(dblRampDur*fs); % Number of samples in ramp

% Creat ramps
vecRamps(1:nRamp) = linspace(0,1,nRamp);
vecRamps(end-nRamp+1:end) = linspace(1,0,nRamp);

% Multiply white noise with ramps
y = x.* vecRamps;


