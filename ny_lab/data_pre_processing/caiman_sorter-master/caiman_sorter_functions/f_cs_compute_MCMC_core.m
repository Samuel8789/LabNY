function [SAMP, spikeRaster] = f_cs_compute_MCMC_core(y, params)

num_samples = params.Nsamples;
num_frames = numel(y);

% for some reason it breaks when Y is zeros, so remove that part
y_nonzero = find(y~=0);
sig_start = y_nonzero(1);

spikeRaster = zeros(num_samples, num_frames);

SAMP.process_ok = 0;
try
    SAMP = cont_ca_sampler(y(sig_start:end),params);
    % need to reconstructing cropped signal, cuz there is Cin parameter for initial c
    SAMP.C_rec = extract_C_YS(SAMP,y(sig_start:end)); % 
    
    for rep = 1:num_samples
        temp = ceil(SAMP.ss{rep}) + sig_start - 1;
        spikeRaster(rep,temp) = 1;
    end
    SAMP.process_ok = 1;
catch
    try
        warning('Error in MCMC, try splitting into 2 if too large')
        second_start = floor((numel(y)-sig_start)/2)+sig_start;
        samp1 = cont_ca_sampler(y(sig_start:second_start),params);
        samp2 = cont_ca_sampler(y((second_start+1):end),params);

        samp1.C_rec = extract_C_YS(samp1,y(sig_start:second_start));
        samp2.C_rec = extract_C_YS(samp2,y((second_start+1):end));

        for rep = 1:num_samples
            temp1 = ceil(samp1.ss{rep})+sig_start-1;
            temp2 = ceil(samp2.ss{rep})+second_start;
            spikeRaster(rep,temp1) = 1;
            spikeRaster(rep,temp2) = 1;
        end

        SAMP.samp1 = samp1;
        SAMP.samp2 = samp2;
        SAMP.C_rec = [samp1.C_rec, samp2.C_rec];
        SAMP.error  = 'data needed to be split in 2 for MCMC';
        SAMP.second_start = second_start;
        SAMP.process_ok = 1;
    catch
        warning('Error in MCMC, unable to process, skipping cell')
        SAMP.error = 'MCMC failed';
    end
end

SAMP.sig_start = sig_start;

end