% Simple demo to use Ensemblex method

% Generate synthetic data
raster = Get_Synthetic_Ensembles();
load test_raster_2022_03_01_10_02_44.mat raster

% Get ensembles from raster
analysis = Get_Ensembles(raster);

% Plot raster with ensembles identified
Plot_Ensemble_Raster(analysis,28,true,true, true)
Plot_Ensemble_Raster(analysis,28,false, true)

Plot_Ensemble_Raster(analysis,28,true)





figure
plot(daq_data.voltage(:,5))