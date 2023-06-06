clear 
clc

%% using iQSM and iQFM for qsm and local field recon is quite easy; 

phpath = 'ph_demo.nii'; % see our repo for the address to the demo data; 
parampath = 'params.mat'; 
maskpath = 'mask_demo.nii'; 

mag_path = ''; % default for no mag; 

ResultPath = './'; 

%% QSM recon
Recon_iQSM(phpath, parampath, maskpath, magpath,ResultPath)
% now you see iQSM.nii in Resultpath; 

Recon_iQFM(phpath, parampath, maskpath, magpath,ResultPath)
% now you see iQFM.nii in Resultpath; 

