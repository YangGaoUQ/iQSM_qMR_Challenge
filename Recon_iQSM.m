function Recon_iQSM(PhasePath, paramspath, MaskPath, MagPath, ReconDir)
%% iQSM reconstruction for data from ChinaJapan hospital
% PhasePath: path for raw phase data; e.g., ph.nii;
% paramspath: reconstruction parameters including TE, vox, B0, and z_prjs;
% e.g., params.mat;
% MaskPath (optional): path for bet mask;
% MagPath(optional): path for magnitude;
% ReconDir (optional): path for reconstruction saving;

% example usage: Recon_iQFM('./ph.nii', 'params.mat', './mask.nii', './mag.nii', './');

%------------- Assume all your data is in NIFTI format--------------------%
%
% for more deep learning based algorithms for background removal and dipole
% inversion, plese
% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMR
% (2) download demo data and checkpoints here: https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0
%
% for more conventional algorithms, e.g., phase combination, phase unwrapping, please
% download or clone github repo for Hongfu's QSM toolbox: https://github.com/sunhongfu/QSM

%  Authors: Yang Gao [1,2]
%  yang.gao@csu.edu.cn
%  [1]: Central South University, China, Lecturer
%  [2]: University of Queensland, Australia, Honorary Fellow
%  22 Mar, 2023

%  Reference: Gao Y, et al. Instant tissue field and magnetic susceptibility mapping
%  from MRI raw phase using Laplacian enhanced deep neural networks. Neuroimage. 2022
%  doi: 10.1016/j.neuroimage.2022.119410. Epub 2022 Jun 23. PMID: 35753595.

%------------------- data preparation guide ------------------------------%

% 1. phase evolution type:
% The relationship between the phase data and filed pertubation (delta_B)
% is assumed to satisfy the following equation:
% "phase = -delta_B * gamma * TE"
% Therefore, if your phase data is in the format of "phase = delta_B * gamma * TE;"
% it will have to be preprocessed by multiplication by -1;

% 2. For Ultra-high resolutin data:
% it is recommended that the phase data of ultra-high resolution (higher
% than 0.7 mm) should be interpoloated into 1 mm for better reconstruction results.

% created 11.08, 2022
% last modified 01.25, 2022
% last modified 05.06, 2023
% lateast 06.02, 2023


if ~exist('PhasePath','var') || isempty(PhasePath)
    error('Please input the path for raw phase data!')
end

if ~exist('ReconDir','var') || isempty(ReconDir)
    ReconDir = './';  %% where to save reconstruction output
else
    ReconDir = dir(ReconDir).folder;
end


%% Set your own data paths and parameters
%% Set your own data paths and parameters
deepMRI_root = '~/Downloads/iQSM_qMR_Challenge/'; % where deepMRI git repo is downloaded/cloned to
CheckPoints_folder = '~/Downloads/iQSM_qMR_Challenge/PythonCodes/Evaluation/checkpoints';
PyFolder = '~/Downloads/iQSM_qMR_Challenge/PythonCodes/Evaluation/'; 

KeyWord = 'iQSM_original'; % original iQSM method, to be extended to iQSM+; 

checkpoints  = sprintf('%s/%s/', CheckPoints_folder ,KeyWord);
InferencePath = sprintf('%s/%s/Inference_iQSMplus.py', PyFolder, KeyWord);

if ~exist('paramspath','var') || isempty(paramspath)
    error('Please input the params file!')
end

%% load recon parameters; 
load(paramspath);

Eroded_voxel = 3; %% brain erosion for 3 voxels, or 0 for whole head recon;

%% add MATLAB paths
addpath(genpath([deepMRI_root,'/iQSM_fcns/']));  % add necessary utility function for saving data and echo-fitting;
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;

%% read data

%% 1. read in data
sf = 1;  %% for cooridinates mismatch; set as -1 for ChinaJapan data;
nii = load_nii(PhasePath);
phase = nii.img;
phase = sf * phase;

% interpolate the phase to isotropic
imsize = size(phase);
if length(imsize) == 3
    imsize(4) = 1;
end
imsize2 = [round(imsize(1:3).*vox/min(vox)), imsize(4)];

vox2 = imsize(1:3).*vox ./ imsize2(1:3);

vox2 = round(vox2 * 100) / 100; %% only keep 2 floating points precesion;

interp_flag = ~isequal(imsize,imsize2);

if interp_flag
    for echo_num = 1:imsize(4)
        phase2(:,:,:,echo_num) = angle(imresize3(exp(1j*phase(:,:,:,echo_num)),imsize2(1:3)));
    end
    phase = phase2;
    clear phase2
end

if ~ exist('MagPath','var') || isempty(MagPath)
    mag = ones(imsize2);
else
    nii = load_nii(MagPath);
    mag = nii.img;
    % interpolate the mag to isotropic
    if interp_flag
        for echo_num = 1:imsize(4)
            mag2(:,:,:,echo_num) = imresize3(mag(:,:,:,echo_num),imsize2(1:3));
        end
        mag = mag2;
        clear mag2
    end
end

if ~ exist('MaskPath','var') || isempty(MaskPath)
    mask = ones(imsize2(1:3));
else
    nii = load_nii(MaskPath);
    mask = nii.img;
    % interpolate the mask to isotropic
    if interp_flag
        mask = imresize3(mask,imsize2(1:3));
    end
end

%% mkdir for output folders
if ~exist(ReconDir, 'dir')
    mkdir(ReconDir)
end

%% recon starts;
tmp_phase = ZeroPadding(phase, 16);
[mask, pos] = ZeroPadding(mask, 16);

mask_eroded = Save_Input_iQSMplus(tmp_phase, mask, TE', B0, Eroded_voxel, z_prjs, vox, ReconDir);

mask_eroded = ZeroRemoving(mask_eroded, pos);

% Call Python script to conduct the reconstruction; use python API to run iQSM on the demo data
PythonRecon(InferencePath, [ReconDir,'/Network_Input.mat'], ReconDir, checkpoints);

%% load reconstruction data and save as NIFTI
load([ReconDir,'/iQSM.mat']);

pred_chi = squeeze(pred_chi);

if length(size(pred_chi)) ~= 3
    pred_chi = permute(pred_chi, [2, 3, 4, 1]);
end

chi = ZeroRemoving(pred_chi, pos);

clear tmp_phase;

%% save results of all echoes before echo fitting
nii = make_nii(chi, vox2);
save_nii(nii, [ReconDir, filesep, 'iQSM_all_echoes.nii']);

%% magnitude weighted echo-fitting and save as NIFTI

if imsize(4) > 1
    for echo_num = 1 : imsize(4)
        chi(:,:,:,echo_num) = TE(echo_num) .* chi(:,:,:,echo_num);
    end

    chi_fitted = echofit(chi, mag, TE);
else
    chi_fitted = chi;
end

if interp_flag

%     nii = make_nii(chi_fitted, vox2);
%     save_nii(nii, [ReconDir, 'iQFM_interp_echo_fitted.nii']);
    mask_eroded  = imresize3(mask_eroded ,imsize(1:3));
    % back to original resolution if anisotropic
    chi_fitted = imresize3(chi_fitted,imsize(1:3));

    chi_fitted = chi_fitted .* mask_eroded;
end

nii = make_nii(chi_fitted, vox);
save_nii(nii, [ReconDir,'/iQSM.nii']);

delete([ReconDir,'/Network_Input.mat']);
delete([ReconDir,'/iQSM.mat']);

end

