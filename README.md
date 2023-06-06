# Yang's iQSM MATLAB toolbox for ISMRM 2023 quantitative MRI Open Challenge

- This repo is a matlab toolbox for our paper "Instant tissue field and magnetic susceptibility mapping from MRI raw phase using Laplacian enhanced deep neural networks";

* The iQSM (QSM reconstruction from raw phase) and iQFM (local field recon from raw phase) were encapsulated into two matlab function files, which are easy to use;

- This code was built and tested on win10 system with Nvdia A4000, ubuntu with Nvidia A6000, and macos12.0.1 (on CPU);

* Currently, the performance of iQSM will decrease on ultra-high resolution data (0.7 mm isotropic or higher) due to training data limitation; an enhanced version, iQSM+, will be updated soon later next month. please visit, https://github.com/sunhongfu/deepMRI for our latest work, including DL-based QSM acquisition acceleration, dipole inversion, backgroun field removal, and phase unwrapping. 

[demo data for a quick start](https://drive.google.com/file/d/1-rKqddWCujQ9MYnngtA7W0hVMM2tEg3p/view?usp=sharing) &nbsp;  | &nbsp;  [our paper @ NeuroImage (full paper)](https://www.sciencedirect.com/science/article/pii/S1053811922005274)

- See https://github.com/sunhongfu/deepMRI/tree/master/iQSM or contact Yang (yang.gao@csu.edu.cn or yang.gao@uq.edu.au) for more details about network training. 

# Content

- [ Overview](#head1)
  - [(1) Overall Framework](#head2)
  - [(2) Representative Results](#head3)
- [ Manual](#head4)
  - [Requirements](#head5)
  - [Quick Start](#head6)

# <span id="head1"> Overview </span>

## <span id="head2">(1) Overall Framework </span>

![Whole Framework](https://www.dropbox.com/s/7bxkyu1utxux76k/Figs_1.png?raw=1)
Fig. 1: Overview of iQFM and iQSM framework using the proposed LoT-Unet architecture, composed of a tailored Lap-Layer and a 3D residual Unet.

## <span id="head3">(2) Representative Results </span>

![Representative Results](https://www.dropbox.com/s/9jt391q22sgber6/Figs_2.png?raw=1)
Fig. 2: Comparison of different QSM methods on three ICH patients. Susceptibility images of two orthogonal views are illustrated for each subject. Red arrows point to the artifacts near the hemorrhage sources in different QSM reconstructions.

# <span id="head4"> Manual </span>

## <span id="head5"> Requirements </span>

    - Python 3.7 or later
    - NVDIA GPU (CUDA 10.0 or higher)
    - Anaconda Navigator (4.6.11) for Pytorch Installation
    - Pytorch 1.8 or later
    - MATLAB 2017b or later
    - BET tool from FSL tool box (optional)

## <span id="head6"> Quick Start (on demo data) </span>

1. Clone this repository.

2. Install prerequisites (on mac or linux system);
   1. Installl anaconda
   2. open a terminal and create your conda environment to install Pytorch and supporting packages (scipy); the following is an example
      ```
          conda create --name Pytorch
          conda activate Pytorch
          conda install pytorch cudatoolkit=10.2 -c pytorch
          conda install scipy
      ```
3. Now, you have every thing ready! the usage of iQSM and iQFM is easy: 
    1. convert your magnitude and phase data into NIFTI files, one can also make a BrainMask.nii if BET tool is installed;
    2. save your acquisition parameters (B0 strength, B0_dir, voxel size, TE) in a mat file
      ```
        Recon_iQSM(PhasePath, ParamsPath, MaskPath, MagPath, ReconPath) % for QSM recon; ReconPath is where results saved; 
        Recon_iQFM(PhasePath, ParamsPath, MaskPath, MagPath, ReconPath) % for lfs recon
      ```
    Only Phase data and Parameters file is compulsory. The other inputs are optional!


[⬆ top](#readme)
