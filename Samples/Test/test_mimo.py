import numpy as np
import scipy.io
import os
import sys
sys.path.append("..");
from BPIC import BPIC

project_name = "phy_detect_bpic";
path_folder = os.path.abspath(os.path.dirname(__file__)).lower();
path_folder = path_folder[:path_folder.find(project_name)+len(project_name)];
path_file = os.path.normpath(path_folder+"/_dist/Samples/Test/test_mimo.mat")

# load matlab data
matlab_data = scipy.io.loadmat(path_file);
y_all = matlab_data["y_all"];
H_all = matlab_data["H_all"];
x_all = matlab_data["x_all"];
SERs_mat = matlab_data["SERs"].squeeze();
sympool = matlab_data["sympool"].squeeze();

# Param Config - Model
SNR_range = np.arange(4,20, 3);                             # SNR range
SERs = np.zeros(len(SNR_range));                            # SERs for every SNR                     
#sympool = np.array([-0.707106781186548 + 0.707106781186548j,-0.707106781186548 - 0.707106781186548j,0.707106781186548 + 0.707106781186548j,0.707106781186548 - 0.707106781186548j]);
tx_num = 6;                                                 # Tx antenna number
rx_num = 8;                                                 # Rx antenna number

# Frames for each SNR    
nFrames = 3e4*np.ones(len(SNR_range));                                      
nFrames[-2] = 5e4;
nFrames[-1] = 1e5;
nFrames = nFrames.astype(int);

# Param Config
iter_times = 10;    # The maximal iteration number of detection algorithms

# prepare detectors
bpic = BPIC(sympool, bso_var_cal=BPIC.BSO_VAR_CAL_MMSE, dsc_ise=BPIC.DSC_ISE_MMSE, detect_sour=BPIC.DETECT_SOUR_BSE);
# Simulation
for idx in range(len(SNR_range)):
    # Get current SNR
    SNR = SNR_range[idx];
    noiseLevel = 10**(-SNR/10);
    print("SNR = %f, noiselevel = %f"%(SNR, noiseLevel));
    
    # Prepare the space to store all BERs during 'nFrames' times
    SER_TMP = np.zeros(nFrames[idx]);
    # Try several times to do average on all BERs to avoid fluctuation
    for try_times in range(nFrames[idx]):
        y = y_all[:, :, idx, try_times].squeeze();
        H = H_all[:, :, idx, try_times];
        x = x_all[:, :, idx, try_times].squeeze();
        
        # B-PIC-DSC 
        syms_BPIC_MMSE = bpic.detect(y, H, noiseLevel, sym_map=True);
        # SER
        SER_TMP[try_times] = sum(abs(syms_BPIC_MMSE - x) > np.finfo(float).eps)/tx_num;
    # do SER average
    SERs[idx] = np.mean(SER_TMP);
# plot
SERs_diff = SERs - SERs_mat;



print("SER difference is %.16f"%sum(SERs_diff));