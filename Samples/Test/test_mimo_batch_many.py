import numpy as np
import scipy.io
import sys
sys.path.append("..");
from BPIC import BPIC

# load matlab data
matlab_data = scipy.io.loadmat('Data/test_mimo.mat');
y_all = matlab_data["y_all"];
H_all = matlab_data["H_all"];
x_all = matlab_data["x_all"];
SERs_mat = matlab_data["SERs"].squeeze();
sympool = matlab_data["sympool"].squeeze();

# batch
batch_size = 100;

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
bpic = BPIC(sympool, bso_var_cal=BPIC.BSO_VAR_CAL_MMSE, dsc_ise=BPIC.DSC_ISE_MMSE, detect_sour=BPIC.DETECT_SOUR_BSE, batch_size=batch_size);
# Simulation
for idx in range(len(SNR_range)):
    # Get current SNR
    SNR = SNR_range[idx];
    noiseLevel = 10**(-SNR/10);
    print("SNR = %f, noiselevel = %f"%(SNR, noiseLevel));
    
    # Prepare the space to store all BERs during 'nFrames' times
    SER_TMP = np.zeros(int(nFrames[idx]/batch_size));
    # Try several times to do average on all BERs to avoid fluctuation
    for try_times in range(int(nFrames[idx]/batch_size)):
        y = y_all[:, :, idx, try_times*batch_size:(try_times*batch_size + batch_size)];
        H = H_all[:, :, idx, try_times*batch_size:(try_times*batch_size + batch_size)];
        x = x_all[:, :, idx, try_times*batch_size:(try_times*batch_size + batch_size)];
        y = np.moveaxis(y, -1, 0).squeeze(-1);
        H = np.moveaxis(H, -1, 0);
        x = np.moveaxis(x, -1, 0).squeeze(-1);
        
        # B-PIC-DSC 
        syms_BPIC_MMSE = bpic.detect(y, H, noiseLevel, sym_map=True);
        # SER
        SER_TMP[try_times] = np.mean(np.sum(abs(syms_BPIC_MMSE - x) > np.finfo(float).eps, axis=-1)/tx_num);
    # do SER average
    SERs[idx] = np.mean(SER_TMP);
# plot
SERs_diff = SERs - SERs_mat;



print("SER difference is %.16f"%sum(SERs_diff));