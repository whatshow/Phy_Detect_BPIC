import numpy as np
from numpy import sqrt
from numpy.random import randint as randi
from numpy.random import randn
import matplotlib.pyplot as plt
import sys
sys.path.append("..");
from BPIC import BPIC

# Param Config - Model
SNR_range = np.arange(4,20, 3);                             # SNR range
SERs = np.zeros(len(SNR_range));                            # SERs for every SNR
sympool = np.asarray([-0.707106781186548 + 0.707106781186548j,-0.707106781186548 - 0.707106781186548j,0.707106781186548 + 0.707106781186548j,0.707106781186548 - 0.707106781186548j]);                     
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
        # Create symbols
        x = np.take(sympool, randi(0, len(sympool), size=tx_num));
        # Channel 
        H = (randn(rx_num, tx_num) + 1j*randn(rx_num, tx_num))/sqrt(2*tx_num);
        # Noise
        noise = sqrt(noiseLevel/2) * (randn(rx_num) + 1j*randn(rx_num));
        # Through AWGN channel to get y 
        y = H@x + noise;
        
        # B-PIC-DSC 
        syms_BPIC_MMSE = bpic.detect(y, H, noiseLevel, sym_map=True);
        # SER
        SER_TMP[try_times] = sum(syms_BPIC_MMSE - x > np.finfo(float).eps)/tx_num;
    # do SER average
    SERs[idx] = np.mean(SER_TMP);
# plot
SERs_mat = [0.0385333333333332,0.0119388888888889,0.00192777777777778,0.000277777777777778,2.66666666666667e-05,6.66666666666667e-06];
SERs_diff = SERs - SERs_mat;



print("SER difference is %.16f"%sum(SERs_diff));