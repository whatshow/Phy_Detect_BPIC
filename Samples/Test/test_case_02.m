clear;
clc;

% symbols & channel
SNR_dB = 10;
No = 10^(-SNR_dB/10);
M = 16;                                                      % M-ary QAM
sym_bitnum = log2(M);                                       % Bit number in 1 M-ary modulation symbol
sympool = qammod(0:M-1, M, "UnitAveragePower", true);       % The symbol pool to store all possible M-ary modulation symbols               
tx_num = 8;                                                 % Tx antenna number
rx_num = 8;

% detection settings
iter_times = 10;

%% simulation
% sim - Tx
nbits_len = tx_num*sym_bitnum;
nbits = randi([0 1], nbits_len, 1);
% Create symbols
x = qammod(nbits, M,'InputType','bit','UnitAveragePower',true);

% sim - channel
% Rayleigh fadding channel
H = (randn(rx_num, tx_num) + 1j*randn(rx_num, tx_num))/sqrt(2*tx_num) ;
% Noise
noise = sqrt(No/2) * (randn(rx_num,1) + 1j*randn(rx_num,1)) ;
% through Rayleigh fadding channel to get y 
y = H*x + noise;

% sim - Rx
% sim - Rx - Alva - B-PIC-DSC MMSE
[syms_BPIC_MMSE_Alva] = Detect_B_PIC_DSC_MMSE(sympool, y, H, No, iter_times);
% sim - Rx - Alva - ZF
[syms_BPIC_ZF_Alva] = Detect_B_PIC_DSC(sympool, y, H, No, iter_times);
% BPIC - MMSE
bpic = BPIC(sympool, "bso_mean_init", BPIC.BSO_MEAN_INIT_MMSE, "bso_var", BPIC.BSO_VAR_APPRO, "bso_var_cal", BPIC.BSO_VAR_CAL_MMSE, "dsc_ise", BPIC.DSC_ISE_MMSE, "detect_sour", BPIC.DETECT_SOUR_BSE);
syms_BPIC_MMSE = bpic.detect(y, H, No);
% BPIC - ZF
bpic = BPIC(sympool, "bso_mean_init", BPIC.BSO_MEAN_INIT_ZF, "bso_var", BPIC.BSO_VAR_APPRO, "bso_var_cal", BPIC.BSO_VAR_CAL_ZF, "dsc_ise", BPIC.DSC_ISE_ZF, "detect_sour", BPIC.DETECT_SOUR_BSE);
syms_BPIC_ZF = bpic.detect(y, H, No);
% BPIC - ZF - uniform
bpic = BPIC(sympool, "bso_mean_init", BPIC.BSO_MEAN_INIT_ZF, "bso_mean_cal", BPIC.BSO_MEAN_CAL_ZF, "bso_var", BPIC.BSO_VAR_APPRO, "bso_var_cal", BPIC.BSO_VAR_CAL_ZF, "dsc_ise", BPIC.DSC_ISE_ZF, "detect_sour", BPIC.DETECT_SOUR_BSE);
syms_BPIC_ZF_Uniform = bpic.detect(y, H, No);
% BPIC - Paper
bpic = BPIC(sympool, "bso_mean_init", BPIC.BSO_MEAN_INIT_MRC, "bso_var", BPIC.BSO_VAR_APPRO, "bso_var_cal", BPIC.BSO_VAR_CAL_MRC, "dsc_ise", BPIC.DSC_ISE_MRC, "detect_sour", BPIC.DETECT_SOUR_BSE);
syms_BPIC_paper = bpic.detect(y, H, No);

fprintf("Estimated Symbol Difference(MMSE) is %.16f\n", sum(abs(syms_BPIC_MMSE - syms_BPIC_MMSE_Alva)));
fprintf("Estimated Symbol Difference(ZF) is %.16f\n", sum(abs(syms_BPIC_ZF - syms_BPIC_ZF_Alva)));