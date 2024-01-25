clear;
clc;

% symbols & channel
SNR_dB = 10;
No = 10^(-SNR_dB/10);
M = 16;                                                      % M-ary QAM
sym_bitnum = log2(M);                                       % Bit number in 1 M-ary modulation symbol
sympool = qammod(0:M-1, M, "UnitAveragePower", true);       % The symbol pool to store all possible M-ary modulation symbols               
tx_num = 6;                                                 % Tx antenna number
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
H = (randn(rx_num, tx_num) + 1j*randn(rx_num, tx_num))/sqrt(2*tx_num);
% Noise
noise = sqrt(No/2) * (randn(rx_num,1) + 1j*randn(rx_num,1)) ;
% through Rayleigh fadding channel to get y 
y = H*x + noise;

% sim - Rx
% B-PIC-DSC MMSE
[syms_BPIC_Alva] = Detect_B_PIC_DSC_MMSE(sympool, y, H, No, iter_times);
% BPIC 
%bpic = BPIC(sympool);
bpic = BPIC(sympool, "bso_init", BPIC.BSO_INIT_MMSE, "bso_var", BPIC.BSO_VAR_ACCUR, "dsc_ise", BPIC.DSC_ISE_MMSE);
syms_BPIC = bpic.detect(y, H, No);