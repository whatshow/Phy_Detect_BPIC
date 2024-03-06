close all;
clear;
clc;

%% load data
path_file = "_dist/Samples/Test/test_mimo.mat";
load(path_file);

%% Param Config - Model
SNR_range = 4:3:19;                                        % SNR range
SERs = zeros(1, length(SNR_range));                         % SERs for every SNR
SERs2 = zeros(1, length(SNR_range));                         % SERs for every SNR
M = 4;                                                     % M-ary QAM
sym_bitnum = log2(M);                                       % Bit number in 1 M-ary modulation symbol
sympool = qammod([0: M - 1], M, "UnitAveragePower", true); % The symbol pool to store all possible M-ary modulation symbols               
tx_num = 6;                                                % Tx antenna number
rx_num = 8;                                                % Rx antenna number

% Frames for each SNR    
nFrames = 3e4*ones(length(SNR_range), 1).';                                      
nFrames(end-1) = 5e4;
nFrames(end) = 1e5;

%% Param Config
iter_times = 10;    % The maximal iteration number of detection algorithms 

%% Simulation
for idx = 1:length(SNR_range)
    % Get current SNR
    SNR = SNR_range(idx);
    noiseLevel = 10^(-SNR/10);
    fprintf("SNR = %f, noiselevel = %f \n", SNR, noiseLevel);
    
    % Prepare the space to store all BERs during 'nFrames' times
    SER_TMP = zeros(1, nFrames(idx));
    SER_TMP2 = zeros(1, nFrames(idx));
    % Try several times to do average on all BERs to avoid fluctuation
    parfor try_times = 1:nFrames(idx)
        y = y_all(:, :, idx, try_times);
        H = H_all(:, :, idx, try_times);
        x = x_all(:, :, idx, try_times);
        
        % B-PIC-DSC MMSE
        [syms] = Detect_B_PIC_DSC_MMSE(sympool, y, H, noiseLevel, iter_times);
        % BPIC-MMSE
        bpic = BPIC(sympool, "bso_mean_init", BPIC.BSO_MEAN_INIT_MMSE, "bso_var", BPIC.BSO_VAR_APPRO, "bso_var_cal", BPIC.BSO_VAR_CAL_MMSE, "dsc_ise", BPIC.DSC_ISE_MMSE, "detect_sour", BPIC.DETECT_SOUR_BSE);
        syms_BPIC_MMSE = bpic.detect(y, H, noiseLevel, "sym_map", true);
        syms_BPIC_MMSE = syms_BPIC_MMSE(:);
        
        % calculate SER
        % To bits
        nbits_pred = qamdemod(syms, M,'OutputType','bit','UnitAveragePower',true);
        % To symbols
        x_est = qammod(nbits_pred, M,'InputType','bit','UnitAveragePower',true);
        SER_TMP(1, try_times) = sum(abs(x_est - x) > eps)/tx_num;
        SER_TMP2(1, try_times) = sum(abs(syms_BPIC_MMSE - x) > eps)/tx_num;        
    end
    % do SER average
    SERs(idx) = mean(SER_TMP);
    SERs2(idx) = mean(SER_TMP2);
end

%% save
save(path_file, "SERs", "SERs2", "-append");
save(path_file, "sympool", "-append");

%% plot
semilogy(SNR_range, SERs, '--r','LineWidth',2, 'MarkerSize', 12);
hold on
semilogy(SNR_range, SERs2, '--sb', 'LineWidth',1, 'MarkerSize', 12);
% semilogy(SNR_range, SERs_BPDnet_Batch128, '-v',  'Color', [0.75, 0, 0.75],'LineWidth',2, 'MarkerSize', 12);
% hold on
% semilogy(SNR_range, SERs_BPDnet_Batch256, '-*k','LineWidth',2, 'MarkerSize', 12);
hold off
grid on;
xlabel("SNR(dB)");
ylabel("SER");
xlim([min(SNR_range), max(SNR_range)]);
%ylim([10^-4, 1]);
%xlim([16, 30]);
legend("BPIC-MMSE(Alva)", "BPIC-MMSE");
title("MIMO Test Tx=6 Rx=8 4QAM")