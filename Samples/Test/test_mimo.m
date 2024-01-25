close all;
clear;
clc;

%% Param Config - Model
SNR_range = 4:3:19;                                        % SNR range
SERs = zeros(1, length(SNR_range));                         % SERs for every SNR
M = 4;                                                     % M-ary QAM
sym_bitnum = log2(M);                                       % Bit number in 1 M-ary modulation symbol
sympool = qammod([0: M - 1], M, "UnitAveragePower", true); % The symbol pool to store all possible M-ary modulation symbols               
tx_num = 6;                                                % Tx antenna number
rx_num = 8;                                                % Rx antenna number

% Frames for each SNR    
nFrames = 3e4*ones(length(SNR_range), 1).';                                      
nFrames(end-1) = 5e4;
nFrames(end) = 1e5;

%% Param Config - EP algorithm 
iter_times = 10;                                         % The times to estimate the distribution of  

%% Simulation
for idx = 1:length(SNR_range)
    % Get current SNR
    SNR = SNR_range(idx);
    noiseLevel = 10^(-SNR/10);
    fprintf("SNR = %f, noiselevel = %f \n", SNR, noiseLevel);
    
    % Prepare the space to store all BERs during 'nFrames' times
    SER_TMP = zeros(1, nFrames(idx));
    % Try several times to do average on all BERs to avoid fluctuation
    parfor try_times = 1:nFrames(idx)
        % nbits
        nbits_len = tx_num*sym_bitnum;
        nbits = randi([0 1], nbits_len, 1);
        % Channel 
        H = (randn(rx_num, tx_num) + 1j*randn(rx_num, tx_num))/sqrt(2*tx_num) ;
        % Noise
        noise = sqrt(noiseLevel/2) * (randn(rx_num,1) + 1j*randn(rx_num,1)) ;
        % Create symbols
        x = qammod(nbits, M,'InputType','bit','UnitAveragePower',true);
        % Through AWGN channel to get y 
        y = H*x + noise;
        
        % B-PIC-DSC MMSE
        [syms] = Detect_B_PIC_DSC_MMSE(sympool, y, H, noiseLevel, iter_times);
        
        % To bits
        nbits_pred = qamdemod(syms, M,'OutputType','bit','UnitAveragePower',true);
        % To symbols
        x_est = qammod(nbits_pred, M,'InputType','bit','UnitAveragePower',true);
        % calculate SER
        SER_TMP(1, try_times) = sum((x_est - x > eps))/length(x);
        
    end
    % do SER average
    SERs(idx) = mean(SER_TMP);
end

% plot
semilogy(SNR_range, SERs, '--r','LineWidth',2, 'MarkerSize', 12);
% hold on
% semilogy(SNR_range, SERs_BPDnet_Batch64, '-sb', 'LineWidth',2, 'MarkerSize', 12);
% hold on
% semilogy(SNR_range, SERs_BPDnet_Batch128, '-v',  'Color', [0.75, 0, 0.75],'LineWidth',2, 'MarkerSize', 12);
% hold on
% semilogy(SNR_range, SERs_BPDnet_Batch256, '-*k','LineWidth',2, 'MarkerSize', 12);
% hold off
grid on;
xlabel("SNR(dB)");
ylabel("SER");
xlim([min(SNR_range), max(SNR_range)]);
%ylim([10^-4, 1]);
%xlim([16, 30]);
legend("Matlab", "Python-Batch64", "Python-Batch128", "Python-Batch256");
title("MIMO Test Tx=6 Rx=8 4QAM")