close all;
clear;
clc;

%% delete data
path_folder = "_dist/Samples/Test/";
path_file = path_folder + "test_mimo.mat";
if ~exist(path_folder, 'dir')
    mkdir(path_folder);
end
if exist(path_file, 'file')
    delete(path_file)
end

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
x_all = zeros(tx_num, 1, length(SNR_range), 1e5);
H_all = zeros(rx_num, tx_num, length(SNR_range), 1e5);
y_all = zeros(rx_num, 1, length(SNR_range), 1e5);
No_all = zeros(rx_num, 1, length(SNR_range), 1e5);
for idx = 1:length(SNR_range)
    % Get current SNR
    SNR = SNR_range(idx);
    noiseLevel = 10^(-SNR/10);
    fprintf("SNR = %f, noiselevel = %f \n", SNR, noiseLevel);
    % Try several times to do average on all BERs to avoid fluctuation
    parfor try_times = 1:nFrames(idx)
        % nbits
        nbits_len = tx_num*sym_bitnum;
        nbits = randi([0 1], nbits_len, 1);
        % Create symbols
        x = qammod(nbits, M,'InputType','bit','UnitAveragePower',true);
        % Channel 
        H = (randn(rx_num, tx_num) + 1j*randn(rx_num, tx_num))/sqrt(2*tx_num) ;
        % Noise
        noise = sqrt(noiseLevel/2) * (randn(rx_num,1) + 1j*randn(rx_num,1)) ;
        % Through AWGN channel to get y 
        y = H*x + noise;
        
        % store data
        x_all(:, :, idx, try_times) = x;
        H_all(:, :, idx, try_times) = H;
        y_all(:, :, idx, try_times) = y;
        No_all(:, :, idx, try_times) = noise;
    end
end

%% store data into the file
save(path_file, "x_all", "H_all", "y_all", "No_all");

fprintf("\nData is generated!\n");
