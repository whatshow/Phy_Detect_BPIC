%% Gaussian Estimation (soft decoding)
% <AUTHOR>
% Xinwei Qu, 21/1/2021
% email: xiqi4237@uni.sydney.edu.au
%
% <INTRODUCTION>
% y represent p(y|x)'s mean, noisePower represents p(y|x)'s variance. Then
% we can assume y fits Gaussian distribution of a mean. Then we test that
% mean of all possible x values. After that, we normalise all Gaussian
% distribution PDF of every x value. (The sum of those must be 1). Then we
% use those Gaussian distribution on every x to recalculate the mean and
% the variance of p(x|y)
%
% <WARNING>
% y and x must have the same dimension. 
%
% <INPUT>
% @y:               vector, the observation of received signals
% @noisePower:      scalar or vector, the noise power (that can be seen as the variance of y)
% @xPool:           vector, all possible values for x
% @isDecoding:      scalar, if it is decoding, we just output the most possible x 
%
function [pxyMean, pxyVar] = GaussianEst(y, noisePower, xPool, varargin)

    %% Inputs Name-Value Pair 
    % Load Delay Min Max, Doppler Min Max
    inPar = inputParser;
    % Set default values
    defaultProbCoefMin = -1;
    probCoefMin = defaultProbCoefMin;
    isDecoding = false;
    % Register names
    addParameter(inPar,'ProbabilityCoefficientMin', probCoefMin, @isnumeric);
    addParameter(inPar,'Decoding', isDecoding, @islogical);
    % Allow unmatched cases
    inPar.KeepUnmatched = true; 
    % Allow capital or small characters
    inPar.CaseSensitive = false;
    % Try to load those inputs 
    parse(inPar, varargin{:}); 
    % No input = using default type
    if isempty(find(inPar.UsingDefaults == "ProbabilityCoefficientMin", 1))
        probCoefMin = inPar.Results.ProbabilityCoefficientMin;
    end
    if isempty(find(inPar.UsingDefaults == "Decoding", 1))
        isDecoding = inPar.Results.Decoding;
    end


    %% Parameter Setting
    % y
    ySize = length(y);
    y = y(:);
    % noisePower
    noisePower = noisePower(:);
    noisePowerSize = length(noisePower);
    if noisePowerSize ~= 1 && noisePowerSize ~= ySize
        error("Noise Power must either a scalar or a vector with the same dimension of y");
    end
    
    % xPool
    xPoolSize = length(xPool);
    xPool = xPool(:);
    xPool = xPool.';                        % xPool -> a row vector
    xPoolVec = xPool;                       % save the vector form for the last decoding process
    
    % Estimate P(x|y) using Gaussian distribution
    y = repmat(y, 1, xPoolSize);
    xPool = repmat(xPool, ySize, 1);
    pxyPdfExpPower = -1./(2*noisePower).*abs(y - xPool).^2;
    pxypdfExpNormPower = pxyPdfExpPower - max(pxyPdfExpPower, [], 2);   % make every row the max power is 0
    pxyPdf = exp(pxypdfExpNormPower);
    % Calculate the coefficient of every possible x to make the sum of all
    % possbilities is 1
    if probCoefMin == defaultProbCoefMin
        pxyPdfCoeff = 1./sum(pxyPdf, 2);                                % this sum is a column vector
    else
        pxyPdfCoeff = max(1./sum(pxyPdf, 2), probCoefMin);              % this sum is a column vector
    end
    pxyPdfCoeff = repmat(pxyPdfCoeff, 1, xPoolSize);                    % make sum a matrix, ySize rows, xPoolSize colums
    % PDF normalisation
    pxyPdfNorm = pxyPdfCoeff.*pxyPdf;
    
    % recalculate the new mean and variance
    if isDecoding == 1
        % decoding
        [~, pxyPdfMaxId] = max(pxyPdfNorm, [], 2);
        pxyMean = xPoolVec(pxyPdfMaxId);
        pxyMean = pxyMean(:);
        pxyVar = noisePower;
    else
        % no decoding but esitmating
        pxyMean = sum(pxyPdfNorm.*xPool, 2);
        pxyMeanMat = repmat(pxyMean, 1, xPoolSize);
        pxyVar = sum(abs(pxyMeanMat - xPool).^2.*pxyPdfNorm, 2);
    end
end