classdef BPIC < handle
    % constants
    properties(Constant)
        % BSO
        BSO_INIT_NO     = 0;
        BSO_INIT_MMSE   = 1;
        BSO_INIT_MRC    = 2;        % in Alva's paper, he calls this matched filter
        BSO_INIT_ZF     = 3;        % x_hat=inv(H'*H)*H'*y (this requires y.len >= x.len)
        BSO_INIT_TYPES  = [BPIC.BSO_INIT_NO, BPIC.BSO_INIT_MMSE, BPIC.BSO_INIT_MRC, BPIC.BSO_INIT_ZF];
        BSO_VAR_APPRO   = 1;        % use approximated variance
        BSO_VAR_ACCUR   = 2;        % use accurate variance (will update in the iterations)
        BSO_VAR_TYPES   = [BPIC.BSO_VAR_APPRO, BPIC.BSO_VAR_ACCUR];
        % DSC
        % DSC - instantaneous square error
        DSC_ISE_NO      = 0;        % use the error directly
        DSC_ISE_MRC     = 1;        % in Alva's paper, he calls this matched filter
        DSC_ISE_ZF      = 2;
        DSC_ISE_MMSE    = 3;
        DSC_ISE_TYPES = [BPIC.DSC_ISE_NO, BPIC.DSC_ISE_MRC, BPIC.DSC_ISE_ZF, BPIC.DSC_ISE_MMSE];
    end
    properties
        constellation {mustBeNumeric}
        constellation_len {mustBeNumeric}
        bso_init = BPIC.BSO_INIT_MMSE;
        bso_var = BPIC.BSO_VAR_APPRO;
        dsc_ise = BPIC.DSC_ISE_MRC;
        min_var {mustBeNumeric} = eps       % the default minimal variance is 2.2204e-16
        iter_num = 10                       % maximal iteration
    end
    methods
        % init
        % @constellation:       the constellation
        % @MMSE:                whether use MMSE or not
        % @min_var
        function self = BPIC(constellation, varargin)
            % register optional inputs 
            inPar = inputParser;
            addParameter(inPar,"bso_init", self.bso_init, @(x) isscalar(x)&ismember(x, BPIC.BSO_INIT_TYPES));
            addParameter(inPar,"bso_var", self.bso_var, @(x) isscalar(x)&ismember(x, BPIC.BSO_VAR_TYPES));
            addParameter(inPar,"dsc_ise", self.dsc_ise, @(x) isscalar(x)&ismember(x, BPIC.DSC_ISE_TYPES));
            addParameter(inPar,"min_var", self.min_var, @isnumeric);
            addParameter(inPar,"iter_num", self.iter_num, @isnumeric);
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            
            % take inputs
            if ~isvector(constellation)
                error("The constellation must be a vector.");
            else
                % constellation must be a row vector or an 1D vector
                constellation = constellation(:);
                constellation = constellation.';
                self.constellation = constellation;
                self.constellation_len = length(constellation);
            end
            self.bso_init = inPar.Results.bso_init;
            self.bso_var = inPar.Results.bso_var;
            self.dsc_ise = inPar.Results.dsc_ise;
            self.min_var = inPar.Results.min_var;
            self.iter_num = inPar.Results.iter_num;
            if self.iter_num < 1
                error("The iteration number must be positive.")
            end
        end
        
        % detect
        % @y:       the received signal
        % @H:       the channel matrix
        % @No:      the noise (linear) power
        function x = detect(self, y, H, No)
            % input check
            if isscalar(y) 
                error("The received signal must be a vector.")
            elseif ~isvector(y)
                error("The received signal must be a vector.")
            end
            if isscalar(H) 
                error("The received signal must be a vector.")
            elseif ~ismatrix(H)
                error("The channel must be a matrix.")
            end
            [y_num, x_num] = size(H);
            if y_num ~= length(y)
                error("The channel row number does not equal to the signal number.");
            end
            if y_num < x_num
                error("The channel is a correlated channel.")
            end
            if ~isscalar(No)
                error("The noise power must be a scalar.");
            end
            
            % constant values
            Ht = H';
            Hty = Ht*y;
            HtH = Ht*H;
            HtH_off = ((eye(x_num)+1) - eye(x_num).*2).*HtH;
            HtH_off_sqr = HtH_off.^2;
            HtH_diag = diag(HtH);
            HtH_diag_sqr = HtH_diag.^2;
            % constant values - BSO
            bso_zigma_others = diag(1./diag(HtH));
            bso_zigma_1 = eye(x_num);
            if self.bso_init == BPIC.BSO_INIT_MMSE
                bso_zigma_1 = inv(HtH + No*diag(ones(x_num, 1)));
            end
            if self.bso_init == BPIC.BSO_INIT_MRC
                bso_zigma_1 = bso_zigma_others;
            end
            if self.bso_init == BPIC.BSO_INIT_ZF
                bso_zigma_1 = inv(HtH);
            end
            
            % iterative detection
            x_bso = zeros(x_num, 1);
            v_bso = zeros(x_num, 1)
            x_dsc = zeros(x_num, 1);
            v_dsc = zeros(x_num, 1);
            for iter_id = 1:self.iter_num
                % BSO
                % BSO - mean
                if iter_id == 1
                    x_bso = bso_zigma_1*(Hty - HtH_off*x_dsc);
                else
                    x_bso = bso_zigma_others*(Hty - HtH_off*x_dsc);
                end
                % BSO - variance
                if self.bso_var == BPIC.BSO_VAR_APPRO
                    v_bso = No./HtH_diag;
                end
                if self.bso_var == BPIC.BSO_VAR_ACCUR
                    v_bso = No./HtH_diag + HtH_off_sqr*v_dsc./HtH_diag_sqr;
                end
                v_bso = max(v_bso, self.min_var);
                % BSE
                % BSE - Estimate P(x|y) using Gaussian distribution
                pxyPdfExpPower = -1./No.*abs(repmat(x_bso, 1, self.constellation_len) - repmat(self.constellation, x_num, 1)).^2;
                pxypdfExpNormPower = pxyPdfExpPower - max(pxyPdfExpPower, [], 2);   % make every row the max power is 0
                pxyPdf = exp(pxypdfExpNormPower);
                % BSE - Calculate the coefficient of every possible x to make the sum of all
                pxyPdfCoeff = 1./sum(pxyPdf, 2);
                pxyPdfCoeff = repmat(pxyPdfCoeff, 1, self.constellation_len);
                % BSE - PDF normalisation
                pxyPdfNorm = pxyPdfCoeff.*pxyPdf;
                % BSE - calculate the mean and variance
                x_bse = sum(pxyPdfNorm.*self.constellation, 2);
                x_bse_mat = repmat(x_bse, 1, self.constellation_len);
                v_bse = sum(abs(x_bse_mat - self.constellation).^2.*pxyPdfNorm, 2);
                v_bse = max(v_bse, self.min_var);
                
                % DSC
                
            end
        end
    end
end