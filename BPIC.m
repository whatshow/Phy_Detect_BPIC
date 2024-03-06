classdef BPIC < handle
    % constants
    properties(Constant)
        % BSO
        BSO_MEAN_INIT_NO     = 0;
        BSO_MEAN_INIT_MMSE   = 1;
        BSO_MEAN_INIT_MRC    = 2;        % in Alva's paper, he calls this matched filter
        BSO_MEAN_INIT_ZF     = 3;        % x_hat=inv(H'*H)*H'*y (this requires y.len >= x.len)
        BSO_MEAN_INIT_TYPES  = [BPIC.BSO_MEAN_INIT_NO, BPIC.BSO_MEAN_INIT_MMSE, BPIC.BSO_MEAN_INIT_MRC, BPIC.BSO_MEAN_INIT_ZF];
        BSO_MEAN_CAL_MRC = 1;
        BSO_MEAN_CAL_ZF = 2;
        BSO_MEAN_CAL_TYPES = [BPIC.BSO_MEAN_CAL_MRC, BPIC.BSO_MEAN_CAL_ZF];
        BSO_VAR_APPRO   = 1;        % use approximated variance
        BSO_VAR_ACCUR   = 2;        % use accurate variance (will update in the iterations)
        BSO_VAR_TYPES   = [BPIC.BSO_VAR_APPRO, BPIC.BSO_VAR_ACCUR];
        BSO_VAR_CAL_MMSE   = 1;     % use MMSE to estimate the variance
        BSO_VAR_CAL_MRC    = 2;     % use the MRC to estimate the variance
        BSO_VAR_CAL_ZF     = 3;     % use ZF to estimate the variance
        BSO_VAR_CAL_TYPES = [ BPIC.BSO_VAR_CAL_MMSE, BPIC.BSO_VAR_CAL_MRC, BPIC.BSO_VAR_CAL_ZF];
        % DSC
        % DSC - instantaneous square error
        DSC_ISE_NO      = 0;        % use the error directly
        DSC_ISE_MRC     = 1;        % in Alva's paper, he calls this matched filter
        DSC_ISE_ZF      = 2;
        DSC_ISE_MMSE    = 3;
        DSC_ISE_TYPES = [BPIC.DSC_ISE_NO, BPIC.DSC_ISE_MRC, BPIC.DSC_ISE_ZF, BPIC.DSC_ISE_MMSE];
        % DSC - mean previous source
        DSC_MEAN_PREV_SOUR_BSE = 1; % default in Alva's paper
        DSC_MEAN_PREV_SOUR_DSC = 2;
        DSC_MEAN_PREV_SOUR_TYPES = [BPIC.DSC_MEAN_PREV_SOUR_BSE, BPIC.DSC_MEAN_PREV_SOUR_DSC];
        % DSC - variance previous source
        DSC_VAR_PREV_SOUR_BSE = 1;  % default in Alva's paper
        DSC_VAR_PREV_SOUR_DSC = 2;
        DSC_VAR_PREV_SOUR_TYPES = [BPIC.DSC_VAR_PREV_SOUR_BSE, BPIC.DSC_VAR_PREV_SOUR_DSC];
        % Detect
        DETECT_SOUR_BSE = 1;
        DETECT_SOUR_DSC = 2;
        DETECT_SOURS = [BPIC.DETECT_SOUR_BSE, BPIC.DETECT_SOUR_DSC];
        
    end
    properties
        constellation {mustBeNumeric}
        constellation_len {mustBeNumeric}
        bso_mean_init = BPIC.BSO_MEAN_INIT_MMSE;            % default in Alva's paper
        bso_mean_cal = BPIC.BSO_MEAN_CAL_MRC;               % default in Alva's paper
        bso_var = BPIC.BSO_VAR_APPRO;                       % default in Alva's paper
        bso_var_cal = BPIC.BSO_VAR_CAL_MRC;                 % default in Alva's paper
        dsc_ise = BPIC.DSC_ISE_MRC;                         % default in Alva's paper
        dsc_mean_prev_sour = BPIC.DSC_MEAN_PREV_SOUR_BSE;   % default in Alva's paper
        dsc_var_prev_sour = BPIC.DSC_VAR_PREV_SOUR_BSE;     % default in Alva's paper
        min_var {mustBeNumeric} = eps       % the default minimal variance is 2.2204e-16
        iter_num = 10                       % maximal iteration
        iter_diff_min = eps;                % the minimal difference between 2 adjacent iterations
        detect_sour = BPIC.DETECT_SOUR_DSC;
    end
    methods
        % init
        % @constellation:         the constellation, a vector.
        % @bso_mean_init:         1st iteration method in **BSO** to calculate the mean. Default: `BPIC.BSO_INIT_MMSE`, others: `BPIC.BSO_INIT_MRC, BPIC.BSO_INIT_ZF` (`BPIC.BSO_INIT_NO` should not be used but you can try)
        % @bso_mean_cal:          other iteration method in **BSO** to calculate the mean. Default: `BPIC.BSO_MEAN_CAL_MRC` (`BPIC.BSO_MEAN_CAL_ZF` should not be used but you can try)
        % @bso_var:               use approximate or accurate variance in **BSO**. Default: `BPIC.BSO_VAR_APPRO`, others: `BPIC.BSO_VAR_ACCUR`
        % @bso_var_cal:           the method in **BSO** to calculate the variance. Default: `BPIC.BSO_VAR_CAL_MRC`, others: `BPIC.BSO_VAR_CAL_MRC` (`BSO_VAR_CAL_ZF` should not be used but you can try)
        % @dsc_ise:               how to calculate the instantaneous square error. Default: `BPIC.DSC_ISE_MRC`, others: `BPIC.DSC_ISE_NO, BPIC.DSC_ISE_ZF, BPIC.DSC_ISE_MMSE`
        % @dsc_mean_prev_sour:    the source of previous mean in DSC. Default: `BPIC.DSC_MEAN_PREV_SOUR_BSE`, others: `BPIC.DSC_MEAN_PREV_SOUR_DSC`
        % @dsc_var_prev_sour:     the source of previous variance in DSC. Default: `BPIC.DSC_VAR_PREV_SOUR_BSE`, others: `BPIC.DSC_VAR_PREV_SOUR_DSC`
        % @min_var:               the minimal variance.
        % @iter_num:              the maximal iteration.
        % @iter_diff_min:         the minimal difference in **DSC** to early stop.
        % @detect_sour:           the source of detection result. Default: `BPIC.DETECT_SOUR_DSC`, others: `BPIC.DETECT_SOUR_BSE`.
        % @batch_size:            the batch size
        function self = BPIC(constellation, varargin)
            % register optional inputs 
            inPar = inputParser;
            addParameter(inPar,"bso_mean_init", self.bso_mean_init, @(x) isscalar(x)&ismember(x, BPIC.BSO_MEAN_INIT_TYPES));
            addParameter(inPar,"bso_mean_cal", self.bso_mean_cal, @(x) isscalar(x)&ismember(x, BPIC.BSO_MEAN_CAL_TYPES));
            addParameter(inPar,"bso_var", self.bso_var, @(x) isscalar(x)&ismember(x, BPIC.BSO_VAR_TYPES));
            addParameter(inPar,"bso_var_cal", self.bso_var_cal, @(x) isscalar(x)&ismember(x, BPIC.BSO_VAR_CAL_TYPES));
            addParameter(inPar,"dsc_ise", self.dsc_ise, @(x) isscalar(x)&ismember(x, BPIC.DSC_ISE_TYPES));
            addParameter(inPar,"dsc_mean_prev_sour", self.dsc_mean_prev_sour, @(x) isscalar(x)&ismember(x, BPIC.DSC_MEAN_PREV_SOUR_TYPES));
            addParameter(inPar,"dsc_var_prev_sour", self.dsc_var_prev_sour, @(x) isscalar(x)&ismember(x, BPIC.DSC_VAR_PREV_SOUR_TYPES));
            addParameter(inPar,"min_var", self.min_var, @isnumeric);
            addParameter(inPar,"iter_num", self.iter_num, @isnumeric);
            addParameter(inPar,"iter_diff_min", self.iter_diff_min, @isnumeric);
            addParameter(inPar,"detect_sour", self.detect_sour, @(x) isscalar(x)&ismember(x, BPIC.DETECT_SOURS));
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
            self.bso_mean_init = inPar.Results.bso_mean_init;
            self.bso_mean_cal = inPar.Results.bso_mean_cal;
            self.bso_var = inPar.Results.bso_var;
            self.bso_var_cal = inPar.Results.bso_var_cal;
            self.dsc_ise = inPar.Results.dsc_ise;
            self.dsc_mean_prev_sour = inPar.Results.dsc_mean_prev_sour;
            self.dsc_var_prev_sour = inPar.Results.dsc_var_prev_sour;
            self.min_var = inPar.Results.min_var;
            self.iter_num = inPar.Results.iter_num;
            if self.iter_num < 1
                error("The iteration number must be positive.")
            end
            self.iter_diff_min = inPar.Results.iter_diff_min;
            self.detect_sour = inPar.Results.detect_sour;
        end
        
        % detect
        % @y:           the received signal
        % @H:           the channel matrix
        % @No:          the noise (linear) power
        % @sym_map:     false by default. If true, the output will be mapped to the constellation
        function x = detect(self, y, H, No, varargin)
            % register optional inputs 
            inPar = inputParser;
            addParameter(inPar,"sym_map", false, @(x) isscalar(x)&islogical(x));
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            sym_map = inPar.Results.sym_map;
            
            % input check
            if isscalar(y) 
                error("The received signal must be a vector.")
            elseif ~isvector(y)
                error("The received signal must be a vector.")
            end
            if isscalar(H) 
                error("The channel must be a matrix.")
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
            y = y(:);
            Ht = H';
            Hty = Ht*y;
            HtH = Ht*H;
            HtH_off = ((eye(x_num)+1) - eye(x_num).*2).*HtH;
            HtH_off_sqr = HtH_off.^2;
            % constant values - inverse matrix
            mrc_mat = diag(1./diag(HtH));
            zf_mat = inv(HtH);
            % constant values - BSO - mean - 1st iter
            bso_zigma_1 = eye(x_num);
            if self.bso_mean_init == BPIC.BSO_MEAN_INIT_MMSE
                bso_zigma_1 = inv(HtH + No*eye(x_num));
            end
            if self.bso_mean_init == BPIC.BSO_MEAN_INIT_MRC
                bso_zigma_1 = mrc_mat;
            end
            if self.bso_mean_init == BPIC.BSO_MEAN_INIT_ZF
                bso_zigma_1 = zf_mat;
            end
            % constant values - BSO - mean - other iteration
            bso_zigma_others = mrc_mat;
            if self.bso_mean_cal == BPIC.BSO_MEAN_CAL_ZF
                bso_zigma_others = zf_mat;
            end
            % constant values - BSO - variance
            bso_var_mat = 1./diag(HtH);
            if self.bso_var_cal == BPIC.BSO_VAR_CAL_MMSE
                bso_var_mat = diag(inv(HtH + No*eye(x_num)));
            end
            if self.bso_var_cal == BPIC.BSO_VAR_CAL_ZF
                bso_var_mat = diag(zf_mat);
            end
            bso_var_mat_sqr = bso_var_mat.^2;
            % constant values - DSC
            dsc_w = eye(x_num); % the default is `BPIC.DSC_ISE_NO`
            if self.dsc_ise == BPIC.DSC_ISE_MRC
                dsc_w = mrc_mat;
            end
            if self.dsc_ise == BPIC.DSC_ISE_ZF
                dsc_w = zf_mat;
            end
            if self.dsc_ise == BPIC.DSC_ISE_MMSE
                dsc_w = inv(HtH + No*eye(x_num));
            end
            
            % iterative detection
            x_dsc = zeros(x_num, 1);
            v_dsc = zeros(x_num, 1);
            ise_dsc_prev = zeros(x_num, 1);
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
                    v_bso = No.*bso_var_mat;
                end
                if self.bso_var == BPIC.BSO_VAR_ACCUR
                    v_bso = No.*bso_var_mat + HtH_off_sqr*v_dsc.*bso_var_mat_sqr;
                end
                v_bso = max(v_bso, self.min_var);
                
                % BSE
                % BSE - Estimate P(x|y) using Gaussian distribution
                pxyPdfExpPower = -1./(2*v_bso).*abs(repmat(x_bso, 1, self.constellation_len) - repmat(self.constellation, x_num, 1)).^2;
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
                % DSC - error
                ise_dsc = (dsc_w*(Hty - HtH*x_bse)).^2;
                ies_dsc_sum = ise_dsc + ise_dsc_prev;
                ies_dsc_sum = max(ies_dsc_sum, self.min_var);
                % DSC - rho (if we use this rho, we will have a little difference)
                rho_dsc = ise_dsc_prev./ies_dsc_sum;
                % DSC - mean
                if iter_id == 1
                    x_dsc = x_bse;
                else
                    if self.dsc_mean_prev_sour == BPIC.DSC_MEAN_PREV_SOUR_BSE
                        %x_dsc = ise_dsc./ies_dsc_sum.*x_bse_prev + ise_dsc_prev./ies_dsc_sum.*x_bse;
                        x_dsc = (1 - rho_dsc).*x_bse_prev + rho_dsc.*x_bse;
                    end
                    if self.dsc_mean_prev_sour == BPIC.DSC_MEAN_PREV_SOUR_DSC
                        x_dsc = (1 - rho_dsc).*x_dsc + rho_dsc.*x_bse;
                    end
                end
                % DSC - variance
                if iter_id == 1
                    v_dsc = v_bse;
                else
                    if self.dsc_var_prev_sour == BPIC.DSC_VAR_PREV_SOUR_BSE
                        %v_dsc = ise_dsc./ies_dsc_sum.*v_bse_prev + ise_dsc_prev./ies_dsc_sum.*v_bse;
                        v_dsc = (1 - rho_dsc).*v_bse_prev + rho_dsc.*v_bse;
                    end
                    if self.dsc_var_prev_sour == BPIC.DSC_VAR_PREV_SOUR_DSC
                        v_dsc = (1 - rho_dsc).*v_dsc + rho_dsc.*v_bse;
                    end
                end
                
                % early stop
                if iter_id > 1 && sum(abs(v_dsc - v_dsc_prev).^2) <= self.iter_diff_min
                    break;
                end
                
                % update statistics
                % update statistics - BSE
                if self.dsc_mean_prev_sour == BPIC.DSC_MEAN_PREV_SOUR_BSE
                    x_bse_prev = x_bse;
                end
                if self.dsc_var_prev_sour == BPIC.DSC_VAR_PREV_SOUR_BSE
                    v_bse_prev = v_bse;
                end
                % update statistics - DSC
                v_dsc_prev = v_dsc;
                % update statistics - DSC - instantaneous square error
                ise_dsc_prev = ise_dsc;
            end
            % take the detection value
            if self.detect_sour == BPIC.DETECT_SOUR_BSE
                x = x_bse;
            end
            if self.detect_sour == BPIC.DETECT_SOUR_DSC
                x = x_dsc;
            end
            % hard estimation
            if sym_map
                x = self.symmap(x);
            end
        end
        
        % symbol mapping (hard)
        function syms_mapped = symmap(self, syms)
            if ~isvector(syms)
                error("Symbols must be into a vector form to map.");
            end
            % the input must be a column vector
            syms = syms(:);
            syms_dis = abs(syms - self.constellation).^2;
            [~,syms_dis_min_idx] =  min(syms_dis,[],2);
            syms_mapped = self.constellation(syms_dis_min_idx);
        end
        
    end
end