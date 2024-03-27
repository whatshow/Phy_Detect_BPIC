# Bayesian PIC-DSC detector
[![PyPi](https://img.shields.io/badge/PyPi-1.0.4-blue)](https://pypi.org/project/whatshow-phy-mod-otfs/) [![MathWorks](https://img.shields.io/badge/MathWorks-1.0.4-red)](https://mathworks.com/matlabcentral/fileexchange/161136-whatshow_phy_mod_otfs)


This detection method is proposed in [A Bayesian Receiver With Improved Complexity-Reliability Trade-Off in Massive MIMO Systems](https://ieeexplore.ieee.org/document/9464346) by **Alva Kosasih**. It has three modules: BSO does the parallel interference cancellation, BSE does the Bayesian symbol estimation, and DSC does the update.
> Kosasih, A., Miloslavskaya, V., Hardjawana, W., She, C., Wen, C. K., & Vucetic, B. (2021). A Bayesian receiver with improved complexity-reliability trade-off in massive MIMO systems. *IEEE Transactions on Communications*, 69(9), 6251-6266.

## How to install
Currently, we offer three options to install this tool.
* Install through `Matlab Add-Ons`
    * Install through Matlab `Get Add-Ons`: search `whatshow_phy_detect_bpic` and install it.
    * Install through `.mltbx`: Go to ***Releases*** to download the file in the latest release to install.
* Install through `pip`
    ```sh
    pip install whatshow-phy-detect-bpic
    ```
    * **import this module**
        ```
        from whatshow_phy_detect_bpic import BPIC
        ```
* Install through `git` under another local repositiory
    ```sh
    git submodule add git@github.com:whatshow/Phy_Detect_BPIC.git Modules/Detect_BPIC
    ```
    * **import this module**
        * Matlab
            ```matlab
            addpath("Modules/Detect_BPIC");
            ```
        * Python
            ```python
            if '.' not in __name__ :
                from Modules.Detect_BPIC.BPIC import BPIC
            else:
                from .Modules.Detect_BPIC.BPIC import BPIC
            ```

## How to use
All Bayesian PIC-DSC detector codes are uniform in matlab and python as a class of `BPIC`. This class is the whole process of the detection. This section will illustrate the methods of this class following the detection process.
* BPIC<br>
    `@constellation:` the constellation, a vector.<br>
    `@bso_mean_init`: 1st iteration method in **BSO** to calculate the mean. Default: `BPIC.BSO_INIT_MMSE`, others: `BPIC.BSO_INIT_MRC, BPIC.BSO_INIT_ZF` (`BPIC.BSO_INIT_NO` should not be used but you can try)<br>
    `@bso_mean_cal`: other iteration method in **BSO** to calculate the mean. Default: `BPIC.BSO_MEAN_CAL_MRC` (`BPIC.BSO_MEAN_CAL_ZF` should not be used but you can try)<br>
    `@bso_var`: use approximate or accurate variance in **BSO**. Default: `BPIC.BSO_VAR_APPRO`, others: `BPIC.BSO_VAR_ACCUR`<br>
    `@bso_var_cal`: the method in **BSO** to calculate the variance. Default: `BPIC.BSO_VAR_CAL_MRC`, others: `BPIC.BSO_VAR_CAL_MRC` (`BSO_VAR_CAL_ZF` should not be used but you can try)<br>
    `@dsc_ise`: how to calculate the instantaneous square error. Default: `BPIC.DSC_ISE_MRC`, others: `BPIC.DSC_ISE_NO, BPIC.DSC_ISE_ZF, BPIC.DSC_ISE_MMSE`<br>
    `@dsc_mean_prev_sour`: the source of previous mean in DSC. Default: `BPIC.DSC_MEAN_PREV_SOUR_BSE`, others: `BPIC.DSC_MEAN_PREV_SOUR_DSC`<br>
    `@dsc_var_prev_sour`: the source of previous variance in DSC. Default: `BPIC.DSC_VAR_PREV_SOUR_BSE`, others: `BPIC.DSC_VAR_PREV_SOUR_DSC`<br>
    `@min_var`: the minimal variance.<br>
    `@iter_num`: the maximal iteration.<br>
    `@iter_diff_min`: the minimal difference in **DSC** to early stop.<br>
    `@detect_sour`: the source of detection result. Default: `BPIC.DETECT_SOUR_DSC`, others: `BPIC.DETECT_SOUR_BSE`.<br>
    ```sh, c, matlab, python
    // paper version 1: for BSO, MMSE in 1st iteration but MRC in others
    bpic = BPIC(sympool);
    // paper version 2: MRC in all iterations
    bpic = BPIC(sympool, "bso_mean_init", BSO_MEAN_INIT_MRC); % matlab
    bpic = BPIC(sympool, bso_mean_init=BSO_MEAN_INIT_MRC); # python
    // other configurations
    % matlab
    bpic = BPIC(sympool, "bso_mean_init", BPIC.BSO_MEAN_INIT_MMSE, "bso_var", BPIC.BSO_VAR_APPRO, "bso_var_cal", BPIC.BSO_VAR_CAL_MMSE, "dsc_ise", BPIC.DSC_ISE_MMSE, "detect_sour", BPIC.DETECT_SOUR_BSE);
    # python
    bpic = BPIC(sympool, bso_mean_init=BPIC.BSO_MEAN_INIT_MMSE, bso_var=BPIC.BSO_VAR_APPRO, bso_var_cal=BPIC.BSO_VAR_CAL_MMSE, dsc_ise=BPIC.DSC_ISE_MMSE, detect_sour=BPIC.DETECT_SOUR_BSE);
    ```
* detect: the estimated symbols from Tx<br>
    `@y`: the received signal, a vector<br>
    `@H`: the channel matrix, a matrix<br>
    `@No`: the noise power, a scalar<br>
    ```sh, c, matlab, python
    // symbol estimation - soft
    x_est = bpic.detect(y, H, No);
    // symbol estimation - hard
    x_est = bpic.detect(y, H, No, "sym_map", true); % matlab
    x_est = bpic.detect(y, H, No, sym_map=true); # python
    ```
## Samples
Before running any sample code, please make sure you are at the root path of this repository. Also, Matlab codes require running `init` in the command window first to load directories.
* `Test`
    * `Test/test_mimo`: test the performance of Bayesian PIC-DSC detector in MIMO case.
    * `Test/test_case_01`: compare the output from this module and Alva's original code.