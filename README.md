# Bayesian PIC-DSC detector
This detection method is proposed in [A Bayesian Receiver With Improved Complexity-Reliability Trade-Off in Massive MIMO Systems](https://ieeexplore.ieee.org/document/9464346) by **Alva Kosasih**. It has three modules: BSO does the parallel interference cancellation, BSE does the Bayesian symbol estimation, and DSC does the update.
> Kosasih, A., Miloslavskaya, V., Hardjawana, W., She, C., Wen, C. K., & Vucetic, B. (2021). A Bayesian receiver with improved complexity-reliability trade-off in massive MIMO systems. *IEEE Transactions on Communications*, 69(9), 6251-6266.
* **In another local repositiory, add this module**
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

## Samples
Before running any sample code, please make sure you are at the root path of this repository. Also, Matlab codes require running `init` in the command window first to load directories.
* `Test`
    * `Test/test_mimo`: test the performance of Bayesian PIC-DSC detector in MIMO case.
    * `Test/test_case_01`: compare the output from this module and Alva's original code.