import numpy as np
import sys
sys.path.append("..");
from BPIC import BPIC

# batch_size
batch_size = 12;

# symbols & channel
SNR_dB = 10;
No = 10**(-SNR_dB/10);
sympool = [-0.948683298050514 + 0.948683298050514j,-0.948683298050514 + 0.316227766016838j,-0.948683298050514 - 0.948683298050514j,-0.948683298050514 - 0.316227766016838j,-0.316227766016838 + 0.948683298050514j,-0.316227766016838 + 0.316227766016838j,-0.316227766016838 - 0.948683298050514j,-0.316227766016838 - 0.316227766016838j,0.948683298050514 + 0.948683298050514j,0.948683298050514 + 0.316227766016838j,0.948683298050514 - 0.948683298050514j,0.948683298050514 - 0.316227766016838j,0.316227766016838 + 0.948683298050514j,0.316227766016838 + 0.316227766016838j,0.316227766016838 - 0.948683298050514j,0.316227766016838 - 0.316227766016838j];            
tx_num = 6;     # Tx antenna number
rx_num = 8;

# detection settings
iter_times = 10;

# simulation
# sim - Tx
# Create symbols
x = np.asarray([-0.948683298050514 + 0.948683298050514j,0.948683298050514 - 0.316227766016838j,-0.316227766016838 - 0.316227766016838j,0.948683298050514 - 0.948683298050514j,-0.316227766016838 - 0.948683298050514j,-0.316227766016838 - 0.316227766016838j]);
x = np.expand_dims(np.tile(x, (batch_size, 1)), axis=-1);


# sim - channel
# Rayleigh fadding channel
H = np.asarray([[0.0779842980275662 - 0.622759649033581j,-0.142875802631333 - 0.408488171598999j,-0.174323667474292 + 0.109478245089320j,-0.368966477081844 - 0.276921619825361j,0.00153623951780131 - 0.154502633467246j,-0.122964279233259 - 0.0164804978420562j],[-0.183137392048928 + 0.0271114323201603j,-0.180371427216861 - 0.00962913960198872j,-0.106442681285029 - 0.650355701183657j,-0.168757492195561 - 0.142235811473194j,0.328844132623829 - 0.0756248523168705j,-0.587173094540851 + 0.0842314391352496j],[0.181796970844150 - 0.274582437390115j,-0.269517005491472 + 0.0753782055348761j,-0.241958959549706 - 0.337273794607423j,-0.178807013491370 + 0.127322067357151j,0.123368565654773 - 0.316219049081301j,-0.383520086717652 - 0.177425089797263j],[-0.0228728145864274 + 0.338497555994102j,-0.0804609520079283 + 0.150975517592066j,-0.0815435205981982 + 0.346977587619771j,0.154528101457900 - 0.0591103469392555j,0.0522594800681023 + 0.158550051599941j,-0.0923430240796915 - 0.00393844929589852j],[0.397490985713344 + 0.500879032759204j,-0.0578865696423046 + 0.195371532926671j,1.03053206097488 - 0.190738801731184j,0.139426352501500 - 0.282268546757656j,0.189798052064987 - 0.627758884668258j,0.238281250454043 + 0.118966918969035j],[-0.406515930069576 - 0.107299405920014j,0.0394737737134317 + 0.194681257226061j,0.983669864502787 + 0.0945581499725686j,0.0313140593238832 - 0.454070567845001j,0.168657178927005 + 0.0417466539734718j,-0.0661911325773498 - 0.270531785576486j],[0.0407700751879210 + 0.344350212901169j,-0.255000850654891 + 0.195836910913300j,0.331086667554220 + 0.435358787558427j,-0.0982317389826576 - 0.106885307261230j,-0.466629295288548 + 0.193133316984322j,0.0757052888255151 + 0.241089069892722j],[0.372309919923952 + 0.275294472323379j,0.0238266459516009 - 0.0902339925878703j,0.227261674303623 - 0.256605913053121j,-0.267269417226659 - 0.0638391972248551j,0.00541968860813382 - 0.245035871941119j,-0.259224365936243 + 0.315966037522292j]])
H = np.tile(H, (batch_size, 1, 1));
# Nojse
noise = np.asarray([0.0718181951981610 - 0.0880111639136088j,0.00337933343174761 - 0.486637718583503j,-0.260388448145171 - 0.267356458760811j,0.110649754927316 + 0.0816899119588328j,-0.271872098044113 + 0.539438666768092j,0.355711005382155 + 0.165524310537081j,0.117723399711715 + 0.0692613980711147j,0.0592152372889183 - 0.121192018437500j]);
noise = np.expand_dims(np.tile(noise, (batch_size, 1)), axis=-1);
# through Rayleigh fadding channel to get y 
y = H@x + noise;


# sim - Rx
# BPIC - MMSE
bpic = BPIC(sympool, bso_var_cal=BPIC.BSO_VAR_CAL_MMSE, dsc_ise=BPIC.DSC_ISE_MMSE, detect_sour=BPIC.DETECT_SOUR_BSE, batch_size=batch_size);
syms_BPIC_MMSE = bpic.detect(y, H, No);
syms_BPIC_MMSE_mat = np.asarray([[-0.942655192229299 + 0.906258788176649j],[0.913885830049333 - 0.345609884043461j],[-0.312669600541753 - 0.315145911885506j],[0.636944663543423 - 0.471641689278999j],[-0.744111924324306 - 0.946999468925245j],[-0.306303586092814 + 0.320004589570229j]]).squeeze();
syms_BPIC_MMSE_diff = abs(syms_BPIC_MMSE - syms_BPIC_MMSE_mat);