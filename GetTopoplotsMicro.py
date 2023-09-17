import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.integrate import simps
from Topoplots_functions import *


str_DataPath = 'C:/Users/afcad/Desktop/Taller EEG/Neuroingenieria EMB/Code/RawData/'  # path file where the data is stored
str_FileName = 'EEGData.mat'  # Name of the File

dict_allData = sio.loadmat(str_DataPath + str_FileName) 

d_SampleRate = np.double(dict_allData['s_Freq'][0]) 

#index_sta = 123*d_SampleRate
#index_end = 249*d_SampleRate

index_sta = 270*d_SampleRate
index_end = 374*d_SampleRate

#m_Data = dict_allData['m_Data'][:,int(index_sta):int(index_end)] #2:03-4:00
m_Data = dict_allData['m_Data'][:,int(index_sta):int(index_end)] #4:30-5:44

v_ChanNames = np.array(dict_allData['v_ChanNames'])
    
v_FreqBands = [[1, 4], [4, 8], [8, 12], [18, 30]]
v_FreqBands_Names = ['Delta', 'Theta', 'Alpha', 'Fast Beta']
    
d_WindSec = 3
d_stepSec = 1.5

m_PSDData_MF = getWelch(m_Data, d_SampleRate, v_ChanNames, v_FreqBands, v_FreqBands_Names, d_WindSec, d_stepSec)  # Get PSD data for MF

str_FileName_Ref = 'EEGData.mat'  # Name of the File

dict_allData = sio.loadmat(str_DataPath + str_FileName_Ref)  # Load .mat data

index_sta = 0*d_SampleRate
index_end = 120*d_SampleRate

m_Data_R = dict_allData['m_Data'][:,int(index_sta):int(index_end)]  #0-2min

m_PSDData_R = getWelch(m_Data_R, d_SampleRate, v_ChanNames, v_FreqBands, v_FreqBands_Names, d_WindSec, d_stepSec)   # Get PSD data for R

### normalize data

m_PSDData_MF_norm = normalize(m_PSDData_MF, m_PSDData_R) 

### mean of normalized data

m_PSDData_MF_norm_mean = meanofNormalizedData(m_PSDData_MF_norm)

##PSD Evolution Graph

v_TimeArray = np.arange(0,len(m_PSDData_MF[0][0]))*d_stepSec
fig, axs = plt.subplots(len(v_FreqBands), 1, figsize=(10, 8))
for i in range(len(v_FreqBands)):
    for j in range(len(v_ChanNames)):
        axs[i].plot(v_TimeArray, m_PSDData_MF_norm[j][i], linewidth=0.5)
        axs[i].set_title("Ondas " + str(v_FreqBands_Names[i]))
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Amplitude")
        axs[i].set_ylim([0, 10])
fig.suptitle("      PSD Evolution", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()


##Save data in mat

str_DataPath_Save='C:/Users/afcad/Desktop/Taller EEG/Neuroingenieria EMB/'

#sio.savemat(str_DataPath_Save + 'm_PSDData_MF_norm_mean.mat', mdict={'m_Data': m_PSDData_MF_norm_mean,
#                                             'v_ChanNames': v_ChanNames,
#                                             'chanlocs': dict_allData['chanlocs'],
#                                             'd_SampleRate': d_SampleRate,
#                                             'v_FreqBands': v_FreqBands})

### topos

getTopoplots(m_PSDData_MF_norm_mean, v_FreqBands_Names,8)

### connectivity
connectivity(m_PSDData_MF_norm, v_FreqBands_Names)
#print(Connectivity(m_PSDData_MF_norm_mean))

