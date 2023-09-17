##
# Personal script created by M.Sc Jose Gabriel Cordoba
# 2023, Do not disclose


import scipy.io as sio
import matplotlib.pyplot as plt


from Functions.f_TRC_Reader import *
from Functions.f_SignalProcFuncLibs import *

str_DataPath = 'RawData/'  # path file where the data is stored
str_OutPath = 'Data/'  # path file where the data will be stored

if not os.path.isdir(str_OutPath):  # Create the path if this doesn't exist
    os.mkdir(str_OutPath)

str_ReadName = str_DataPath + 'EEG.TRC'  # Name of the file
str_SaveName = str_OutPath + 'EEGData'  # Name of the new file
str_ChannStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2']  # Labels of channels to extract

m_AllData = f_GetSignalsTRC(str_ReadName, str_ChannStr)
dict_TRCHead = f_GetTRCHeader(str_ReadName)  # Function that extracts the header of the TCR
d_SampleRate = dict_TRCHead['RecFreq']  # Sample frequency

m_DataFilt = [] #Matrix with filtered data
for i_chann in range(len(str_ChannStr)): #For each channel
    print(f'##################################################')
    print(f'TRC - processing channel: {str_ChannStr[i_chann]}')
    print(f'##################################################')
    v_Data = m_AllData[i_chann] #Select the channel
    st_Filt = f_GetIIRFilter(d_SampleRate, [1,30], [0.95, 31])
    v_DataFilt = f_IIRBiFilter(st_Filt, v_Data) #1p Filtro creado, señal a filtrar
    m_DataFilt.append(v_DataFilt)

v_TimeArray = np.arange(0, np.size(v_DataFilt)) / d_SampleRate  # Time values

str_MyRed = '#7B241C'
str_MyBlue = '#0C649E'

fig, ax = plt.subplots(len(str_ChannStr), 1, sharex=True,  sharey=True)
for i in range(len(str_ChannStr)):
    ax[i].plot(v_TimeArray, m_AllData[i], linewidth=0.85, color=str_MyRed, alpha=0.4, label='RawData')
    ax[i].plot(v_TimeArray, m_DataFilt[i], linewidth=1.5, color=str_MyBlue, label='Filtered Data')
    ax[i].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[i].set_ylabel(f'{str_ChannStr[i]}')
    ax[i].set_ylim(-500,500)

ax[i].set_xlabel('Time (Seconds)')
plt.subplots_adjust(hspace=0)
plt.legend()
plt.show()
sio.savemat(str_SaveName + '.mat', mdict={'m_Data': m_DataFilt,
                                          'v_ChanNames': str_ChannStr,
                                          's_Freq': d_SampleRate})

print(f'---------------------------------------------')
