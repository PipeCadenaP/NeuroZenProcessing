import numpy as np
import scipy.io as sio
from Functions.f_SignalProcFuncLibs import *
from scipy.signal import welch
from scipy.integrate import simps
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.close()
import matplotlib.pylab as pl

##
str_DataPath = 'Data/'  # path file where the data is stored
str_FileName = 'EEGData.mat'  # Name of the File
str_MyRed = '#7B241C'
str_MyBlue = '#0C649E'

dict_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data

m_Data = dict_allData['m_Data']
d_SampleRate = np.double(dict_allData['s_Freq'][0])
v_ChanNames = np.array(dict_allData['v_ChanNames'])
v_TimeArray = np.arange(0, np.size(m_Data[0])) / d_SampleRate  # Time values


v_FreqBands = [[1, 4], [8, 12], [18, 30]]
v_FreqBands_Names = ['Delta', 'Alpha', 'Fast Beta']  # Names of the frequency bands

d_WindSec = 5 #Segundos
d_stepSec = 2.5 #Segundos

d_WindSam = d_WindSec * d_SampleRate
d_stepSam = d_stepSec * d_SampleRate

m_PSDData = []

for i_chann in range(len(v_ChanNames)):

    print(f'##################################################')
    print(f'PSD - processing channel: {v_ChanNames[i_chann]}')
    print(f'##################################################')

    m_ChannData = [[], [], []]

    v_ChannData = np.array(m_Data[i_chann])


    d_indexStart = 0
    d_indexEnd = int(d_indexStart + d_WindSam)

    while d_indexEnd <= len(v_ChannData):
        i_dataWind = v_ChannData[d_indexStart:d_indexEnd]


        freqs, psd = welch(i_dataWind, d_SampleRate, nfft=len(i_dataWind))
        freq_res = freqs[1] - freqs[0]
        # Find the closest indices of band in frequency vector

        for i_band in range(len(v_FreqBands)):
            d_MinFreq = v_FreqBands[i_band][0]
            d_MaxFreq = v_FreqBands[i_band][1]
            idx_band = np.logical_and(freqs >= d_MinFreq, freqs <= d_MaxFreq)
            # Integral approximation of the spectrum using Simpson's rule.
            d_BandPSD = simps(psd[idx_band], dx=freq_res)
            m_ChannData[i_band].append(d_BandPSD)

        d_indexStart = int(d_indexStart + d_stepSam)
        d_indexEnd = int(d_indexStart + d_WindSam)

    m_PSDData.append(np.array(m_ChannData))


v_SessionTimes = [120, 240]

fig, ax = plt.subplots(len(v_FreqBands), 1, sharex=True, figsize=(11, 6))
v_TimeArray = np.arange(0, len(m_PSDData[0][0])) * d_stepSec
AverageWind = 1
colors = pl.cm.Reds(np.linspace(0, 1, len(m_PSDData) + 2))

for i_band in range(len(v_FreqBands)):
    i_max = 0
    i_min = 1
    for i_chan in range(len(v_ChanNames)):
        v_Data = m_PSDData[i_chan][i_band]

        d_maxValue = np.mean(v_Data) + (np.std(v_Data) * 2)
        d_minValue = np.mean(v_Data) - (np.std(v_Data) * 2)

        v_Data[v_Data > d_maxValue] = np.mean(v_Data)
        v_Data[v_Data < d_minValue] = np.mean(v_Data)


        v_DataNorm = (v_Data - np.mean(v_Data[0:int(v_SessionTimes[0]/d_stepSec)])) / np.std(
            v_Data[0:int(v_SessionTimes[0]/d_stepSec)])


        m_PSDData[i_chan][i_band] = v_DataNorm
        v_Data_Soft = AverageMean(v_DataNorm, AverageWind)

        if i_max < max(v_Data_Soft):
            i_max = max(v_Data_Soft)
        if i_min > min(v_Data_Soft):
            i_min = min(v_Data_Soft)

        ax[i_band].plot(v_TimeArray, v_Data_Soft, color=colors[i_chan + 2], label=v_ChanNames[i_chan])

    if i_band == 0:
        ax[i_band].legend(ncol=1, bbox_to_anchor=(1.125, -0.5), fancybox=True, shadow=True, fontsize=7)
        if len(v_SessionTimes) == 2:
            ax[i_band].text(v_SessionTimes[0] - 5, i_max * 1.5, 'Start', rotation=90,
                            fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props
            ax[i_band].text(v_SessionTimes[1] - 5, i_max * 1.5, 'End', rotation=90,
                            fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props
        else:
            ax[i_band].text(v_SessionTimes[0] - 5, i_max * 1.5, 'Start', rotation=90,
                            fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props

    if len(v_SessionTimes) == 2:
        ax[i_band].axvline(v_SessionTimes[0], linewidth=1, linestyle='--', color='k')
        ax[i_band].axvline(v_SessionTimes[1], linewidth=1, linestyle='--', color='k')
    else:
        ax[i_band].axvline(v_SessionTimes[0], linewidth=1, linestyle='--', color='k')

    ax[i_band].grid(linewidth=0.4, linestyle=':', color='k', which='both')
    ax[i_band].set_ylabel(f'{v_FreqBands_Names[i_band]}')
    ax[i_band].yaxis.tick_right()
    if i_min < 0:
        ax[i_band].set_ylim([i_min * 1.4, i_max * 1.4])
    else:
        ax[i_band].set_ylim([i_min * 0.6, i_max * 1.4])
    ax[i_band].yaxis.set_tick_params(labelsize=6)
    ax[i_band].set_xlabel('Time (s)')

fig.supylabel('FrequencyBand')
fig.suptitle('Evolution PSD', fontsize='18')
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()
sio.savemat(str_DataPath + 'PSD.mat', mdict={'m_Data': m_PSDData,
                                             'v_ChanNames': v_ChanNames,
                                             'd_SampleRate': d_SampleRate,
                                             'v_FreqBands': v_FreqBands,
                                             'v_SessionTimes': v_SessionTimes})
