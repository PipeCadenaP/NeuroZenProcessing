
import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.integrate import simps
from visbrain.objects import TopoObj, ColorbarObj, SceneObj
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from scipy.stats import pearsonr

def getWelch(m_Data, d_SampleRate, v_ChanNames, v_FreqBands, v_FreqBands_Names, d_WindSec, d_stepSec):

    d_WindSam = d_WindSec * d_SampleRate
    d_stepSam = d_stepSec * d_SampleRate

    m_PSDData = []

    for i_chann in range(len(v_ChanNames)):

        print(f'PSD - processing channel: {v_ChanNames[i_chann]}')

        m_ChannData = [[], [], [], []]
        v_ChannData = np.array(m_Data[i_chann])
        d_indexStart = 0
        d_indexEnd = int(d_indexStart + d_WindSam)

        while d_indexEnd <= len(v_ChannData):
            i_dataWind = v_ChannData[d_indexStart:d_indexEnd]
            freqs, psd = welch(i_dataWind, d_SampleRate, nfft=len(i_dataWind))
            freq_res = freqs[1] - freqs[0]
            
            for i_band in range(len(v_FreqBands)):
                d_MinFreq = v_FreqBands[i_band][0]
                d_MaxFreq = v_FreqBands[i_band][1]
                idx_band = np.logical_and(freqs >= d_MinFreq, freqs <= d_MaxFreq)
                
                d_BandPSD = simps(psd[idx_band], dx=freq_res)
                m_ChannData[i_band].append(d_BandPSD)

            d_indexStart = int(d_indexStart + d_stepSam)
            d_indexEnd = int(d_indexStart + d_WindSam)

        m_PSDData.append(np.array(m_ChannData))

    return m_PSDData

def normalize(m_PSDData_MF, m_PSDData_R):

    m_PSDData_MF_norm = []

    for i_chann in range(len(m_PSDData_MF)):

        m_PSDData_MF_norm.append([])
        
        for i_band in range(len(m_PSDData_MF[i_chann])):

            m_PSDData_MF_norm[i_chann].append(np.subtract(m_PSDData_MF[i_chann][i_band],np.mean(m_PSDData_R[i_chann][i_band]))/ np.std(m_PSDData_R[i_chann][i_band]))

    return m_PSDData_MF_norm



def meanofNormalizedData(m_PSDData_MF_norm):

    m_PSDData_MF_norm_mean = []

    for i_chann in range(len(m_PSDData_MF_norm)):

        m_PSDData_MF_norm_mean.append([])
        
        for i_band in range(len(m_PSDData_MF_norm[i_chann])):

            m_PSDData_MF_norm_mean[i_chann].append(np.mean(m_PSDData_MF_norm[i_chann][i_band]))

    return np.array(m_PSDData_MF_norm_mean)



def getTopoplots(m_PSDData_MF_norm_mean, v_FreqBands_Names, elec, v_ChanNames):

    sc = SceneObj(bgcolor='white', size=(1500 * (len(v_FreqBands_Names)-1), 1000))
    for i_band in range(len(v_FreqBands_Names)):
        v_Data = m_PSDData_MF_norm_mean[:, i_band]
        d_clim = np.max(np.abs([np.min(v_Data), np.max(v_Data)]))
        v_clim = [-d_clim, d_clim]
    #'PuBu'
        kw_top = dict(margin=25 / 100, chan_offset=(0.1, 0.1, 0.), chan_size=5, levels=5, cmap='jet',
                    level_colors='k',clim = v_clim)
        kw_cbar = dict(cbtxtsz=10, txtsz=10., width=.5, txtcolor='black', cbtxtsh=1.8, rect=(0., -1.5, 1., 3),
                    border=True)
        kw_title = dict(title_color='black', title_size=9.0, width_max=300)
        if elec==61:
            ch_names = ['Fp1', 'Fpz','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2',
                    'CP6','P7','P3','Pz','P4','P8','POz','O1','O2','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FC3','FCz','FC4',
                    'C5','C1','C2','C6','CP3','CP4','P5','P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8','Oz']
        if elec==8:
            ch_names = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2']
        ch_names = v_ChanNames
        t_obj = TopoObj('topo', v_Data, channels=ch_names, chan_mark_symbol=None, **kw_top)
        sc.add_to_subplot(t_obj, row=0, col=i_band, title=f' {v_FreqBands_Names[i_band]} ', **kw_title)


    cb_obj1 = ColorbarObj(t_obj, cblabel='Z-Score PSD', **kw_cbar)
    sc.add_to_subplot(cb_obj1, row=0, col=(i_band + 1), width_max=45)

    sc.preview()
    
    
def connectivity(m_PSDData_MF_norm, v_FreqBands_Names,v_ChanNames):

    m_Data = m_PSDData_MF_norm

    d_WindSec = 25
    d_stepSec = 10
    d_SampleRate = 1/2.5

    d_WindSam = int(d_WindSec * d_SampleRate)
    d_stepSam = int(d_stepSec * d_SampleRate)

    m_AllCorrelation_Dur = []

    for i_band in range(len(m_Data[0])):

        print(f'##################################################')
        print(f'PSD - processing freq band: {v_FreqBands_Names[i_band]}')
        print(f'##################################################')

        m_channCorrelation_Dur = np.zeros([len(m_Data), len(m_Data)])
    

        for i_chan1 in range(len(m_Data)):
            v_PSDEvolution1 = m_Data[i_chan1][i_band]

            for i_chan2 in range(len(m_Data)):
                v_PSDEvolution2 = m_Data[i_chan2][i_band]

                if i_chan1 != i_chan2:

                    d_indexStart = 0
                    d_indexEnd = int(d_indexStart + d_WindSam)
                    v_PhaseCorr = []
                    d_count = 0
                    while d_indexEnd <= len(v_PSDEvolution1):

                        i_dataWind1 = v_PSDEvolution1[d_indexStart:d_indexEnd]
                        i_dataWind2 = v_PSDEvolution2[d_indexStart:d_indexEnd]

                        corrpearson_Dr = pearsonr(i_dataWind1, i_dataWind2)[0]
                        v_PhaseCorr.append(corrpearson_Dr)
                        d_count += 1

                        d_indexStart = int(d_indexStart + d_stepSam)
                        d_indexEnd = int(d_indexStart + d_WindSam)

                    m_channCorrelation_Dur[i_chan1, i_chan2] = np.mean(v_PhaseCorr)

                else:

                    m_channCorrelation_Dur[i_chan1, i_chan2] = 0

        m_AllCorrelation_Dur.append(m_channCorrelation_Dur)


    df = pd.DataFrame(m_AllCorrelation_Dur[0])
    corr = df.corr()
    
    #tick_labels =  ['Fp1', 'Fpz','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2',
     #           'CP6','P7','P3','Pz','P4','P8','POz','O1','O2','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FC3','FCz','FC4',
      #          'C5','C1','C2','C6','CP3','CP4','P5','P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8','Oz']

    tick_labels = v_ChanNames
    
    
    sn.heatmap(corr, xticklabels=tick_labels, yticklabels=tick_labels, cmap = 'coolwarm')


    title_props = {'family': 'serif', 'size': 20, 'weight': 'bold', 'color': 'black'}
    plt.title("Effective connectivity (weighted directed network)", fontdict=title_props)
    plt.show()
    


