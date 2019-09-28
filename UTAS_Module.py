"""
Completed on Tue, June 11,  2019
UTAS Module
@author: Eric Starner

This module is a direct translation of the MATLAB UTAS code utilized by H-1
Drives diagnostic team.

"""
## Imports for Module functionality ##
import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import butter, lfilter, hilbert
import pandas as pd
import warnings
import matplotlib.pyplot as plt


def load_data(fp):
    dataFile = pd.read_csv(fp)
    return dataFile['Value'].values * 32786/32768 # scale to correct for mdan translation bug


def envXfrm( AD, F, B, lwF,hiF ):
    """
    % Enveloping routine
    % Implements ZOOM FFT with Overlap-Add DFT Convolution Filtering 

    % <<<<< Inputs >>>>>
    % AD:  raw accelerometer data
    % F :  the sample rate
    % B :  input block size (FFT size, a power of 2)
    % lwF : low end of the band pass range (Hz)
    % hiF : high end of the band pass range (Hz)
    
    """
    In = AD[:]
    
    NewFs = F / np.floor(F / (2 * (hiF - lwF)))
    
    # ---------------------------------------------------
    # ZOOM FFT with Overlap-Add DFT Convolution Filtering 
    # ---------------------------------------------------
    
    hB = B // 2 # half block size
    D = (hiF - lwF) / (2 * F) # difference frequency
    j = np.arange(0, B+1) 
    k = abs(j - hB)
    
    # suppress division warning that doesnt affect calculations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = np.divide(np.sin(2 * D * np.pi * k), (np.pi * k))
   
    
    f[hB] = 2 * D # limit as k approaches zero (avoids divide by zero)
    # Apply a hamming window
    # NOTE: length, N = B + 1, thus N - 1 = B
    f = np.multiply(f, (0.54 - 0.46 * np.cos((2 * np.pi * j) / B)))
    
    # Normalize the time domain coefficients
    f = f / abs(sum(f))
    
    # Pad hi half
    f = np.append(f, np.zeros((B - 1, 1))) # variable f now has a size of 2 * B
    
    # take the fft
    f = fft(f)
    
    # clear unneeded variables
    del D, j, k
    
    # Computer the envelope transform
    C = (hiF + lwF) / (2 * F) # the center frequency ratio
    D = np.floor(F / (2 * (hiF - lwF))) # decimation factor
    i = 0 + 1j
    N = np.size(In)
    
    j = np.arange(0, N)
    
    # Heterodyne the input vector
    In = np.multiply(In, np.exp(-2 * np.pi * i * j * C))
    
    
    Hdyn = np.real(In);
    
    # preallocate the accumulation array
    j = N + B
    y = np.zeros((j, 1)) + 0j
    
    L = np.floor(N / B) - 1
    
    # --- This loop implements filtering using Overlap-Add DFT Convolution ---
    # REFERENCE:
    # John G. Proakis and Dimitris G. Manolakis, Digital Signal Processing:
    # Principles, Algorithms, and Applications, 4th edition, 2007.
    # The padded DFT is length 2*B b/c the filter is length B+1, not B
    
    for k in range(0, int(L + 1)):
        j = np.arange(0, B)
        z = In[k * B + (j)] # get B - element block of raw data
        
        # pad to 2*B elements
        z = np.append(z, np.zeros((B, 1)))
        fft_z = fft(z)
        
        z = ifft(np.multiply(f, fft_z))
        
        for j in range(0, 2*B):
            y[k * B + (j)] = y[k * B + (j)] + z[j]
        
    
    # Decimate, discarding first and last "B" points
    j = np.arange(np.floor(B / D), np.ceil((B / D) * np.floor(N / B)))
    out = np.abs(y[(D * j).astype(int)-1])
    
    fOut = np.empty(shape=np.shape(j))
    for i,v in enumerate(out):
        fOut[i,] = v
    
    return fOut, NewFs, Hdyn


def utas_env_spec(env, F, B, V):
    """
    % env = enveloped time domain data
    % F =  sampling frequency
    % B = window length
    % V = overlap
    """
    
    # calculate the env spectrum
    N = len(env) # 3073
    K = 2
    M = np.mean(env)
    
    L = int(np.fix((N - V) / (B - V)))
    
    hann = 1 / 2 * (1 - np.cos(2 * np.pi * np.arange(0, B) / (B - 1)))
    nfft = len(hann)
    
    # preallocate spectrum array
    spec = np.zeros((nfft,))
    for i in range(1, L + 1):
        # define the overalp window
        X = env[(i - 1) * (B - V): (i - 1) * (B - V) + B]
        # get the fft of the overlap window
        temp_fft = fft(np.multiply(hann, (X - M)))
        # square the absolute of the fft and add to the spectrum value
        spec = spec + np.power(np.abs(temp_fft / B), K)
        
    if nfft % 2:
        select = np.arange(1, ((nfft + 1) / 2) + 1)
    else:
        select = np.arange(1, (nfft/2) + 2)
    
    spec = spec[select.astype(int)]
    
    freq = (select - 1) * F / nfft
    
    # from utas analysis documentation
    Hj = 16 * B / (3 * F)
    spec = Hj * spec / L
    
    return spec, freq


def pat_env_filt(ts, F, lwF, hiF):
    ''' High Frequency Enveloping of Signal '''
    
    freq = F
    
    # bandpass filter the signal via Butterworth filter
    order = 10
    a, b = butter(order, np.divide(np.array([lwF, hiF]), (F / 2)), btype='bandpass')
    ts_bp = lfilter(a, b, ts)
    
    # Get the hilbert transform of the bandpassed signal
    upper1 = hilbert(ts_bp)
    
    rect_ts = np.abs(upper1)
    
    return rect_ts, freq


def pat_env_spec(env, F, lwF, hiF):
    
    N1 = len(env)
    
    NFFT1 = np.power(2, np.ceil(np.log2(np.abs(N1))))
    
    # create the frequency vector from the sample frequency and the number of
    # points in the FFT
    freq = np.multiply(np.arange(NFFT1), (F / NFFT1)) # frequency vector
    
    # calculate what datapoint is associated with the Nyquest cut off
    f_nqmax1 = int(np.fix((hiF - lwF) / (freq[1] - freq[0])))
    
    # cut off the frequency vector at the Nyquest cutoff which is 1/2 of the 
    # sample frequency
    freq = freq[:f_nqmax1 + 1] # frequency vector cutoff at nyquest
    
    chlrect = env - np.mean(env)
    chlrect_fft = np.power(np.divide(np.fft.fft(chlrect), N1), 2)
    chlrect_nq = 2 * np.abs(chlrect_fft[:len(freq)])
    
    spec = chlrect_nq
    
    return spec, freq


if __name__ == '__main__':
    ''' Main Function:
        ## Evaluation_wrapper in matlab code
        This allows the module to run as a self contained program.
    '''
    
    # load in MDAN data
    ts = load_data('PC2/168796planet_raw.csv')
    
    # sample rate
    sr = 104167
    # low frequency bandwidth cutoff
    lwF = 13000
    # High frequency bandwidth cutoff
    hiF = 18000
 
    
    # UTAS envelope process
    B = 1024 # reference ASM VPUA config, env FFTsize = 10, 2^10 = 1024
    utas_env, utas_f, Hd = envXfrm(ts, sr, B, lwF,hiF ) # Hd is real Heterodyned signal
    
    # calculate the envelope spectrum
    B = 2048 # largest B to ensure one overlap with two windows
    V = 1024 
    spec, freq = utas_env_spec(utas_env, utas_f, B, V)
    
    
    # Pat's Envelope Process
    pat_env, pat_f = pat_env_filt(ts, sr, lwF, hiF)
    pat_spec, pat_freq = pat_env_spec(pat_env, pat_f, lwF, hiF)
    pat_spec_env, pat_freq_env = utas_env_spec(pat_env, pat_f, 8192, 4096)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.plot(freq, spec)
    ax1.set_title('UTAS Env Spec Spectrum')
    ax1.set_xlim([0, 5000])
    ax1.set_ylim([0, np.max(spec) *1.1])
    
    ax2.plot(pat_freq, pat_spec)
    ax2.set_title('Pat Env Spectrum')
    ax2.set_xlim([0, 5000])
    ax2.set_ylim([0, np.max(pat_spec) *1.1])
    
    ax3.plot(pat_freq_env, pat_spec_env)
    ax3.set_title('Pat Env Spectrum')
    ax3.set_xlim([0, 5000])
    ax3.set_ylim([0, np.max(pat_spec_env) *1.1])

    
    
    # Generate plots
    fig.subplots_adjust(hspace=.5)
    plt.show()
    
    
    
    
    
    
    