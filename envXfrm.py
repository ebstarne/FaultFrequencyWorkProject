# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:20:34 2019

@author: eric.starner

"""
import numpy as np
from numpy.fft import fft, ifft
import pandas as pd
import warnings
import matplotlib.pyplot as plt

def load_data(fp):
    dataFile = pd.read_csv(fp)
    return dataFile['Value'].values * 32786/32768, dataFile['X'] # scale to correct for mdan translation bug
    

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
    #decfactor  = np.floor(F / (2 * (hiF - lwF)))
    #zoomfactor = (np.floor(len(AD) / B) - 2) / decfactor
    

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
    
    L = int((N - V) / (B - V))
    
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

def utas_env_opt(defect_freq, freq, spec):
    """
    defect_freq: dictionary entry containing name:value
    freq: frequency data produced by utas_env_spec
    spec: spec data produced by utas_env_spec
    """
    freq_list_min = (np.abs((defect_freq - 5) - freq)).tolist()
    freq_list_max = (np.abs((defect_freq + 5) - freq)).tolist()
    start_index = freq_list_min.index(min(freq_list_min))
    end_index = freq_list_max.index(min(freq_list_max))
    
    return np.max(spec[start_index:end_index + 1])

def polynomial_eqn_eval(opt_freqs, degree=16):
    coefs = poly.polyfit(np.arange(len(opt_freqs)), opt_freqs, degree)
    ffit = poly.Polynomial(coefs)
    
    ### Finding maximum
    der = poly.polyder(coefs)
    der_2 = poly.polyder(coefs, m=2)
    ffit_2der = poly.Polynomial(der_2)
    crit_points = poly.polyroots(der).real
    

    maximums = {}
    for root in crit_points:
        if ffit_2der(root) < 0 and ffit(root) < 1:
            maximums[root] = ffit(root)
            
    key_max = max(maximums.keys(), key=(lambda k: maximums[k]))
    max_val = maximums[key_max]
    
    return key_max, max_val, ffit, crit_points

def generate_plot(opt_freqs, x, title_freq = '', degree_polynomial=16):
   key_max, max_val, ffit, crit_points =  polynomial_eqn_eval(opt_freqs, degree_polynomial)
  # spectrum_idx = ffit(x_new).tolist().index(max(ffit(x_new).tolist())) // 20
  # optimized_lwF = 9000 + spectrum_idx*100
  # optimized_hiF = 14000 + spectrum_idx*100
  
   max_pt = np.max(opt_freqs)
   max_pt_idx = np.argmax(opt_freqs)
   max_frequency = max_pt_idx * 10 + 10500
   
   str = "Optimized Natural Frequency for {}: {:,.4f}".format( title_freq, max_frequency ) #11500 , key_max * 10 + 10500
    

   #plt.plot(x, ffit(x))
   plt.plot(opt_freqs,'.-' )
   #plt.plot(crit_points[np.logical_and(crit_points > 0, crit_points < 50)], ffit(crit_points[np.logical_and(crit_points > 0, crit_points < 50)]), 'r*')
   #plt.plot(key_max, max_val, 'gp', markersize=15)
   plt.title(str)
   plt.xlabel('Natural Frequency Estimate (Hz)')
   plt.gca().set_facecolor('xkcd:black')
   plt.gcf().set_facecolor('xkcd:light grey')
   labels = np.append([0], np.round(np.linspace(10500, 20500, 6)))
   
   plt.gca().set_xticklabels(labels)
   plt.show()
   
   return max_frequency, max_pt

def generate_defect_dict():
    roller=647.1849161
    outer=694.5260817
    inner=1006.354118
    cage=57.8299268
    shaft=8256.7*1.03/60
    bpfi = 1285.014991
    bpfi2x = 2570.03
    bpfi_1lowersb = 1143.315
    bpfi_2lowersb = 1001.615
    bpfi_1uppersb = 1426.7
    bpfi_2uppersb = 1568.4
    shaft1 = 141.7
    shaft2 = 283.4
    

    
    defect_freqs = {'roller':roller, 'outer':outer, 'inner':inner, 'cage':cage, 'shaft':shaft, 'bpfi':bpfi, 'bpfi2x':bpfi2x, \
                    'bpfi_1lowersb':bpfi_1lowersb, 'bpfi_2lowersb':bpfi_2lowersb, 'bpfi_1uppersb':bpfi_1uppersb, 'bpfi_2uppersb':bpfi_2uppersb, \
                    'shaft1':shaft1, 'shaft2':shaft2}
    
    return defect_freqs
   



if __name__ == '__main__':
    ''' Main Function '''
    
    import numpy.polynomial.polynomial as poly
    import time
    
    # time program
    s = time.time()
    
    # load in MDAN data
    ts, x_time = load_data('PC2/167807planet_raw.csv')
    
    
    
    # generate dictionary of defect frequencies
    defect_freqs = generate_defect_dict()
    
    # specific componenet defect being evaluated
    component = 'bpfi'
    
    # sample rate
    sr = 104167
    
    # create empty array for optimal frequencies
    opt_freqs = np.empty(1001,)
    
    for i in range(1001):
        x = i * 10 # increment value
        lwF = 9000 + x # low frequency
        hiF = 14000 + x # high frequency
        c = .5 * (lwF + hiF)
    
        # UTAS envelope process
        B = 1024 
        utas_env, utas_f, Hd = envXfrm(ts, sr, B, lwF,hiF ) # Hd is real Heterodyned signal
    
    
        # calculate the envelope spectrum
        B = 2048
        V = 1024 # largest B to ensure one overlap with two windows
        
        # generating envelope spectrum data
        spec, freq = utas_env_spec(utas_env, utas_f, B, V)
        
        
        # finding optimal envelope frequency for specific defect
        opt_freqs[i] = utas_env_opt(defect_freqs[component], freq, spec)
        
        
       
    # x data for polynomial fit    
    x = np.linspace(0, 1000, 1000)
    
    # generate plots and output pertinent information
    max_frequency, max_amplitude = generate_plot(opt_freqs, x, degree_polynomial=24, title_freq=component)
   
    # time program
    e = time.time()
    print(e - s)
    
    
    
    
    
    
    
    
    
    
    '''
    
    x_new = np.linspace(0, 50, 1000)
    spectrum_idx = ffit(x_new).tolist().index(max(ffit(x_new).tolist())) // 50
    optimized_lwF = 9000 + spectrum_idx*100
    optimized_hiF = 14000 + spectrum_idx*100
    str = "Optimized Envelope: {:,} - {:,}\nOptimized Frequency: {:,.4f}".format(optimized_lwF, optimized_hiF, key_max * 100 + 9000 )
    
    
    plt.plot(x_new, ffit(x_new))
    plt.plot(opt_freqs,'.' )
    #plt.plot(x_new, ffit_1der(x_new))
    plt.plot(crit_points[np.logical_and(crit_points > 0, crit_points < 50)], ffit(crit_points[np.logical_and(crit_points > 0, crit_points < 50)]), 'r*')
    plt.plot(key_max, max_val, 'gp', markersize=15)
    #plt.plot(x_new, ffit_2der(x_new))
    plt.title(str)
    plt.xlabel('Frequency')
    plt.gca().set_facecolor('xkcd:black')
    plt.gcf().set_facecolor('xkcd:light grey')
    plt.show()

    '''

    
    
    
    """
    plt.plot(opt_freqs,'-' )
    plt.plot(opt_freqs,'*r', markersize=10)
    plt.show()
    """
    
    """
    plt.plot(freq, spec)
    plt.gca().set_facecolor('k')
    plt.xlabel('Frequency')
    plt.title('Low Freq. = ' + str(lwF) + ',    High Freq = ' + str(hiF) + \
              ',    Est. Natural Freq. = ' + str(c))
    
    plt.axvline(x=cage, color='w')
    plt.axvline(x=2*cage, color='w')
    plt.axvline(x=roller, color='c')
    plt.axvline(x=2*roller, color='c')
    plt.axvline(x=3*roller, color='c')
    plt.axvline(x=4*roller, color='c')
    
    plt.axis([0, 3000, 0, .009])
    
    plt.show()
    """
    
    
    
    
    