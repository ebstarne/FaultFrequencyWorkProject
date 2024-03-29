"""
Natural Frequency Estimator

Self-contained module used to determine a natural frequency estimate concerning 
a set of BUNOs related to a specific failure.

Created on Mon Jun  24 2019

@author: eric.starner
"""
import numpy as np
from numpy.fft import fft, ifft
from numpy import multiply, divide, exp, pi, sin, power, sqrt
from scipy.signal import butter, lfilter, hilbert
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from statistics import stdev


def load_data(fp):
    dataFile = pd.read_csv(fp)
    return dataFile['Value'].values * 32786/32768, dataFile['X'].values # scale to correct for mdan translation bug

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
    mbd = 820.6
    

    
    defect_freqs = {'roller':roller, 'outer':outer, 'inner':inner, 'cage':cage, 
                    'shaft':shaft, 'bpfi':bpfi, 'bpfi2x':bpfi2x,
                    'bpfi_1lowersb':bpfi_1lowersb, 'bpfi_2lowersb':bpfi_2lowersb, 
                    'bpfi_1uppersb':bpfi_1uppersb, 'bpfi_2uppersb':bpfi_2uppersb,
                    'shaft1':shaft1, 'shaft2':shaft2, 'mbd':mbd}
    
    return defect_freqs

def env_filt_hilbert(ts, F, lwF, hiF):
    ''' High Frequency Enveloping of Signal '''
    
    # bandpass filter the signal via Butterworth filter
    order = 10
    a, b = butter(order, np.divide(np.array([lwF, hiF]), (F / 2)), btype='bandpass')
    ts_bp = lfilter(a, b, ts)
    
    # Get the hilbert transform of the bandpassed signal
    upper1 = hilbert(ts_bp)
    
    rect_ts = np.abs(upper1)
    
    return rect_ts, F

def env_spec_hilbert(env, F, lwF, hiF):
    
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

def uniqe_method(wave, F, lwF, hiF):
    # Constants defined
    B = len(wave)
    j = np.arange(B)
    k = np.abs(j.astype(float) - hB)
    k[B//2] = .000001
    i = 1j
    
    # x values defined
    fft_xVals = abs(F * np.arange(B + 1) / B)
    
    # center frequency (natural frequency estimate)
    c = (hiF + lwF) / (2 * F)
    
    # heterodyne and its fft
    hD = multiply(wave, exp(-2 * pi * i * j * c))
    fft_hD = fft(hD)
    
    # Draws unique filter wave around the defect frequency
    D = 2570/F
        
    # filter wave and its fft
    # suppress division warning that doesnt affect calculations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Unique filter generated by Robbie Kirk
        filter_wave = (divide(multiply(sin( pi * D * k), sin(2 * pi * D * k)), pi * k)) * 10000
    filter_wave[hB] = 2 * D
        
    fft_filter_wave = fft(filter_wave)

    filtered_wave = ifft(multiply(fft_filter_wave, fft_hD))
    return fft_xVals, abs(fft(abs(filtered_wave))/B)
    

def UTAS_method(wave, F, lwF, hiF):
    # Constants defined
    B = len(wave)
    j = np.arange(B)
    k = np.abs(j.astype(float) - hB)
    k[B//2] = .000001
    i = 1j
    
    # x values defined
    fft_xVals = abs(F * np.arange(B + 1) / B)
    
    # center frequency (natural frequency estimate)
    c = (hiF + lwF) / (2 * F)
    
    # heterodyne and its fft
    hD = multiply(wave, exp(-2 * pi * i * j * c))
    fft_hD = fft(hD)

    # filter wave and its fft
    # suppress division warning that doesnt affect calculations
    D = (hiF - lwF) / (2 * F) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filter_wave = divide(sin(2 * pi * D * k), pi * k)
    filter_wave[hB] = 2 * D
        
    fft_filter_wave = fft(filter_wave)

    filtered_wave = ifft(multiply(fft_filter_wave, fft_hD))
    return fft_xVals, abs(fft(abs(filtered_wave))/B)
    

def hilbert_method(wave, F, lwF, hiF):
    env, f = env_filt_hilbert(wave, F, lwF, hiF)
    return env_spec_hilbert(env, f, lwF, hiF)

def hfe_optimization(defect_freq, freq, spec):
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

def generate_optimal_frequencies(num_points, lwF_start, hiF_start, defect_frequency, wave, F):
    # Main for loop that exectues each method and generates optimal 
    # natural frequency data points
    for idx in range(num_points):
        # incremental values for low and high frequency window
        lwF = lwF_start + idx * defect_frequency
        hiF = hiF_start + idx * defect_frequency
        
        # Hilbert Method with Butterworth Filter
        hilbert_spec, hilbert_freq = hilbert_method(wave, F, lwF, hiF)
        opt_freqs_hilbert[idx] = hfe_optimization(defect_frequency, hilbert_freq, hilbert_spec)
        
        # UTAS FFT method with nomral bandpass filter
        fft_xVals, UTAS_spec = UTAS_method(wave, F, lwF, hiF)
        opt_freqs_UTAS[idx] = hfe_optimization(defect_frequency, fft_xVals, UTAS_spec)
        
        # UTAS FFT method with unique filter created by Robbie Kirk
        fft_xVals, unique_spec = uniqe_method(wave, F, lwF, hiF)
        opt_freqs_unique[idx] = hfe_optimization(defect_frequency, fft_xVals, unique_spec)
        
    return opt_freqs_hilbert, opt_freqs_UTAS, opt_freqs_unique

def poly_fit(defect_frequency, lwF_start, hiF_start, opt_freqs, num_points):
    start = (hiF_start + lwF_start) / 2
    X = np.arange(start, start + defect_frequency * num_points - 1, defect_frequency)
    
    coefs = poly.polyfit(X, opt_freqs, 5)
    ffit = poly.Polynomial(coefs)
    
    x_fit = np.linspace(start, start + defect_frequency * num_points - 1, 1000)
    y_fit = ffit(x_fit)
    
    return x_fit, y_fit
    

def generate_plots(defect_frequency, lwF_start, hiF_start, opt_freqs_hilbert, opt_freqs_UTAS,
                       opt_freqs_unique, file_name, num_points):
    x_fit_hilbert, y_fit_hilbert = poly_fit(defect_frequency, lwF_start, 
                                            hiF_start, opt_freqs_hilbert, num_points)
    
    x_fit_UTAS, y_fit_UTAS = poly_fit(defect_frequency, lwF_start, 
                                            hiF_start, opt_freqs_UTAS, num_points)
    
    x_fit_unique, y_fit_unique = poly_fit(defect_frequency, lwF_start, 
                                            hiF_start, opt_freqs_unique, num_points)
    
    m_hilbert = max(y_fit_hilbert[200:-200])
    m_UTAS = max(y_fit_UTAS[200:-200])
    m_unique = max(y_fit_unique[200:-200])
    
    max_idx_hilbert = [i for i, j in enumerate(y_fit_hilbert) if j == m_hilbert]
    max_idx_UTAS = [i for i, j in enumerate(y_fit_UTAS) if j == m_UTAS]
    max_idx_unique = [i for i, j in enumerate(y_fit_unique) if j == m_unique]
    
    
    ax1.plot(x_fit_hilbert, y_fit_hilbert / m_hilbert, label=file_name[4:-8] + 
             ' Optimized Natural Frequency: ' + str(x_fit_hilbert[max_idx_hilbert])[1:-7])
    ax1.set_ylim([0, 1.05])
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Magnitude (Normalized)')
    ax1.legend()
    
    ax2.plot(x_fit_UTAS, y_fit_UTAS / m_UTAS, label=file_name[4:-8] + 
             ' Optimized Natural Frequency: ' + str(x_fit_UTAS[max_idx_UTAS])[1:-7])
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude (Normalized)')
    ax2.legend()
    
    ax3.plot(x_fit_unique, y_fit_unique / m_unique, label=file_name[4:-8] + 
             ' Optimized Natural Frequency: ' + str(x_fit_unique[max_idx_unique])[1:-7])
    ax3.set_ylim([0, 1.05])
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Magnitude (Normalized)')
    ax3.legend()
    
    return x_fit_hilbert[max_idx_hilbert], x_fit_UTAS[max_idx_UTAS], x_fit_unique[max_idx_unique]

def finalize_plots(hilb, UTAS, unique):
    
    # Hilbert finalization of plot
    ax1.set_title('PC2 Quill Inner Race Hilbert Method\n Mean: ' + str(hilb[0]) + 
              '\nStandard Deviation: ' + str(hilb[1]))
    
    mu, sigma = np.mean(hilb[2]), stdev(hilb[2])
    
    ax1.plot([mu, mu], [0, 1.1], 'b--')
    ax1.plot([mu-sigma, mu-sigma], [0, 1.1], 'k', linewidth=5)
    ax1.plot([mu+sigma, mu+sigma], [0, 1.1], 'k', linewidth=5)
    
    # UTAS finalization of plot
    ax2.set_title('PC2 Quill Inner Race UTAS Method\n Mean: ' + str(UTAS[0]) + 
              '\nStandard Deviation: ' + str(UTAS[1]))
    
    mu, sigma = np.mean(UTAS[2]), stdev(UTAS[2])
    
    ax2.plot([mu, mu], [0, 1.1], 'b--')
    ax2.plot([mu-sigma, mu-sigma], [0, 1.1], 'k', linewidth=5)
    ax2.plot([mu+sigma, mu+sigma], [0, 1.1], 'k', linewidth=5)
    
    # Unique filter finalization fo plot
    ax3.set_title('PC2 Quill Inner Race Unique Method\n Mean: ' + str(unique[0]) + 
              '\nStandard Deviation: ' + str(unique[1]))
    
    mu, sigma = np.mean(unique[2]), stdev(unique[2])
    
    ax3.plot([mu, mu], [0, 1.1], 'b--')
    ax3.plot([mu-sigma, mu-sigma], [0, 1.1], 'k', linewidth=5)
    ax3.plot([mu+sigma, mu+sigma], [0, 1.1], 'k', linewidth=5)
    
def generate_standard_curve(max_freqs_hilbert, max_freqs_UTAS, max_freqs_unique):
    # When main loop finishes, mu and sigma are calculated for each method    
    mu_hilbert, sigma_hilbert = np.mean(max_freqs_hilbert), stdev(max_freqs_hilbert)
    mu_UTAS, sigma_UTAS = np.mean(max_freqs_UTAS), stdev(max_freqs_UTAS)
    mu_unique, sigma_unique = np.mean(max_freqs_unique), stdev(max_freqs_unique)
    
    finalize_plots((mu_hilbert, sigma_hilbert, max_freqs_hilbert), (mu_UTAS, sigma_UTAS, max_freqs_UTAS), 
                   (mu_unique, sigma_unique, max_freqs_unique))
    
    # mu and sigmal of all plots, used to determine overall mean and standard deviation
    mu_all, sigma_all = np.mean(max_freqs_hilbert + max_freqs_UTAS + max_freqs_unique), \
    stdev(max_freqs_hilbert + max_freqs_UTAS + max_freqs_unique)
    
    # final plot of starnd normal curve
    fig4, ax4 = plt.subplots()
    x_normal = np.linspace(10000, 20000, 20000)
    normal_f = 1 / (sigma_all * sqrt(2 * pi)) * np.exp(-power(x_normal - mu_all, 2) / (2 * sigma_all**2))
    ax4.plot(x_normal, normal_f)
    ax4.set_title("Normal Distribution")
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('f(x)')
    
    
###############################################################################  
if __name__ == '__main__':
    ''' Main Function '''
###############################################################################
    # clear workspace
    #get_ipython().magic('reset -sf')
    # generate dictionary of defect frequencies
    defect_freqs = generate_defect_dict()
 
    #BUNOs = (168412, 168796, 168955, 166768, 167807, 168958, 169233, 169287)
    BUNO_pc2 = (168412, 168796, 168955, 166768, 168958, 169233, 169287)
    
    BUNO_rotor = (168522, 169236,168514)
    
    BUNO_mbd = (167805, 168405, 168501)
    
    defect_frequency = defect_freqs['bpfi'] 
    
    max_freqs_hilbert = []
    max_freqs_UTAS = []
    max_freqs_unique = []
    
    for BUNO in BUNO_pc2:
        # Load data files
        file_name = 'PC2/' + str(BUNO) + 'planet_raw.csv'
        
        # x and y time wave form data
        wave, time = load_data(file_name)
    
        # Constants defined, related to data being analyzed
        F = 104167 # sample rate
        B = len(wave) # fft block size (same as size of data)
        hB = B // 2 - 1 # half block size
    
        # create empty arrays for optimal frequencies
        opt_freqs_UTAS = np.empty(7,)
        opt_freqs_hilbert = np.empty(7,)
        opt_freqs_unique = np.empty(7,)
    
        # determine starting frequencies
        # Based on starting at and interval of the defect frequency over 9000
        # with a 5000 gap between the low and hi
        for n in range(15):
            if n * defect_frequency > 9000:
                lwF_start = ((n * defect_frequency * 2) - 5000) / 2
                hiF_start = lwF_start + 5000
                break
        
        # number of data points to be taken ( = number of loop iterations for the function)
        num_points = 7
        
        opt_freqs_hilbert, opt_freqs_UTAS, opt_freqs_unique = generate_optimal_frequencies(num_points, 
                                                              lwF_start, hiF_start, defect_frequency, wave, F)
        
       # if figures dont yet exist, create them, else continue to use the same figures
       # These figures and axes are created here in order to make them global scope
        if not plt.fignum_exists(1) or not plt.fignum_exists(2) or not plt.fignum_exists(3):
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
            fig3, ax3 = plt.subplots()
        
        # Generate the plots on the pre-allocated figures and axes
        max_freq_hilbert, max_freq_UTAS, max_freq_unique = generate_plots(defect_frequency, lwF_start,
                                                            hiF_start, opt_freqs_hilbert, opt_freqs_UTAS,
                                                            opt_freqs_unique, file_name, num_points)
    
        # append the max frequency values from this BUNO to the three growing lists associated
        # with each method. This will be used later
        max_freqs_hilbert.append(max_freq_hilbert[0])
        max_freqs_UTAS.append(max_freq_UTAS[0])
        max_freqs_unique.append(max_freq_unique[0])
        
    generate_standard_curve(max_freqs_hilbert, max_freqs_UTAS, max_freqs_unique)
    