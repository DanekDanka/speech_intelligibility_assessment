# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, fs):
    # Function to read rttm files and generate VAD's markup in samples
    
    vad_markup = np.zeros(len(signal)).astype('float32')
        
    ###########################################################
    # Here is your code
    with open(path_to_rttm, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        fields = line.split()
        if len(fields) < 5:
            continue
            
        object_type = fields[0]
        if object_type != 'SPEAKER':
            continue
                
        start_time = float(fields[3])
        duration = float(fields[4])
        end_time = start_time + duration
        
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        start_sample = max(0, min(start_sample, len(signal)))
        end_sample = max(0, min(end_sample, len(signal)))
        
        if start_sample < end_sample:
            vad_markup[start_sample:end_sample] = 1.0
    
    ###########################################################
    
    return vad_markup

def framing(signal, window=320, shift=160):
    # Function to create frames from signal
    
    shape   = (int((signal.shape[0] - window)/shift + 1), window)
    frames  = np.zeros(shape).astype('float32')

    ###########################################################
    # Here is your code
    
    num_frames = int((len(signal) - window) / shift + 1)
    for i in range(num_frames):
        start_idx = i * shift
        end_idx = start_idx + window
        frames[i] = signal[start_idx:end_idx]
    
    ###########################################################
    
    return frames

def frame_energy(frames):
    # Function to compute frame energies
    
    E = np.zeros(frames.shape[0]).astype('float32')

    ###########################################################
    # Here is your code
    for i in range(frames.shape[0]):
        E[i] = np.sum(frames[i])
    ###########################################################
    
    return E

def norm_energy(E):
    # Function to normalize energy by mean energy and energy standard deviation
    
    E_norm = np.zeros(len(E)).astype('float32')

    ###########################################################
    # Here is your code
    E_mean = np.mean(E)
    E_std = np.std(E)
    E_norm = (E - E_mean) / E_std
    ###########################################################
    
    return E_norm

def gmm_train(E, gauss_pdf, n_realignment):
    # Function to train parameters of gaussian mixture model
    
    # Initialization gaussian mixture models
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])

    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):
        

        # E-step
        ###########################################################
        # Here is your code
        for j in range(len(w)):
            g[:, j] = w[j] * gauss_pdf(E, m[j], sigma[j])
        
        g_sum = np.sum(g, axis=1, keepdims=True)
        g = g / (g_sum + 1e-10)

        ###########################################################

        # M-step
        ###########################################################
        # Here is your code
        N_k = np.sum(g, axis=0)
        w = N_k / len(E)
        
        for j in range(len(w)):
            m[j] = np.sum(g[:, j] * E) / (N_k[j] + 1e-10)
        
        for j in range(len(w)):
            sigma[j] = np.sqrt(np.sum(g[:, j] * (E - m[j])**2) / (N_k[j] + 1e-10))

        ###########################################################
        
    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    # Function to estimate a posterior probability that frame isn't speech

    g0 = np.zeros(len(E))

    ###########################################################
    # Here is your code
    denominator = np.zeros(len(E))
    for j in range(len(w)):
        denominator += w[j] * gauss_pdf(E, m[j], sigma[j])
    
    numerator = w[0] * gauss_pdf(E, m[0], sigma[0])
    g0 = numerator / (denominator + 1e-10)

    ###########################################################
            
    return g0

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    
    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup

    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    # Function to create reverberation effect
    
    signal_reverb = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    signal_reverb = np.convolve(signal, impulse_response, mode='same')
    signal_reverb = signal_reverb / np.max(np.abs(signal_reverb)) if np.max(np.abs(signal_reverb)) > 0 else signal_reverb
    
    ###########################################################
    
    return signal_reverb

def awgn(signal, sigma_noise):
    # Function to add white gaussian noise to signal
    
    signal_noise = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    noise = np.random.normal(0, sigma_noise, len(signal))
    signal_noise = signal + noise
    signal_noise = signal_noise / np.max(np.abs(signal_noise)) if np.max(np.abs(signal_noise)) > 0 else signal_noise
    
    ###########################################################
    
    return signal_noise