# Q2-tech
Code-:
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import noisereduce as nr
import os
from scipy.signal import wiener
import pywt
import librosa

# ==============================================
# Configuration (ADJUST THESE VALUES)
# ==============================================
INPUT_AUDIO = "signal_modulated_noisy_audio.wav"
OUTPUT_AUDIO = "clean_audio6.wav"
LOW_CUTOFF = 9000
HIGH_CUTOFF = 10500
FILTER_ORDER = 8
NOISE_REDUCTION_STRENGTH = 0.85  # 0.5 (mild) to 1 (aggressive)
PLOT_DIR = "plots_main"

# ==============================================
# Core Functions
# ==============================================

def load_audio(file_path):
    """Load audio file and convert to mono."""
    try:
        sample_rate, data = wavfile.read(file_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return sample_rate, data.astype(np.float32)  # Ensure float32 for noisereduce
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        exit()

def design_bandpass_filter(sample_rate, lowcut, highcut, order):
    """Butterworth bandpass filter."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def adaptive_wiener(data):
    """Apply Wiener filter to the bandpassed signal."""
    return wiener(data, mysize=11)  # Larger window for high-frequency noise

def estimate_carrier_frequency(signal, sample_rate):
    """Estimate the dominant frequency (AM carrier) using FFT."""
    fft_spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)
    positive_freqs = freqs[:len(freqs)//2]
    magnitude = np.abs(fft_spectrum[:len(freqs)//2])
    
    dominant_freq_index = np.argmax(magnitude)
    fc = positive_freqs[dominant_freq_index]
    print(f"Estimated Carrier Frequency (Fc): {fc:.2f} Hz")
    return fc

def am_demodulate(signal, fc, sample_rate):
    """Demodulate an AM signal using synchronous detection."""
    t = np.arange(len(signal)) / sample_rate
    carrier = np.cos(2 * np.pi * fc * t)
    demodulated = signal * carrier  # Mix down
    # Low-pass filter to recover baseband
    b, a = butter(5, 0.1)  # Adjust cutoff if needed
    baseband = filtfilt(b, a, demodulated)
    return baseband

def plot_signal(signal, sample_rate, title, filename):
    plt.figure(figsize=(12, 3))
    t = np.arange(len(signal)) / sample_rate
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def harmonic_enhancement(data, sample_rate):
    """Boost harmonics within 9000-10500 Hz."""
    nyquist = 0.5 * sample_rate
    enhanced_data = np.zeros_like(data)
    
    # Example: Boost harmonics at 9500 Hz, 10000 Hz, 10500 Hz
    harmonics = [9500 , 10500]  # Adjust based on your spectrogram
    bandwidth = 50  # Hz around each harmonic to boost
    
    for freq in harmonics:
        low = (freq - bandwidth) / nyquist
        high = (freq + bandwidth) / nyquist
        b, a = butter(2, [low, high], btype='bandpass')
        enhanced_data += filtfilt(b, a, data)
    
    return enhanced_data

def selective_spectral_gate(data, sample_rate, time_ranges=[(0.4, 1.0)], freq_range=(9000, 10500), reduction_db=20):
    """
    Reduce energy in a frequency band for specific time regions using STFT masking.
    """
    n_fft = 2048
    hop_length = n_fft // 4
    stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sample_rate, hop_length=hop_length)

    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    for (start_t, end_t) in time_ranges:
        time_mask = (times >= start_t) & (times <= end_t)
        # Apply attenuation (reduce by `reduction_db`)
        attenuation = 10 ** (-reduction_db / 20)
        magnitude[np.ix_(freq_mask, time_mask)] *= attenuation

    # Reconstruct signal
    modified_stft = magnitude * np.exp(1j * phase)
    y_out = librosa.istft(modified_stft, hop_length=hop_length, length=len(data))
    return y_out

def spectral_subtraction(data, sample_rate):
    """Subtract noise within 9000-10500 Hz."""
    n_fft = 4096  # Higher resolution for precise frequency editing
    D = librosa.stft(data.astype(np.float32), n_fft=n_fft)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Frequency bins corresponding to 9000-10500 Hz
    freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    target_mask = (freq_bins >= 9000) & (freq_bins <= 10500)
    
    # Estimate noise from non-target frequencies (optional)
    noise_profile = np.mean(magnitude[~target_mask, :], axis=0, keepdims=True)
    
    # Subtract noise within the target band
    magnitude[target_mask, :] -= 1.2 * noise_profile  # Adjust multiplier (1.2 to 2.0)
    magnitude = np.maximum(magnitude, 0)  # Avoid negative values
    
    # Reconstruct audio
    D_clean = magnitude * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean)
    return y_clean

def reduce_noise(data, sample_rate, strength=0.8):
    """Spectral noise reduction using noisereduce."""
    # Estimate noise from the first 0.5 seconds
    noise_start = 0
    noise_end = int(0.5 * sample_rate)
    noise_clip = data[noise_start:noise_end]
    
    # Apply noise reduction (remove 'verbose' argument)
    reduced_data = nr.reduce_noise(
        y=data,
        sr=sample_rate,
        y_noise=noise_clip,
        prop_decrease=strength,
        n_fft=1024
    )
    return reduced_data

def plot_spectrogram(data, sample_rate, title, filename):
    """Plot spectrogram to visualize frequency content."""
    f, t, Sxx = spectrogram(data, fs=sample_rate, nperseg=1024)
    plt.figure(figsize=(12, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.ylim(0, sample_rate//2)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

# ==============================================
# Main Processing
# ==============================================

def main():
    # Setup directories
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load audio
    sample_rate, data = load_audio(INPUT_AUDIO)

    # Plot original spectrogram
    plot_spectrogram(data, sample_rate, "Original Spectrogram", "original_spectrogram.png")

    data_denoised = reduce_noise(data, sample_rate, strength=NOISE_REDUCTION_STRENGTH)

    # Step 2: Bandpass filtering
    b, a = design_bandpass_filter(sample_rate, LOW_CUTOFF, HIGH_CUTOFF, FILTER_ORDER)
    filtered_data = filtfilt(b, a, data_denoised)

    # Step 3: Harmonic enhancement
    filtered_data = harmonic_enhancement(filtered_data, sample_rate)

    # Step 4: Spectral subtraction
    filtered_data = selective_spectral_gate(
    filtered_data,
    sample_rate,
    time_ranges=[
        (0.40, 0.50),
        (0.55, 0.65),
        (0.70, 0.80),
        (0.85, 0.95),
        (1.10, 1.25),
        (1.30, 1.85)
    ],
    freq_range=(9000, 10500),
    reduction_db=28  # Tune this value for stronger or milder suppression
    )

    # Step 6: Post-processing normalization
    filtered_data = np.int16(filtered_data / np.max(np.abs(filtered_data)) * 32767)

    # Save cleaned audio
    wavfile.write(OUTPUT_AUDIO, sample_rate, filtered_data)

    # Plot results
    plot_spectrogram(filtered_data, sample_rate, "Filtered Spectrogram", "filtered_spectrogram.png")
        # FFT to estimate carrier frequency
    fc = estimate_carrier_frequency(filtered_data, sample_rate)

    # Plot signal before demodulation
    plot_signal(filtered_data, sample_rate, "Filtered (Modulated) Signal", "filtered_signal.png")

    # Demodulate
    demodulated_audio = am_demodulate(filtered_data, fc, sample_rate)

    # Normalize demodulated audio
    demodulated_audio = demodulated_audio / np.max(np.abs(demodulated_audio))
    demodulated_int = np.int16(demodulated_audio * 32767)

    # Save final demodulated audio
    wavfile.write("demodulated_audio.wav", sample_rate, demodulated_int)

    # Plot demodulated signal
    plot_signal(demodulated_audio, sample_rate, "Demodulated Audio Signal", "demodulated_signal.png")

    # Final spectrogram
    plot_spectrogram(demodulated_audio, sample_rate, "Final Demodulated Spectrogram", "demodulated_spectrogram.png")
    
    print(f"""
    ========================================
    Processing Complete!
    ========================================
    - Noise reduction strength: {NOISE_REDUCTION_STRENGTH}
    - Filter range: {LOW_CUTOFF}-{HIGH_CUTOFF} Hz
    - Output: {OUTPUT_AUDIO}
    ========================================
    """)

if __name__ == "__main__":
    main()

~~code ends~
I am also pasting my audio files and images on google drive as i am unable to do it here 
Plots and images-:
![demodulated_spectrogram](https://github.com/user-attachments/assets/d734c210-1b86-4d94-a04f-c4a8bbccd616)
![demodulated_signal](https://github.com/user-attachments/assets/2627a602-9ce8-436f-88e8-6dbad84e09dd)
![filtered_signal](https://github.com/user-attachments/assets/8949d28f-f7a0-4314-9f14-97fde9170fb5)
![filtered_spectrogram](https://github.com/user-attachments/assets/56b399f6-b4b7-4c12-a12f-204c2d530ff7)
![original_spectrogram](https://github.com/user-attachments/assets/c28142f2-db28-4f76-b747-902851838824)
explaination of code -:
Butterworth Bandpass Filter
Purpose: Isolate frequencies between LOW_CUTOFF (9,000 Hz) and HIGH_CUTOFF (10,500 Hz) to focus on the AM signal's carrier and sidebands.

Implementation:

Designed using scipy.signal.butter with a specified FILTER_ORDER (8th order for steep roll-off).

Applied with filtfilt for zero-phase filtering to avoid distortion.

Key Parameters:

lowcut: Lower cutoff frequency.

highcut: Upper cutoff frequency.

order: Filter order (higher = sharper cutoff).

Adaptive Wiener Filter
Purpose: Reduce high-frequency noise while preserving sharp signal features.

Implementation: scipy.signal.wiener with a window size of 11 samples.

Mechanism: Estimates local noise variance and attenuates regions where noise dominates.

Harmonic Enhancement
Purpose: Boost specific harmonics (e.g., 9,500 Hz and 10,500 Hz) to strengthen the AM signal.

Implementation:

Applies multiple Butterworth bandpass filters around target frequencies.

Sums the filtered signals to enhance harmonics in the desired band.

Selective Spectral Gating
Purpose: Attenuate noise in predefined time-frequency regions (e.g., transient interference).

Implementation:

Uses STFT (Short-Time Fourier Transform) to create a time-frequency representation.

Applies attenuation (reduction_db) to specific time_ranges and freq_range.

Example: Reduces noise between 0.4–1.0 seconds in the 9–10.5 kHz band.

Spectral Subtraction
Purpose: Subtract noise estimates from the target frequency band.

Implementation:

Computes STFT of the signal.

Estimates noise from non-target frequencies and subtracts it from the target band.

Reconstructs the signal using the modified STFT.

Noise Reduction (noisereduce)
Purpose: Remove broadband noise using a spectral noise profile.

Implementation:

Estimates noise from the first 0.5 seconds of audio.

Reduces noise across the entire signal using spectral masking (prop_decrease controls strength).

2. FFT Usage
Carrier Frequency Estimation
Function: estimate_carrier_frequency

Purpose: Identify the AM carrier frequency.

Steps:

Compute FFT of the filtered signal.

Find the frequency bin with the highest magnitude (dominant peak).

Convert bin index to frequency (Hz).

STFT-Based Processing
Used in: selective_spectral_gate, spectral_subtraction, noisereduce, and spectrogram plotting.

Purpose: Analyze and manipulate time-frequency content.

Mechanism:

Divide the signal into overlapping windows.

Compute FFT for each window to get magnitude/phase.

Modify magnitudes (e.g., noise reduction) and reconstruct with inverse STFT.

Spectrogram Visualization
Function: plot_spectrogram

Purpose: Visualize frequency content over time.

Tools: scipy.signal.spectrogram (uses FFT internally).

3. Key Functions
AM Demodulation
Function: am_demodulate

Purpose: Extract the baseband (original) signal from the AM carrier.

Steps:

Multiply the signal by a cosine wave at the carrier frequency (fc).

Apply a low-pass filter to remove high-frequency components (carrier residue).

Normalization
Purpose: Ensure the final audio is within the 16-bit integer range (-32768 to 32767).

Implementation: Scale the signal to maximum amplitude before converting to int16.

Plotting Utilities
Functions: plot_signal, plot_spectrogram

Purpose: Visualize signals at different stages (time domain, spectrograms).

4. Workflow Summary
Noise Reduction: Broadband noise removal with noisereduce.

Bandpass Filtering: Isolate the 9–10.5 kHz band.

Harmonic Enhancement: Boost key frequencies.

Spectral Gating/Subtraction: Remove interference in specific regions.

Carrier Estimation: Use FFT to find the dominant frequency.

Demodulation: Recover the baseband signal.

Normalization: Prepare for WAV file output.

5. Parameter Tuning
Critical Parameters:

LOW_CUTOFF/HIGH_CUTOFF: Define the AM signal's bandwidth.

NOISE_REDUCTION_STRENGTH: Balances noise removal vs. signal distortion.

reduction_db (spectral gating): Controls how much energy is removed.

Adjustments: Tune based on spectrogram observations (e.g., interference locations).

This pipeline effectively combines filtering, spectral analysis, and demodulation to recover a clean audio signal from a noisy AM transmission.






