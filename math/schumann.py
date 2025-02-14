import numpy as np
import matplotlib.pyplot as plt

class SchumannResonance:
    def __init__(self):
        self.earth_radius = 6371000  # meters
        self.speed_of_light = 299792458  # m/s
        self.fundamental_freq = 7.83  # Hz
        self.resonance_modes = [7.83, 14.3, 20.8, 27.3, 33.8]  # First 5 modes
        
    def calculate_resonant_frequencies(self, n_modes):
        """Calculate resonant frequencies for n modes
        fn = f0 * √(n(n+1)) where f0 ≈ 7.83 Hz"""
        frequencies = []
        for n in range(1, n_modes + 1):
            fn = self.fundamental_freq * np.sqrt(n * (n + 1))
            frequencies.append(fn)
        return frequencies
    
    def wave_amplitude(self, time, frequency, damping=0.1):
        """Model wave amplitude with damping"""
        return np.exp(-damping * time) * np.sin(2 * np.pi * frequency * time)
    
    def plot_resonances(self, duration=1.0, sampling_rate=1000):
        """Plot first 3 Schumann resonance modes"""
        t = np.linspace(0, duration, int(sampling_rate * duration))
        
        plt.figure(figsize=(12, 8))
        for i, freq in enumerate(self.resonance_modes[:3]):
            amplitude = self.wave_amplitude(t, freq)
            plt.plot(t, amplitude + i*2, label=f'Mode {i+1}: {freq:.1f} Hz')
            
        plt.title('First Three Schumann Resonance Modes')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (arbitrary units)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def cavity_properties(self):
        """Calculate Earth-ionosphere cavity properties"""
        wavelength = self.speed_of_light / self.fundamental_freq
        cavity_height = wavelength / 2
        return {
            'cavity_height': cavity_height / 1000,  # km
            'wavelength': wavelength / 1000,  # km
            'circumference': 2 * np.pi * self.earth_radius / 1000  # km
        }

# Example usage
sr = SchumannResonance()

# Print cavity properties
properties = sr.cavity_properties()
print("\nEarth-Ionosphere Cavity Properties:")
print(f"Approximate cavity height: {properties['cavity_height']:.2f} km")
print(f"Fundamental wavelength: {properties['wavelength']:.2f} km")
print(f"Earth circumference: {properties['circumference']:.2f} km")

# Calculate and print first 5 resonant frequencies
frequencies = sr.calculate_resonant_frequencies(5)
print("\nSchumann Resonance Modes:")
for i, freq in enumerate(frequencies):
    print(f"Mode {i+1}: {freq:.2f} Hz")

# Plot resonance modes
sr.plot_resonances()
