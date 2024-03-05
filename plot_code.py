import numpy as np
import scipy.signal as signal
import scipy
import matplotlib.pyplot as plt
import math
bandwidth = 4000
fs = 8000
f0 = [1500, 2500] # distortion
w0 = np.zeros(len(f0))
r = 0.99

# a and b) Design an IIR notch filter to eliminate f0 and f1
poles = np.zeros((2,len(f0)), dtype=complex)
zeros = np.zeros((2,len(f0)), dtype=complex)
for f in range(len(f0)):
  temp_w = 2*np.pi*f0[f]/fs
  w0[f] = temp_w
  pole_temp = r * np.exp(1j*temp_w)
  poles[f,0] = pole_temp
  poles[f,1] = pole_temp.conjugate()

  zero_temp = np.exp(1j*temp_w)
  zeros[f,0] = zero_temp
  zeros[f,1] = zero_temp.conjugate()
zeros_flat = zeros.flatten()
poles_flat = poles.flatten()

b, a = signal.zpk2tf(zeros_flat, poles_flat, 1)

print("b coefficients: ", b)
print("a coefficients: ", a)
# Compute the frequency response
# Normalize the filter coefficients to ensure maximum gain is 1
b_normalized = b / max(abs(np.roots(b)))
a_normalized = a / max(abs(np.roots(a)))

# Compute the frequency response of the normalized filter
w, h = signal.freqz(b_normalized, a_normalized, fs=fs)

# Plot the log magnitude response of the filter
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Log Magnitude Response of Notch Filter f0 = 1500, 2500')
plt.xlabel('Frequency [radians/sample]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()
