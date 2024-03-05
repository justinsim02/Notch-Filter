# Load Audio
sound_file = sci_io.loadmat('handel.mat')
print('Sampling rate: {}'.format(sound_file['Fs'].item()))
perfect_sound = sound_file['y'][:, 0]
IPython.display.Audio(perfect_sound, rate=8000) # we use sampling rate at 8KHz

sound_len = perfect_sound.shape[0]
# create sinusoidal noise
n = np.arange(sound_len)
noise = (np.cos(2 * np.pi * 2 / 8 * n) + np.cos(2 * np.pi * 3 / 8 * n))/16
corrputed_sound = perfect_sound + noise
IPython.display.Audio(corrputed_sound, rate=8000)

bandwidth = 4000
fs = 8000
f0 = [2000, 3000] # distortion
w0 = np.zeros(len(f0))
r = 0.99

# a) Design an IIR notch filter to eliminate f0 and f1
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

b, a = sci_sig.zpk2tf(zeros_flat, poles_flat, 1)

recovered_sound = sci_sig.lfilter(b, a, corrputed_sound)
# Now play it to see if the noise is eliminated.
IPython.display.Audio(recovered_sound, rate=8000)
