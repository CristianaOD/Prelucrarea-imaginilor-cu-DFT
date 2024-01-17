from scipy import misc
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, restoration, data

"""
Exercitiul 1
"""
def generare_imag_spectrul(func, size, step, frecventa_timp=False):
    N1, N2 = np.mgrid[0:size:step, 0:size:step]
    data = func(N1, N2)

    if frecventa_timp:
        # Daca functia este in domeniul frecvenței, aplic transformata Fourier inversa
        imagine = np.fft.ifft2(data).real
        spectru = data
    else:
        # Daca functia se afla in domeniul timpului, aplic transformata Fourier
        imagine = data
        spectru = np.fft.fft2(data)

    # Un singur cadran pentru a evita simetriile
    cadran = (slice(0, spectru.shape[0] // 2), slice(0, spectru.shape[1] // 2))
    spectru_cadran = spectru[cadran]

    # Scala pentru vizibilitate (scara logaritmica pentru spectru)
    spectru_scala = np.log(np.abs(spectru_cadran) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].set_title("Imagine")
    axs[0].imshow(imagine, cmap='gray')

    axs[1].set_title("Spectrul")
    axs[1].imshow(spectru_scala, cmap='gray')

    plt.show()

# xn1,n2=sin(2πn1+3πn2)
def x1(n1, n2):
    return np.sin(2 * np.pi * n1 + 3 * np.pi * n2)

generare_imag_spectrul(x1, 2, 1/20)

"""
Functia reprezintă un semnal sinusoidal bidimensional. Acesta variaza în funcție 
de coordonatele n1 si n2. Dimensiunea imaginii de 2×2 cu un pas de eșantionare de 1/20.

Imaginea:
Imaginea va prezenta două cicluri sinusoidale în direcțiile n1 si n2.
Culorile în imagine reprezintă amplitudinea semnalului în punctele respective.

Spectrul:
Spectrul va arata doua impulsuri situate la frecventele asociate celor două componente 
sinusoidale ale funcției.
Valorile în spectru sunt reprezentate într-o scală logaritmică pentru a evidenția mai 
bine magnitudinile.
"""

# xn1,n2=sin(4πn1)+cos(6πn2)
def x2(n1, n2):
    return np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

generare_imag_spectrul(x2, 5, 1/5)

"""
Functia combina un termen sinusoidal in functie de n1 cu un termen cosinusoidal in functie de n2. 
Dimensiunea imaginii de 5×5 cu un pas de eșantionare de 1/5, iar rezultatul va fi afisat în doua.

Imaginea:
Imaginea combina influența ambelor componente sinusoidale.
Culorile in imagine reprezinta amplitudinea semnalului in punctele respective.

Spectrul:
Spectrul va arata două impulsuri situate la frecventele asociate celor două componente sinusoidale 
ale functiei.
Valorile in spectru sunt reprezentate intr-o scala logaritmica pentru a evidentia mai bine magnitudinile.
Functia este in general o combinatie a frecventelor fundamentale ale celor doua componente.
"""

# Y0,5=Y0,N−5=1, altfel Ym1,m2=0, ∀m1,m2
def y3(n1, n2):
    conditie = (n1 == 0) & ((n2 == 5) | (n2 == n1.shape[1] - 5))
    return np.where(conditie, 1, 0)

generare_imag_spectrul(y3, 20, 1, frecventa_timp=True)

"""
Dimensiunea imaginii de 20×20 cu un pas de eșantionare de 1, iar rezultatul va fi 
afișat într-o fereastră împărțită în două.

Imaginea:
Imaginea va prezenta un model de impuls cu doua varfuri situate la pozițiile (0, 5) 
si (0, N-5).
Culorile in imagine reprezinta amplitudinea impulsurilor.

Spectrul:
Spectrul va arata doua impulsuri situate la frecvențele asociate pozițiilor (0, 5) 
si (0, N-5).
Celelalte componente ale spectrului vor fi zero, conform condițiilor specificate.
Această funcție in domeniul frecventei este conceputa astfel incat sa evidentieze 
doar doua frecvente specifice (5 și N-5), iar rezultatele vor ilustra caracteristica 
distinctiva.
"""

# Y5,0=YN−5,0=1, altfel Ym1,m2=0, ∀m1,m2
def y4(m1, m2):
    conditie = ((m1 == 5) | (m1 == m1.shape[0] - 5)) & (m2 == 0)
    return np.where(conditie, 1, 0)

generare_imag_spectrul(y4, 20, 1, frecventa_timp=True)

"""
Dimensiunea imaginii de 20×20 cu un pas de eșantionare de 1, iar rezultatul va fi afisat 
intr-o fereastra imparțita in doua.

Imaginea:
Imaginea va prezenta un model de impuls cu doua varfuri situate la pozitiile (5, 0) si (N-5, 0).
Culorile in imagine reprezinta amplitudinea impulsurilor.

Spectrul:
Spectrul va arata doua impulsuri situate la frecventele asociate pozitiilor (5, 0) și (N-5, 0).
Celelalte componente ale spectrului vor fi zero, conform condițiilor specificate in functie.
"""

# Y5,5=YN−5,N−5=1, altfel Ym1,m2=0, ∀m1,m2
def y5(m1, m2):
    conditie = ((m1 == 5) & (m2 == 5)) | ((m1 == (m1.shape[0] - 5)) & (m2 == (m2.shape[1] - 5)))
    return np.where(conditie, 1, 0)

generare_imag_spectrul(y5, 20, 1, frecventa_timp=True)

"""
Vom analiza rezultatele pentru dimensiunea imaginii de 20×20 cu un pas de eșantionare de 
1, iar rezultatul va fi afisat in doua: o parte pentru imagine si alta pentru spectru.

Imaginea:
Imaginea va prezenta un model de impuls cu două varfuri situate la pozițiile (5, 5) și (N-5, N-5).
Culorile în imagine reprezintă amplitudinea impulsurilor.

Spectrul:
Spectrul va arata doua impulsuri situate la frecventele asociate pozițiilor (5, 5) si (N-5, N-5).
Celelalte componente ale spectrului vor fi zero, conform condițiilor specificate in functie.
"""

"""
Exercitiul 2
"""
X = misc.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()
Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))

freq_x = np.fft.fftfreq(X.shape[1])
freq_y = np.fft.fftfreq(X.shape[0])

freq_cutoff = 105

Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2
plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.show()

signal_amplitude_initial = np.max(freq_db)
noise_amplitude_initial = np.mean(freq_db[freq_db <= 0])

"""
Exercitiul 3
"""
X = misc.face(gray=True)

pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

X_denoised = restoration.denoise_bilateral(X_noisy)

snr_before = 20 * np.log10(np.max(X) / np.std(noise))
snr_after = 20 * np.log10(np.max(X) / np.std(X - X_denoised))


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(X, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Cu noise")
plt.imshow(X_noisy, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Dupa denoise")
plt.imshow(X_denoised, cmap='gray')
plt.axis('off')

plt.show()

print(f"SNR inainte de denoise: {snr_before:.2f} dB")
print(f"SNR dupa denoise: {snr_after:.2f} dB")

"""
Exercitiul 4
"""
# Incarcati fisierul audio
audio_file = "Lab5Trimmed.wav"
audio = AudioSegment.from_file(audio_file)

# Definiti frecventa de taiere pentru bass (de exemplu, 100 Hz)
cutoff_frequency = 100

# Filtrati frecventele sub cutoff_frequency
filtered_audio = audio.low_pass_filter(cutoff_frequency)

# Salvati rezultatul intr-un nou fisier audio
output_file = "Lab5Trimmed_NoBass.wav"
filtered_audio.export(output_file, format="wav")

# Redare fisierul audio filtrat
play(filtered_audio)









