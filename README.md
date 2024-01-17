# Prelucrarea-imaginilor-cu-DFT

Vom folosi scipy, numpy și matplotlib.

from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

Imaginea cu care lucrăm

Vom folosi o imagine din setul de date oferit implicit de către scipy.

X = misc.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

Transformata Fourier a unei imagini
Transformata Fourier Discretă se extinde ușor la mai multe dimensiuni. Pentru un semnal bidimensional precum o imagine DFT devine:

Ym1,m2=∑n1=0N1−1∑n2=0N2−1xn1,n2e−j2π(m1n1/N1+m2n2/N2)
unde n1
 și n2
 sunt pozițile pixelilor pe orizontală, respectiv, pe verticală
bin-urile rezultate corespund pozițiilor pixelilor
spectrul este în continuare simetric
proprietățile transformatei DFT 1D sunt respectate și în cazul celei 2D
În continuare vom folosi rutina generală fft2 ce servește mai bine activității de învățare, deși pentru semnale reale ar trebui să folosim rfft2 ce întoarce doar informația esențială (ex. omite simetriile). De asemenea vom analiza spectrul în scală logaritmică pentru a diferenția mai bine magnitudinile bin-urilor DTF.

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))

plt.imshow(freq_db)
plt.colorbar()
plt.show()

Operațiile efectuate direct asupra imaginii se reflectă și în spectrul acesteia. Iată un exemplu a unei rotații de 45 de grade:

rotate_angle = 45
X45 = ndimage.rotate(X, rotate_angle)
plt.imshow(X45, cmap=plt.cm.gray)
plt.show()

Y45 = np.fft.fft2(X45)
plt.imshow(20*np.log10(abs(Y45)))
plt.colorbar()
plt.show()


Momentan pe axe sunt afișate numărul bin-urilor. Pentru a obține frecvențele asociate folosiți fftfreq:

freq_x = np.fft.fftfreq(X.shape[1])
freq_y = np.fft.fftfreq(X.shape[0])

plt.stem(freq_x, freq_db[:][0])
plt.show()

Atenuarea frecvențelor înalte
Pentru a anula frecvențele de peste un anumit prag freq_cutoff putem pur și simplu anula intrările din spectru și aplica transformata Fourier inversă:

freq_cutoff = 120

Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2
plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.show()

Zgomot
Zgomotul alb perturbă în mod egal spectrul semnalului. Este astfel egal distribuit și regăsit în toate bin-urile DFT. Zgomotul color se schimbă de-a lungul frecvențelor.

Putem adăuga zgomot în limita a pixel_noise pixeli imaginii folosind random.randint:

pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')
plt.show()


Sarcini:

Produceți imaginile și spectrul pentru funcțiile de mai jos și dați o explicație scurtă pentru fiecare rezultat.
xn1,n2=sin(2πn1+3πn2)
xn1,n2=sin(4πn1)+cos(6πn2)
Y0,5=Y0,N−5=1, altfel Ym1,m2=0, ∀m1,m2
Y5,0=YN−5,0=1, altfel Ym1,m2=0, ∀m1,m2
Y5,5=YN−5,N−5=1, altfel Ym1,m2=0, ∀m1,m2
Atenție: x
 reprezintă informație în domeniul timpului, Y
 în domeninul frecvenței.

 Comprimați imaginea cu ratonul de mai sus prin atenuarea frecvențelor înalte până la un prag SNR autoimpus.

 Eliminați zgomotul adăugat la imaginea cu ratonul produsă mai sus. Prezentați raportul SNR înainte și după.

 Alegeți o secvență scurtă de timp (ex. 5-10 secunde) și eliminați un instrument la alegere din semnalul audio rezultat în urma rezolvării sarcinilor de la laboratorul.
