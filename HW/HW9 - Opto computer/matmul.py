"""
SIMULACIJA MNOŽENJA MATRICA PREKO OPTO-RAČUNARA KORIŠĆENJEM MZI MREŽE
=====================================================================

Ovaj primer NE upravlja stvarnim fotonskim čipom na Azure/AWS-u.
Umesto toga, on simulira kako bi Mach-Zehnder interferometarska (MZI) mreža
mogla da realizuje linearnu transformaciju, a sam cloud (Azure ili AWS)
bi ti služio samo kao okruženje za izvršavanje ovog koda.

Šta fizički predstavljamo:
- ulazni vektor x: optičke amplitude/faze po kanalima
- MZI: 2x2 programabilni optički element
- mreža MZI blokova: aproksimacija ili realizacija veće matrice W
- izlaz y = W x: rezultat optičke propagacije kroz mrežu

Kako iz toga dobiti C = A B:
- matricu B modelujemo kao optičku mrežu
- svaki red matrice A puštamo kao ulazni vektor kroz tu mrežu
- izlazni vektor postaje odgovarajući red matrice C

NAPOMENA:
Ovo je edukativna simulacija, namenjena da ti bude razumljiva i za prezentaciju.
Za realno programiranje fizičkog fotonskog hardvera potrebni su vendor-specifični alati,
kalibracija, modeli gubitaka, modeli šuma i slično.
"""

import numpy as np


# ============================================================
# 1. OSNOVNE FIZIČKE IDEJE KOJE KORISTIMO U KODU
# ============================================================
#
# U fotonici se signal često opisuje kao kompleksna amplituda:
#   E = a * exp(j * phi)
#
# gde je:
#   a    = amplituda (jačina signala)
#   phi  = faza
#
# Interferencija nastaje kada se više takvih signala sabere.
# Ako su faze "poravnate", signali se pojačavaju.
# Ako su u suprotnim fazama, mogu delimično ili potpuno da se ponište.
#
# U klasičnom linearnom računanju:
#   y = W x
#
# U optičkom ekvivalentu:
# - x je raspodela amplituda/faza na ulazu
# - W je fizička mreža (ovde: MZI mreža)
# - y je izlaz posle propagacije
#
# U simulaciji ćemo raditi sa kompleksnim matricama i vektorima.


# ============================================================
# 2. POMOĆNE FUNKCIJE
# ============================================================

def to_complex_vector(x):
    """
    Pretvara realni ili kompleksni ulaz u numpy kompleksni vektor.

    Zašto?
    U optičkim sistemima amplituda i faza prirodno žive u kompleksnom domenu.
    Zato čak i kada kreneš od realnih brojeva, zgodno je da ih interno modeluješ
    kao kompleksne veličine.
    """
    return np.asarray(x, dtype=np.complex128)


def print_matrix(name, M):
    """
    Lepši ispis matrice.
    Korisno za debug i za razumevanje rezultata.
    """
    print(f"\n{name} =")
    print(np.array_str(M, precision=4, suppress_small=True))


# ============================================================
# 3. MODEL JEDNOG MZI BLOKA
# ============================================================
#
# Mach-Zehnder interferometar:
# - deli signal na dve grane
# - uvodi fazne pomeraje
# - ponovo kombinuje grane
#
# Matematički, on se često modeluje kao 2x2 kompleksna matrica.
# U najjednostavnijem obliku, možemo ga predstaviti kao:
#
#   U_MZI(theta, phi) = exp(j*phi) * [[cos(theta),  sin(theta)],
#                                     [-sin(theta), cos(theta)]]
#
# Intuicija:
# - theta kontroliše koliko se dva ulaza "mešaju"
# - phi uvodi dodatni fazni pomeraj
#
# Ekvivalent u klasičnom računanju:
# - mali 2x2 linearni operator / rotacija + fazni faktor


def mzi_block(theta, phi=0.0):
    """
    Vraća 2x2 kompleksnu matricu koja modeluje jedan MZI blok.

    Parametri:
    - theta: kontrola mešanja kanala
    - phi: globalni ili dodatni fazni pomeraj

    Fizičko značenje:
    - theta ~ koliko svetlosti prelazi iz jednog kanala u drugi
    - phi ~ pomeraj faze koji utiče na interferenciju

    Računarski ekvivalent:
    - mali parametarski linearni operator nad 2-dimenzionim vektorom
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.exp(1j * phi) * np.array([
        [c,  s],
        [-s, c]
    ], dtype=np.complex128)


# ============================================================
# 4. UGRADNJA 2x2 MZI BLOKA U VEĆU MREŽU
# ============================================================
#
# Ako imamo više optičkih kanala, jedan MZI tipično deluje samo nad parom kanala,
# recimo kanalima i i j, dok ostali prolaze netaknuti.
#
# To je analogno tome da u velikoj matrici identiteta zameniš jedan 2x2 podblok
# odgovarajućim MZI operatorom.


def embed_2x2_in_n(U2, n, i, j):
    """
    Ugrađuje 2x2 operator U2 u nxn identitet tako da deluje na kanale i i j.

    Parametri:
    - U2: 2x2 kompleksna matrica (npr. MZI blok)
    - n: ukupna dimenzija sistema
    - i, j: indeksi kanala na koje deluje U2

    Napomena:
    - i i j moraju biti različiti
    - ostali kanali ostaju nepromenjeni
    """
    U = np.eye(n, dtype=np.complex128)

    U[i, i] = U2[0, 0]
    U[i, j] = U2[0, 1]
    U[j, i] = U2[1, 0]
    U[j, j] = U2[1, 1]

    return U


# ============================================================
# 5. JEDNOSTAVNA MZI MREŽA
# ============================================================
#
# Veća optička mreža nastaje kaskadnim povezivanjem više MZI blokova.
# Ukupna transformacija je proizvod svih tih matrica.
#
# Fizički:
# - svetlost prolazi kroz više "stepena" mešanja i faznih pomeraja
#
# Računarski:
# - samo množiš matrice operatora redom kojim signal prolazi


class MZINetwork:
    """
    Jednostavna simulacija MZI mreže.

    layers:
        lista slojeva
        svaki sloj je lista tuple-ova:
            (i, j, theta, phi)
        gde:
            i, j   = koji kanali se mešaju
            theta  = parametar mešanja
            phi    = fazni pomeraj
    """

    def __init__(self, n_channels, layers):
        self.n = n_channels
        self.layers = layers

    def transfer_matrix(self):
        """
        Računa ukupnu prenosnu matricu mreže.

        Ako svetlost prolazi kroz slojeve L1, L2, L3...
        ukupna transformacija je njihov matrični proizvod.
        """
        U_total = np.eye(self.n, dtype=np.complex128)

        for layer_idx, layer in enumerate(self.layers):
            U_layer = np.eye(self.n, dtype=np.complex128)

            # Svaki MZI u sloju deluje na određeni par kanala
            for (i, j, theta, phi) in layer:
                U2 = mzi_block(theta, phi)
                U_emb = embed_2x2_in_n(U2, self.n, i, j)

                # MZI blokove u sloju množimo u redosledu definicije
                U_layer = U_emb @ U_layer

            # Ceo sloj se primenjuje na dotad akumuliranu transformaciju
            U_total = U_layer @ U_total

        return U_total

    def propagate(self, x):
        """
        Propagacija ulaznog vektora x kroz mrežu.

        Fizičko značenje:
        - x predstavlja ulazne optičke amplitude/faze
        - izlaz je rezultat prolaska kroz mrežu

        Računarski ekvivalent:
        - y = U x
        """
        x = to_complex_vector(x)
        U = self.transfer_matrix()
        return U @ x


# ============================================================
# 6. KAKO MREŽA PREDSTAVLJA MATRICU ZA MNOŽENJE
# ============================================================
#
# U idealizovanom fotonskom računaru želiš da mreža implementira matricu W.
# Za pravu fizičku realizaciju proizvoljne matrice potrebne su:
# - dekompozicije (npr. u unitarni deo + diagonalna skaliranja),
# - model gubitaka,
# - često dodatni aktivni elementi.
#
# U ovoj edukativnoj verziji radićemo dve stvari:
# 1) pokazaćemo realnu MZI mrežu i njenu matricu prenosa
# 2) za samo množenje C = A B koristićemo "target operator" B kao željenu mrežu
#    da bi ideja bila jasna i praktična za učenje
#
# Drugim rečima:
# - MZI mreža ti pokazuje kako optički blokovi daju linearnu transformaciju
# - funkcija optical_matrix_multiply pokazuje kako se redovi A puštaju kroz mrežu B


def optical_matrix_vector_multiply(W, x):
    """
    Idealizovano optičko matrično-vektorsko množenje.

    Ovde W predstavlja matricu koju je optička mreža realizovala.
    U fizičkom smislu:
    - W je prenosna funkcija mreže
    - x je ulazni optički signal
    - y je izlazni optički signal

    Računarski:
    - klasično y = W x
    """
    W = np.asarray(W, dtype=np.complex128)
    x = to_complex_vector(x)
    return W @ x


def optical_matrix_multiply(A, B):
    """
    Matrično množenje C = A B preko optičke ideje:

    - matricu B posmatramo kao mrežu / linearni operator
    - svaki red matrice A puštamo kao poseban ulaz
    - dobijeni izlaz postaje odgovarajući red rezultata C

    Ovo je direktna implementacija ideje sa tvojih slajdova.
    """
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)

    if A.shape[1] != B.shape[0]:
        raise ValueError("Dimenzije nisu kompatibilne za množenje: A.columns mora biti B.rows")

    rows_A = A.shape[0]
    cols_B = B.shape[1]

    C = np.zeros((rows_A, cols_B), dtype=np.complex128)

    for r in range(rows_A):
        # Uzimamo r-ti red matrice A
        a_row = A[r, :]

        # Da bismo ga pustili kroz operator B, gledamo transponovanu formu:
        # red * B  <=>  (B^T * red^T)^T
        #
        # Ovo je samo tehnička stvar zbog standardnog oblika vektora kao kolone.
        y_col = optical_matrix_vector_multiply(B.T, a_row)

        # Izlaz prepisujemo kao r-ti red rezultata
        C[r, :] = y_col

    return C


# ============================================================
# 7. DETEKCIJA: AMPLITUDA VS INTENZITET
# ============================================================
#
# U realnim optičkim sistemima često ne meriš direktno kompleksnu amplitudu,
# nego intenzitet:
#
#   I = |E|^2
#
# gde je E kompleksna amplituda.
#
# To znači:
# - amplituda/faza = "pun optički signal"
# - intenzitet     = ono što fotodetektor fizički meri
#
# Ako želiš analog klasičnog "čitljivog" izlaza, intenzitet je prirodan kandidat.


def photodetect(field):
    """
    Pretvara kompleksni optički signal u intenzitet.

    Fizička pojava:
    - fotodetekcija meri snagu/intenzitet svetlosti

    Računarski ekvivalent:
    - uzimanje kvadrata modula kompleksne vrednosti
    """
    field = np.asarray(field, dtype=np.complex128)
    return np.abs(field) ** 2


# ============================================================
# 8. DEMO: PRAVA MZI MREŽA + NJENO DEJSTVO
# ============================================================

def build_demo_mzi_network():
    """
    Pravimo jednu malu 4-kanalnu MZI mrežu sa dva sloja.

    Ovo nije optimizovana mreža za realizaciju proizvoljne zadate matrice,
    već pregledan primer koji pokazuje:
    - kako se mreža sklapa
    - kako izgleda njena ukupna matrica prenosa
    """
    layers = [
        # Sloj 1: mešamo kanale (0,1) i (2,3)
        [
            (0, 1, np.pi / 6, 0.10),
            (2, 3, np.pi / 4, -0.15),
        ],
        # Sloj 2: mešamo kanale (1,2)
        [
            (1, 2, np.pi / 5, 0.20),
        ]
    ]
    return MZINetwork(n_channels=4, layers=layers)


# ============================================================
# 9. POKRETANJE PRIMERA
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # A) PRIKAZ JEDNE DEMO MZI MREŽE
    # --------------------------------------------------------
    net = build_demo_mzi_network()
    U_demo = net.transfer_matrix()

    print_matrix("Ukupna transfer matrica demo MZI mreže", U_demo)

    # Ulazni optički vektor: realne vrednosti, ali interno ga tretiramo kao kompleksan
    x = np.array([1.0, 0.5, -0.2, 0.8], dtype=np.float64)
    y_field = net.propagate(x)
    y_intensity = photodetect(y_field)

    print_matrix("Ulazni vektor x", x)
    print_matrix("Izlazni optički signal y_field", y_field)
    print_matrix("Izlazni intenzitet y_intensity", y_intensity)

    # --------------------------------------------------------
    # B) OPTIČKO MNOŽENJE MATRICA C = A B
    # --------------------------------------------------------
    #
    # Ovde B predstavlja "mrežu", a redovi A se puštaju kroz nju.
    #
    A = np.array([
        [1, 2, 3],
        [0, 1, 4]
    ], dtype=np.float64)

    B = np.array([
        [2, 0],
        [1, 3],
        [4, 5]
    ], dtype=np.float64)

    C_optical = optical_matrix_multiply(A, B)
    C_classical = A @ B

    print_matrix("Matrica A", A)
    print_matrix("Matrica B", B)
    print_matrix("Optički dobijeno C = A B", C_optical)
    print_matrix("Klasično C = A @ B", C_classical)

    # Provera tačnosti idealizovanog modela
    print("\nDa li se poklapa sa klasičnim množenjem?")
    print(np.allclose(C_optical.real, C_classical, atol=1e-10))