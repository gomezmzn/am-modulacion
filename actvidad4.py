import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def generar_senales(fs=50_000.0, duracion=0.02, fm=250.0, fc=5_000.0, am=1.0, ac=1.0):
    """Genera el vector de tiempo, señal mensaje m(t) de 250 y 500 Hz y portadora c(t)."""
    t = np.arange(0, duracion, 1.0 / fs)
    mensaje = am * (0.8 * np.sin(2 * np.pi * fm * t) + 0.2 * np.sin(2 * np.pi * (2 * fm) * t))
    portadora = ac * np.cos(2 * np.pi * fc * t)
    return t, mensaje, portadora


def modular_am(t, mensaje, fc=5_000.0, ac=1.0, mu=0.8):
    """
    Implementa AM convencional con portadora transmitida (DSB-LC):
    normaliza m(t) al rango [-1, 1], construye (1 + mu * m_norm) y multiplica por c(t).
    """
    pico_m = np.max(np.abs(mensaje))
    m_norm = mensaje / pico_m if pico_m > 0 else mensaje
    termino_modulador = 1.0 + mu * m_norm
    senial_am = ac * termino_modulador * np.cos(2 * np.pi * fc * t)
    return senial_am, m_norm


def calcular_espectro(senial, fs, referencia=1.0):
    """
    Espectro unilateral de amplitud con ventana de Hanning y referencia común.
    Salida en dB respecto a `referencia` (dBV si referencia=1.0 V).
    """
    n = len(senial)
    ventana = np.hanning(n)
    transformada = np.fft.rfft(senial * ventana)
    amplitud = np.abs(transformada) / np.sum(ventana)
    if n % 2 == 0:
        amplitud[1:-1] *= 2.0
    else:
        amplitud[1:] *= 2.0
    frecuencias = np.fft.rfftfreq(n, d=1.0 / fs)
    espectro_db = 20.0 * np.log10(np.maximum(amplitud / referencia, 1e-12))
    return frecuencias, espectro_db


def simular_canal(senial_am, atenuacion_db=0.0, snr_db=None, distorsion_clip=None, seed=42):
    """
    Simula efectos de canal de forma aislada o combinada:
    - atenuacion_db: pérdida en dB aplicada mediante 10^(-atenuacion_db/20).
    - snr_db: si es None, no añade ruido; si se indica, añade AWGN para esa SNR objetivo.
    - distorsion_clip: fracción del máximo absoluto donde se recortan picos (ej. 0.65).
    """
    senial = senial_am * (10.0 ** (-atenuacion_db / 20.0))

    if distorsion_clip is not None:
        umbral = distorsion_clip * np.max(np.abs(senial))
        senial = np.clip(senial, -umbral, umbral)

    if snr_db is not None:
        rng = np.random.default_rng(seed)
        potencia_senial = np.mean(senial ** 2)
        snr_lineal = 10.0 ** (snr_db / 10.0)
        potencia_ruido = potencia_senial / snr_lineal
        ruido = rng.normal(0.0, np.sqrt(potencia_ruido), size=len(senial))
        senial = senial + ruido

    return senial


def generar_figuras(t, mensaje, portadora, senial_am, senial_aten, senial_ruido, senial_dist, fs):
    """Genera las 6 figuras solicitadas y las guarda en figs/."""
    os.makedirs("figs", exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8
    t_ms = t * 1000.0

    # Figura 1: Mensaje y portadora
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.0))
    axes[0].plot(t_ms, mensaje, color="#1D4ED8", lw=1.5)
    axes[0].set_title("Señal de entrada m(t) (250 Hz + 500 Hz)", fontsize=10, fontweight="bold", color="#0F172A")
    axes[0].set_xlabel("Tiempo (ms)", fontsize=8.5)
    axes[0].set_ylabel("Amplitud (V)", fontsize=8.5)
    axes[0].grid(True, alpha=0.3, ls="--")

    axes[1].plot(t_ms[:250], portadora[:250], color="#475569", lw=1.1)
    axes[1].set_title("Portadora c(t) de 5 kHz (primeros 5 ms)", fontsize=10, fontweight="bold", color="#0F172A")
    axes[1].set_xlabel("Tiempo (ms)", fontsize=8.5)
    axes[1].set_ylabel("Amplitud (V)", fontsize=8.5)
    axes[1].grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig("figs/01_mensaje_portadora.png", dpi=200)
    plt.close()

    # Figura 2: Señal AM y envolvente con Hilbert
    envolvente = np.abs(hilbert(senial_am))
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.plot(t_ms, senial_am, color="#0D9488", lw=0.85, label="Señal AM s(t)")
    ax.plot(t_ms, envolvente, color="#DC2626", lw=1.5, ls="--", label="Envolvente |hilbert(s(t))|")
    ax.plot(t_ms, -envolvente, color="#DC2626", lw=1.2, ls=":")
    ax.set_title("Señal modulada AM ideal (μ = 0.8) y envolvente", fontsize=10, fontweight="bold", color="#0F172A")
    ax.set_xlabel("Tiempo (ms)", fontsize=8.5)
    ax.set_ylabel("Amplitud (V)", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig("figs/02_am_envolvente.png", dpi=200)
    plt.close()

    # Figura 3: Espectro ideal
    f_ideal, db_ideal = calcular_espectro(senial_am, fs, referencia=1.0)
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    ax.plot(f_ideal / 1000.0, db_ideal, color="#0D9488", lw=1.2)
    ax.set_title("Espectro de magnitud AM ideal (referencia común 1.0 V)", fontsize=10, fontweight="bold", color="#0F172A")
    ax.set_xlabel("Frecuencia (kHz)", fontsize=8.5)
    ax.set_ylabel("Magnitud (dBV)", fontsize=8.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(-100, 5)
    ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig("figs/03_espectro_ideal.png", dpi=200)
    plt.close()

    # Figura 4: Atenuación aislada (6 dB)
    f_aten, db_aten = calcular_espectro(senial_aten, fs, referencia=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(t_ms, senial_am, color="#94A3B8", lw=0.8, ls="--", label="Ideal (0 dB)")
    axes[0].plot(t_ms, senial_aten, color="#2563EB", lw=0.9, label="Atenuada (-6 dB)")
    axes[0].set_title("Tiempo: Atenuación aislada (6 dB)", fontsize=9.5, fontweight="bold", color="#0F172A")
    axes[0].set_xlabel("Tiempo (ms)", fontsize=8.5)
    axes[0].set_ylabel("Amplitud (V)", fontsize=8.5)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3, ls="--")

    axes[1].plot(f_ideal / 1000.0, db_ideal, color="#94A3B8", lw=0.9, ls="--", label="Ideal (0 dB)")
    axes[1].plot(f_aten / 1000.0, db_aten, color="#2563EB", lw=1.1, label="Atenuada (-6 dB)")
    axes[1].set_title("Espectro (misma referencia): -6 dB", fontsize=9.5, fontweight="bold", color="#0F172A")
    axes[1].set_xlabel("Frecuencia (kHz)", fontsize=8.5)
    axes[1].set_ylabel("Magnitud (dBV)", fontsize=8.5)
    axes[1].set_xlim(3.5, 6.5)
    axes[1].set_ylim(-90, 0)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig("figs/04_atenuacion.png", dpi=200)
    plt.close()

    # Figura 5: Ruido aislado (SNR = 10 dB)
    f_ruido, db_ruido = calcular_espectro(senial_ruido, fs, referencia=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(t_ms, senial_ruido, color="#D97706", lw=0.85)
    axes[0].set_title("Tiempo: Ruido aislado (SNR = 10 dB)", fontsize=9.5, fontweight="bold", color="#0F172A")
    axes[0].set_xlabel("Tiempo (ms)", fontsize=8.5)
    axes[0].set_ylabel("Amplitud (V)", fontsize=8.5)
    axes[0].grid(True, alpha=0.3, ls="--")

    axes[1].plot(f_ruido / 1000.0, db_ruido, color="#D97706", lw=1.0)
    axes[1].set_title("Espectro: Ruido aislado (SNR = 10 dB)", fontsize=9.5, fontweight="bold", color="#0F172A")
    axes[1].set_xlabel("Frecuencia (kHz)", fontsize=8.5)
    axes[1].set_ylabel("Magnitud (dBV)", fontsize=8.5)
    axes[1].set_xlim(0, 12)
    axes[1].set_ylim(-100, 0)
    axes[1].grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig("figs/05_ruido.png", dpi=200)
    plt.close()

    # Figura 6: Distorsión aislada (recorte al 65% del pico)
    f_dist, db_dist = calcular_espectro(senial_dist, fs, referencia=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(t_ms, senial_dist, color="#DC2626", lw=0.85)
    axes[0].set_title("Tiempo: Recorte aislado (65% del pico)", fontsize=9.5, fontweight="bold", color="#0F172A")
    axes[0].set_xlabel("Tiempo (ms)", fontsize=8.5)
    axes[0].set_ylabel("Amplitud (V)", fontsize=8.5)
    axes[0].grid(True, alpha=0.3, ls="--")

    axes[1].plot(f_dist / 1000.0, db_dist, color="#DC2626", lw=1.0)
    axes[1].set_title("Espectro: Recorte aislado (65% del pico)", fontsize=9.5, fontweight="bold", color="#0F172A")
    axes[1].set_xlabel("Frecuencia (kHz)", fontsize=8.5)
    axes[1].set_ylabel("Magnitud (dBV)", fontsize=8.5)
    axes[1].set_xlim(0, 12)
    axes[1].set_ylim(-100, 0)
    axes[1].grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig("figs/06_distorsion.png", dpi=200)
    plt.close()


def main():
    fs = 50_000.0
    duracion = 0.02
    fm = 250.0
    fc = 5_000.0
    mu = 0.8

    t, mensaje, portadora = generar_senales(fs=fs, duracion=duracion, fm=fm, fc=fc)
    senial_am, _ = modular_am(t, mensaje, fc=fc, ac=1.0, mu=mu)

    senial_aten = simular_canal(senial_am, atenuacion_db=6.0, snr_db=None, distorsion_clip=None)
    senial_ruido = simular_canal(senial_am, atenuacion_db=0.0, snr_db=10.0, distorsion_clip=None)
    senial_dist = simular_canal(senial_am, atenuacion_db=0.0, snr_db=None, distorsion_clip=0.65)

    generar_figuras(t, mensaje, portadora, senial_am, senial_aten, senial_ruido, senial_dist, fs)


if __name__ == "__main__":
    main()