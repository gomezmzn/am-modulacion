# Análisis de un Sistema de Modulación en Amplitud (AM)

Implementación en Python de un sistema de modulación en amplitud con portadora transmitida (DSB-LC) para analizar su comportamiento en el dominio del tiempo y de la frecuencia frente a escenarios aislados de atenuación, ruido aditivo blanco gaussiano (AWGN) y distorsión por recorte.

## Parámetros de simulación
- **Frecuencia de muestreo ($f_s$):** 50,000 Hz
- **Duración:** 0.02 s (1,000 muestras)
- **Señal mensaje $m(t)$:** Suma de senoides de 250 Hz (amplitud 0.8) y 500 Hz (amplitud 0.2)
- **Señal portadora $c(t)$:** Coseno de 5,000 Hz ($A_c = 1.0\text{ V}$)
- **Índice de modulación ($\mu$):** 0.8

## Escenarios evaluados
1. **Referencia ideal:** 0 dB de atenuación, sin ruido, sin recorte.
2. **Atenuación aislada:** Pérdida de 6 dB, sin ruido, sin recorte.
3. **Ruido aislado:** 0 dB de atenuación, ruido AWGN con SNR objetivo de 10 dB, sin recorte.
4. **Distorsión aislada:** 0 dB de atenuación, sin ruido, recorte simétrico al 65 % del valor pico.

## Instalación y ejecución
Desde la raíz del repositorio:

```bash
pip install -r requerimientos.txt
python actvidad4.py