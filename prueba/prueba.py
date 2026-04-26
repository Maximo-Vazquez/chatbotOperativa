from statsmodels.tsa.stattools import acf

# ENTRADAS
datos = [
    100, 102, 101, 103, 104, 106, 107, 109, 110, 112,
    113, 115, 116, 118, 119, 121, 122, 124, 125, 127,
] # Serie de tiempo con autocorrelacion positiva
n_lags = 5 # Cuántos retardos analizar

# PROCESAMIENTO
# 'nlags' es la entrada, 'alpha' genera los intervalos para la interpretación
valores_acf, intervalos = acf(datos, nlags=n_lags, alpha=0.05)

# SALIDAS
print("Coeficientes por cada lag:", valores_acf)
print("Bandas de confianza:", intervalos)
