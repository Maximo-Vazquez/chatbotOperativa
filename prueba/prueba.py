def secuencia_dos_unos(binario: int, contador: int) -> int:
    if binario  == 0:

        if contador == 2:
            return 1
        else:
            return 0
    else:
        digito = binario % 10

        if contador == 1:
            if digito == 1:
                return secuencia_dos_unos(binario // 10, 0) + 1

                
        else:
            if digito == 1:
                return secuencia_dos_unos(binario // 10, 1)
        return secuencia_dos_unos(binario // 10, 0)
            


print(secuencia_dos_unos(10110111101, 0))  # Ejemplo de uso


