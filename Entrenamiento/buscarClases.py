import os
import sys

def buscar_archivos_con_numero(carpeta, numero):
    archivos_con_numero = []

    # Recorre todos los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta_archivo) and ruta_archivo.endswith('.txt'):
            with open(ruta_archivo, 'r') as f:
                # Lee cada línea del archivo
                for linea in f:
                    # Verifica si la línea tiene al menos una palabra
                    palabras = linea.strip().split()
                    if palabras and palabras[0].isdigit() and palabras[0] == numero:
                        archivos_con_numero.append(os.path.splitext(archivo)[0])
                        break  # Detener la búsqueda en este archivo una vez encontrado el número

    return archivos_con_numero

def main():
    if len(sys.argv) != 3:
        print("Uso: python script.py <carpeta> <numero>")
        return

    carpeta = sys.argv[1]
    numero = sys.argv[2]

    if not os.path.exists(carpeta):
        print("La carpeta especificada no existe.")
        return

    archivos_encontrados = buscar_archivos_con_numero(carpeta, numero)

    if archivos_encontrados:
        with open('archivos_con_numero.txt', 'w') as f:
            for archivo in archivos_encontrados:
                f.write(archivo + '\n')
        print("Se han encontrado los siguientes archivos que contienen el número:", numero)
        print("\n".join(archivos_encontrados))
    else:
        print("No se encontraron archivos que cumplan con los criterios.")

if __name__ == "__main__":
    main()



