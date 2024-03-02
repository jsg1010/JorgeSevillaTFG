import os
import sys

def eliminar_archivos(nombre_archivos, carpeta):
    try:
        with open(nombre_archivos, 'r') as archivo_nombres:
            nombres = archivo_nombres.read().splitlines()
            for nombre in nombres:
                nombre_sin_extension = os.path.splitext(nombre)[0]
                for extension in ['.txt', '.jpg']:
                    ruta_archivo = os.path.join(carpeta, nombre_sin_extension + extension)
                    print(f"Intentando eliminar: {ruta_archivo}")  # Imprimir la ruta completa del archivo
                    if os.path.exists(ruta_archivo):
                        os.remove(ruta_archivo)
                        print(f"Archivo '{nombre_sin_extension}' eliminado exitosamente.")
                        break  # Salir del bucle si se encuentra el archivo
                else:
                    print(f"Archivo '{nombre_sin_extension}' no encontrado en la carpeta.")
    except FileNotFoundError:
        print("El archivo de nombres especificado no existe.")

def main():
    if len(sys.argv) != 3:
        print("Uso: python script.py <archivo_nombres> <carpeta>")
        return

    archivo_nombres = sys.argv[1]
    carpeta = sys.argv[2]

    if not os.path.exists(archivo_nombres):
        print("El archivo de nombres especificado no existe.")
        return

    if not os.path.exists(carpeta):
        print("La carpeta especificada no existe.")
        return

    eliminar_archivos(archivo_nombres, carpeta)

if __name__ == "__main__":
    main()
