import os
import sys

def reemplazar_numeros(carpeta, numero_busqueda, nuevo_numero):
    for archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta_archivo):
            with open(ruta_archivo, 'r') as f:
                lineas = f.readlines()
            
            with open(ruta_archivo, 'w') as f:
                for linea in lineas:
                    if linea.strip().startswith(numero_busqueda):
                        f.write(linea.replace(numero_busqueda, nuevo_numero))
                    else:
                        f.write(linea)

def main():
    if len(sys.argv) != 4:
        print("Uso: python script.py <carpeta> <numero_busqueda> <nuevo_numero>")
        return

    carpeta = sys.argv[1]
    numero_busqueda = sys.argv[2]
    nuevo_numero = sys.argv[3]

    if not os.path.exists(carpeta):
        print("La carpeta especificada no existe.")
        return

    reemplazar_numeros(carpeta, numero_busqueda, nuevo_numero)
    print("Se han reemplazado los números satisfactoriamente.")

if __name__ == "__main__":
    main()
