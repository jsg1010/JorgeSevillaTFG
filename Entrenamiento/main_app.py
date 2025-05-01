import tkinter as tk
from tkinter import filedialog
import subprocess

def cargar_video():
    filename = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.mp4")])
    if filename:
        nombre_video.set(filename)

def iniciar_detect():
    if nombre_video.get():
        # Ejecutar el script detect.py usando subprocess
        subprocess.run(["python", "detect.py", nombre_video.get()])
        
# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Subir Video")

# Definir el tamaño de la ventana
ventana.geometry("720x480")  # Ancho x Alto

# Cambiar el color de fondo
ventana.configure(bg="blue")

# Variable para almacenar el nombre del video
nombre_video = tk.StringVar()

# Etiqueta para mostrar el nombre del video seleccionado
lbl_nombre_video = tk.Label(ventana, textvariable=nombre_video)
lbl_nombre_video.pack()

# Botón para cargar el video
btn_cargar = tk.Button(ventana, text="Cargar Video", command=cargar_video)
btn_cargar.pack(side='bottom', pady=10)

# Botón para iniciar la detección
btn_iniciar = tk.Button(ventana, text="Iniciar Detección", command=iniciar_detect)
btn_iniciar.pack(side='bottom', pady=10)

ventana.mainloop()


