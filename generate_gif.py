#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from PIL import Image

def crear_gif_ordenado(folder, output_name="secuencia.gif", duration=200, loop=0):
    """
    Busca en la carpeta 'folder' todos los archivos con nombre tipo
    "compare_epNNN_end.png", donde NNN es un número de episodio (con ceros a la izquierda).
    Los ordena por ese número y genera un GIF animado con la secuencia.

    Parámetros:
    - folder: ruta de la carpeta que contiene los PNGs.
    - output_name: nombre del archivo GIF de salida (por defecto "secuencia.gif").
    - duration: tiempo (en milisegundos) que dura cada frame (por defecto 200 ms).
    - loop: cuántas veces se repite el GIF (0 = infinitas repeticiones).
    """
    # Expresión regular para capturar el número de episodio
    patrón = re.compile(r"^compare_ep(\d+)_end\.png$", re.IGNORECASE)

    # 1. Recorremos todos los archivos de la carpeta y filtramos los que coincidan con el patrón
    episodios = []
    for nombre in os.listdir(folder):
        match = patrón.match(nombre)
        if match:
            num_ep = int(match.group(1))  # extrae el número como entero
            episodios.append((num_ep, nombre))

    # 2. Verificamos si hay archivos válidos
    if not episodios:
        print(f"No se encontraron archivos del tipo 'compare_epNNN_end.png' en {folder}")
        return

    # 3. Ordenamos la lista por número de episodio (el primer elemento de cada tupla)
    episodios.sort(key=lambda x: x[0])

    # 4. Cargamos cada imagen en la lista 'frames', siguiendo el orden de episodios,
    #    y convertimos a modo "P" (paleta) con paleta adaptativa.
    frames = []
    for _, nombre in episodios:
        ruta_completa = os.path.join(folder, nombre)
        try:
            img = Image.open(ruta_completa)
            # 4.1 Convertimos primero a RGB (esto elimina cualquier canal alfa o formato extraño)
            img_rgb = img.convert("RGB")
            # 4.2 Convertimos a "P" (paleta) con paleta adaptativa (256 colores)
            img_p = img_rgb.convert("P", palette=Image.ADAPTIVE)
            frames.append(img_p)
        except Exception as e:
            print(f"Error al abrir/convertir {ruta_completa}: {e}")

    # 5. Creamos el GIF a partir de la primera imagen + resto como append_images
    output_path = os.path.join(folder, output_name)
    try:
        frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop
        )
        print(f"GIF creado correctamente en: {output_path}")
    except Exception as e:
        print(f"Error al guardar el GIF: {e}")


if __name__ == "__main__":
    # -----------------------------
    #  CONFIGURA AQUÍ TU CARPETA
    # -----------------------------
    carpeta_con_pngs = "./visualizations"  # ← Cambia esto a la ruta donde están tus PNGs

    crear_gif_ordenado(
        folder=carpeta_con_pngs,
        output_name="secuencia.gif",  # nombre final del GIF
        duration=150,                # milisegundos por frame
        loop=0                       # 0 = repetir infinitamente
    )
