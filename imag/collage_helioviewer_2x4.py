"""
Collage 2x4 a partir de capturas de pantalla YA TOMADAS en Helioviewer.

Lógica: cada nombre de archivo que exporta Helioviewer sigue el patrón
    YYYY_MM_DD_HH_MM_SS_<capa1>__<capa2>__<capa3>...png
donde las capas van separadas por doble guion bajo "__". Este script:

  1. Busca todas las imágenes .png en IMAGES_DIR que sigan ese patrón.
  2. Agrupa por instante de tiempo (mismo YYYY_MM_DD_HH_MM_SS).
  3. Dentro de cada instante, separa en dos filas según el punto de vista:
       - Fila superior (B): SOHO + SDO [+ PUNCH]
       - Fila inferior (A): STEREO-A
  4. Arma una grilla de 2 (punto de vista) x N_COLS (tiempo, creciente hacia
     la derecha), SIN títulos de fila/columna, SIN bordes ni fondo entre
     celdas, y sin distorsionar las imágenes (se recortan para llenar la
     celda manteniendo su proporción original, en vez de estirarlas).
     Cada panel queda marcado únicamente con su letra (a), (b), (c)...
     sobre un recuadro blanco en la esquina superior izquierda, en tipografía
     Computer Modern (la de LaTeX por defecto), para poder referenciarlos
     directamente desde el \\caption en la tesis.

No hace ninguna descarga: usa las imágenes que ya tengas guardadas junto al
script (o en la carpeta que indiques en IMAGES_DIR).

Requisitos:
    pip install pillow
"""

import os
import re
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están las capturas de Helioviewer. Por defecto, la misma
# carpeta del script; cámbiala si las guardaste en otro lado.
IMAGES_DIR = SCRIPT_DIR

# Carpeta de salida (se crea junto al .py)
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs_collage_helioviewer")
OUT_FILE_PNG = os.path.join(OUT_DIR, "collage_2x4.png")
OUT_FILE_PDF = os.path.join(OUT_DIR, "collage_2x4.pdf")

N_COLS = 4              # número de instantes de tiempo a mostrar (columnas)
CELL_W, CELL_H = 900, 900   # resolución de cada panel, en píxeles (subir para más calidad)
PDF_DPI = 300            # resolución embebida en el PDF (afecta el tamaño físico en LaTeX)

FALLBACK_BG = (25, 25, 30)  # solo para celdas sin imagen disponible

FILENAME_RE = re.compile(
    r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(.+)\.png$", re.IGNORECASE
)

STEREO_A_TOKENS = ("COR1-A", "COR2-A", "EUVI-A")
SOHO_SDO_TOKENS = ("LASCO", "AIA", "PUNCH")


def parse_filename(fname):
    """Devuelve (timestamp_str, etiqueta_legible, tipo_columna) o None si no matchea."""
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    y, mo, d, h, mi, s, layer_part = m.groups()
    timestamp = f"{y}_{mo}_{d}_{h}_{mi}_{s}"
    ts_label = f"{y}-{mo}-{d}  {h}:{mi} UT"
    layers = layer_part.split("__")
    layer_label = " + ".join(l.replace("_", " ") for l in layers)

    joined = layer_part.upper()
    es_stereo_a = any(tok in joined for tok in STEREO_A_TOKENS)
    es_soho_sdo = any(tok in joined for tok in SOHO_SDO_TOKENS)
    if es_stereo_a and not es_soho_sdo:
        columna = "A"
    elif es_soho_sdo and not es_stereo_a:
        columna = "B"
    else:
        columna = None  # ambiguo o mixto; se ignora
    return timestamp, ts_label, layer_label, columna


def collect_images(images_dir):
    """Escanea la carpeta y arma dict: timestamp -> {'A': (path,label,layers), 'B': (...)}"""
    grupos = {}
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(".png"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        timestamp, ts_label, layer_label, columna = parsed
        if columna is None:
            print(f"Aviso: '{fname}' no se pudo clasificar en columna A/B, se omite.")
            continue
        grupos.setdefault(timestamp, {"label": ts_label})
        grupos[timestamp][columna] = (os.path.join(images_dir, fname), layer_label)
    return grupos


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def load_latex_font(size):
    """Computer Modern Roman (tipografía por defecto de LaTeX), tomada de la
    instalación de matplotlib (no requiere tener LaTeX instalado)."""
    try:
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "cmr10.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        return load_font(size, bold=True)


def build_collage(images_dir=IMAGES_DIR, n_cols=N_COLS,
                   out_png=OUT_FILE_PNG, out_pdf=OUT_FILE_PDF):
    grupos = collect_images(images_dir)
    if not grupos:
        raise SystemExit(
            f"No se encontraron imágenes con el patrón esperado en:\n  {images_dir}\n"
            "Verifica que los .png de Helioviewer estén ahí (o ajusta IMAGES_DIR)."
        )

    timestamps = sorted(grupos.keys())[:n_cols]
    print(f"Se usarán {len(timestamps)} instantes de tiempo (columnas, de izq. a der.):")
    for ts in timestamps:
        print(f"  {grupos[ts]['label']}  ->  A: {'sí' if 'A' in grupos[ts] else 'NO'}"
              f"   B: {'sí' if 'B' in grupos[ts] else 'NO'}")

    # Tamaño de letra y márgenes proporcionales al tamaño de celda (calibrado
    # para que se vea bien tanto en 380px como en 900px o más)
    font_size = max(14, round(CELL_W * 26 / 380))
    margin = max(4, round(CELL_W * 8 / 380))
    font_letter = load_latex_font(font_size)

    W = len(timestamps) * CELL_W
    H = 2 * CELL_H
    canvas = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(canvas)

    # a, b, c, d en la fila superior; e, f, g, h en la fila inferior
    letras = [chr(ord("a") + k) for k in range(2 * len(timestamps))]

    for i, fila in enumerate(["B", "A"]):  # B = SOHO+SDO(+PUNCH) arriba, A = STEREO-A abajo
        y0 = i * CELL_H
        for j, ts in enumerate(timestamps):
            x0 = j * CELL_W
            if fila in grupos[ts]:
                path, _layer_label = grupos[ts][fila]
                try:
                    img = Image.open(path).convert("RGB")
                    # Recorta y ajusta manteniendo la proporción (sin estirar)
                    img = ImageOps.fit(img, (CELL_W, CELL_H), method=Image.LANCZOS,
                                        centering=(0.5, 0.5))
                except Exception as e:
                    print(f"Aviso: no se pudo cargar '{path}' ({e}); celda en blanco.")
                    img = Image.new("RGB", (CELL_W, CELL_H), FALLBACK_BG)
            else:
                img = Image.new("RGB", (CELL_W, CELL_H), FALLBACK_BG)

            canvas.paste(img, (x0, y0))

            # Etiqueta (a), (b), (c)... sobre un recuadro blanco, esquina sup. izq.
            letra = letras[i * len(timestamps) + j]
            texto = f"({letra})"
            pad = max(3, round(CELL_W * 6 / 380))
            bbox = draw.textbbox((0, 0), texto, font=font_letter)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            box_x0, box_y0 = x0 + margin, y0 + margin
            box_x1 = box_x0 + tw + 2 * pad
            box_y1 = box_y0 + th + 2 * pad
            draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill="white")
            draw.text((box_x0 + pad - bbox[0], box_y0 + pad - bbox[1]), texto,
                       fill="black", font=font_letter)

    os.makedirs(OUT_DIR, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_pdf, "PDF", resolution=PDF_DPI)
    print(f"\nCollage guardado en:\n  {out_png}\n  {out_pdf}  ({PDF_DPI} DPI)")


if __name__ == "__main__":
    build_collage()
