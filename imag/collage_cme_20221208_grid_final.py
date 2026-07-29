"""
Grilla animada FINAL del CME del 8 de diciembre de 2022: los 6 paneles
(SDO/AIA 304, SDO/AIA 193, LASCO C2, LASCO C3, STEREO-A/EUVI 195, STEREO-A/COR2)
avanzan juntos sobre el MISMO eje temporal absoluto (UT), para que la grilla
sea una animación real de multi-punto de vista y no solo un "mismo índice de
cuadro" desincronizado.

Ventana común: 2022-12-08 02:00 UT -> 11:00 UT (9 h), cada 20 min -> 28 cuadros.
Ajusta GLOBAL_START / GLOBAL_END / STEP_MIN si quieres otra duración/cadencia.

Fuente de datos: Helioviewer API v2 (https://api.helioviewer.org).

Requisitos:
    pip install requests pillow

Salidas (junto al .py):
    outputs_cme_20221208_grid/
        grid_cme_20221208.gif          -> GIF final con los 6 paneles juntos
        grid_frames/frame_000.png ...  -> cada cuadro de la grilla, para armar
                                          el mp4 con ffmpeg (comando al final)
"""

import io
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from PIL import Image, ImageDraw, ImageFont

API = "https://api.helioviewer.org/v2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs_cme_20221208_grid")
GRID_FRAMES_DIR = os.path.join(OUT_DIR, "grid_frames")
GIF_FILE = os.path.join(OUT_DIR, "grid_cme_20221208.gif")

# --- eje temporal compartido por todos los paneles --------------------------
GLOBAL_START = "2022-12-08T02:00:00Z"
GLOBAL_END = "2022-12-08T11:00:00Z"
STEP_MIN = 20

CELL_PX = 380          # tamaño de cada panel dentro de la grilla
LABEL_H = 34           # alto reservado para el nombre del instrumento
HEADER_H = 46          # alto reservado para el reloj UT en la parte superior
FRAME_MS = 220         # duración de cada cuadro del gif (ms)
REQUEST_PAUSE = 0.3    # pausa entre llamadas a la API

COLOR_SDO = "#3fa7ff"
COLOR_SOHO = "#ffcf40"
COLOR_STA = "#4cd964"
BG_COLOR = (5, 6, 12)

PANELS = [
    dict(letter="a", path=["SDO", "AIA", "304"], color=COLOR_SDO,
         nombre="SDO/AIA 304", scale=0.6, half_fov=1100),
    dict(letter="b", path=["SDO", "AIA", "193"], color=COLOR_SDO,
         nombre="SDO/AIA 193", scale=0.6, half_fov=1100),
    dict(letter="c", path=["SOHO", "LASCO", "C2", "white-light"], color=COLOR_SOHO,
         nombre="LASCO C2", scale=11.9, half_fov=3200),
    dict(letter="d", path=["SOHO", "LASCO", "C3", "white-light"], color=COLOR_SOHO,
         nombre="LASCO C3", scale=56.0, half_fov=15000),
    dict(letter="e", path=["STEREO_A", "SECCHI", "EUVI", "195"], color=COLOR_STA,
         nombre="STEREO-A/EUVI 195", scale=1.6, half_fov=1400),
    dict(letter="f", path=["STEREO_A", "SECCHI", "COR2", "white-light"], color=COLOR_STA,
         nombre="STEREO-A/COR2", scale=14.7, half_fov=7500),
]


def find_source_id(node, path):
    if not path:
        return node.get("sourceId") if isinstance(node, dict) else None
    name, resto = path[0], path[1:]
    contenedor = node.get("children", node) if isinstance(node, dict) else node
    if name not in contenedor:
        raise KeyError(f"No se encontró '{name}' (disponibles: {list(contenedor.keys())})")
    return find_source_id(contenedor[name], resto)


def time_axis(start_iso, end_iso, step_min):
    t = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    steps = []
    while t <= end:
        steps.append(t)
        t += timedelta(minutes=step_min)
    return steps


def fetch_frame(session, source_id, date_iso, scale, half_fov, size_px):
    params = {
        "date": date_iso,
        "imageScale": scale,
        "layers": f"[{source_id},1,100]",
        "x1": -half_fov, "y1": -half_fov,
        "x2": half_fov, "y2": half_fov,
        "width": size_px, "height": size_px,
        "display": "true",
        "watermark": "false",
    }
    r = session.get(f"{API}/takeScreenshot/", params=params, timeout=60)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def load_font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def compose_grid_frame(images_by_letter, t_utc, cols=3):
    rows = -(-len(PANELS) // cols)
    W = cols * CELL_PX
    H = HEADER_H + rows * (CELL_PX + LABEL_H)
    canvas = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    font_header = load_font(22, bold=True)
    font_label = load_font(15)

    ts_str = t_utc.strftime("%Y-%m-%d  %H:%M UT")
    draw.text((W / 2, HEADER_H / 2), ts_str, fill="white", font=font_header, anchor="mm")

    for i, p in enumerate(PANELS):
        x = (i % cols) * CELL_PX
        y = HEADER_H + (i // cols) * (CELL_PX + LABEL_H)
        img = images_by_letter.get(p["letter"])
        if img is None:
            img = Image.new("RGB", (CELL_PX, CELL_PX), (40, 40, 40))
        else:
            img = img.resize((CELL_PX, CELL_PX))
        canvas.paste(img, (x, y))
        draw.rectangle([x, y, x + CELL_PX - 1, y + CELL_PX - 1], outline=p["color"], width=3)
        draw.rectangle([x, y + CELL_PX, x + CELL_PX, y + CELL_PX + LABEL_H], fill=BG_COLOR)
        draw.text((x + CELL_PX / 2, y + CELL_PX + LABEL_H / 2), p["nombre"],
                   fill="white", font=font_label, anchor="mm")

    return canvas


def main():
    os.makedirs(GRID_FRAMES_DIR, exist_ok=True)

    print("Consultando árbol de fuentes de Helioviewer...")
    datasources = requests.get(f"{API}/getDataSources/", timeout=30).json()
    source_ids = {p["letter"]: find_source_id(datasources, p["path"]) for p in PANELS}

    steps = time_axis(GLOBAL_START, GLOBAL_END, STEP_MIN)
    print(f"Eje temporal: {len(steps)} cuadros de {GLOBAL_START} a {GLOBAL_END} "
          f"cada {STEP_MIN} min\n")

    grid_frames = []
    with requests.Session() as session:
        for i, t in enumerate(steps):
            date_iso = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"Cuadro {i+1}/{len(steps)}  ({date_iso})")
            images_by_letter = {}
            for p in PANELS:
                try:
                    img = fetch_frame(session, source_ids[p["letter"]], date_iso,
                                       p["scale"], p["half_fov"], CELL_PX)
                    images_by_letter[p["letter"]] = img
                except Exception as e:
                    print(f"  panel {p['letter']} ({p['nombre']}) falló: {e}")
                time.sleep(REQUEST_PAUSE)

            frame = compose_grid_frame(images_by_letter, t)
            frame_path = os.path.join(GRID_FRAMES_DIR, f"frame_{i:03d}.png")
            frame.save(frame_path)
            grid_frames.append(frame)

    if not grid_frames:
        print("No se generó ningún cuadro; revisa la conexión con la API.")
        return

    grid_frames[0].save(
        GIF_FILE, save_all=True, append_images=grid_frames[1:],
        duration=FRAME_MS, loop=0, optimize=False,
    )
    print(f"\nGIF final guardado en: {GIF_FILE}")
    print(f"Cuadros PNG guardados en: {GRID_FRAMES_DIR} (para mp4 con ffmpeg):")
    print(f"  ffmpeg -framerate {1000/FRAME_MS:.1f} -i {GRID_FRAMES_DIR}/frame_%03d.png "
          f"-pix_fmt yuv420p {OUT_DIR}/grid_cme_20221208.mp4")


if __name__ == "__main__":
    main()
