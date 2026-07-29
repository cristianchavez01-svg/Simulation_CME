"""
Collage multi-instrumento del CME del 8 de diciembre de 2022.
Evento: eyección asociada a filamento sobre el limbo SO. Primera aparición en
LASCO/C2 a las 04:12 UT, halo en C3 a las 10:42 UT (PA 240°). Publicado en
Wietske et al. 2024 (A&A, arXiv:2402.14682) y estudios asociados de PyThea/GCS.

Fuente de datos: Helioviewer API v2 (https://api.helioviewer.org).
Nota: PSP/WISPR y Solar Orbiter/SoloHI NO están indexados en Helioviewer;
si quieres reproducir el panel de WISPR del paper original, hay que bajarlo
aparte de https://wispr.nrl.navy.mil/wisprdata y añadirlo manualmente.

Salidas:
    outputs/preview_collage_20221208.png   -> collage vistoso para revisión rápida
    outputs/panels/panel_a_*.png ...        -> paneles individuales en alta resolución
    outputs/figura_cme_20221208.tex         -> snippet listo para pegar en la tesis
                                               (memoir + subcaption)

Requisitos:
    pip install requests pillow matplotlib numpy
"""

import io
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

API = "https://api.helioviewer.org/v2"

# ---------------------------------------------------------------------------
# Paneles: ruta en el árbol de datasources, fecha/hora UT solicitada, escala
# (arcsec/px), semiancho de campo (arcsec), color distintivo de la misión,
# etiqueta corta (para el .tex) y pie de figura completo (para subcaption).
# ---------------------------------------------------------------------------
COLOR_SDO = "#3fa7ff"
COLOR_SOHO = "#ffcf40"
COLOR_STA = "#4cd964"

PANELS = [
    dict(letter="a", path=["SDO", "AIA", "304"], color=COLOR_SDO,
         date="2022-12-08T04:00:00Z", scale=0.6, half_fov=1100,
         nombre="SDO/AIA 304 Å",
         caption="Región fuente del filamento eruptivo visto en 304~\\AA{} desde SDO."),
    dict(letter="b", path=["SDO", "AIA", "193"], color=COLOR_SDO,
         date="2022-12-08T04:00:00Z", scale=0.6, half_fov=1100,
         nombre="SDO/AIA 193 Å",
         caption="Corona baja en 193~\\AA{}, mismo instante, mostrando el entorno coronal de la erupción."),
    dict(letter="c", path=["SOHO", "LASCO", "C2", "white-light"], color=COLOR_SOHO,
         date="2022-12-08T04:12:00Z", scale=11.9, half_fov=3200,
         nombre="LASCO C2",
         caption="Primera aparición del CME en el campo de LASCO/C2 (04:12~UT)."),
    dict(letter="d", path=["SOHO", "LASCO", "C3", "white-light"], color=COLOR_SOHO,
         date="2022-12-08T10:42:00Z", scale=56.0, half_fov=15000,
         nombre="LASCO C3",
         caption="Estructura de halo desarrollada en LASCO/C3 (10:42~UT, PA~240\\textdegree{})."),
    dict(letter="e", path=["STEREO_A", "SECCHI", "EUVI", "195"], color=COLOR_STA,
         date="2022-12-08T04:00:00Z", scale=1.6, half_fov=1400,
         nombre="STEREO-A/EUVI 195 Å",
         caption="Vista de la región fuente desde el punto de observación de STEREO-A."),
    dict(letter="f", path=["STEREO_A", "SECCHI", "COR2", "white-light"], color=COLOR_STA,
         date="2022-12-08T16:23:00Z", scale=14.7, half_fov=7500,
         nombre="STEREO-A/COR2",
         caption="El mismo CME, ahora visto desde STEREO-A/COR2, evidenciando la diferencia de perspectiva respecto a SOHO."),
]

EVENT_LABEL = "CME del 8 de diciembre de 2022"

# Carpeta de salida: siempre junto al .py, sin importar desde dónde se ejecute
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs_cme_20221208")
PANEL_DIR = os.path.join(OUT_DIR, "panels")
PREVIEW_FILE = os.path.join(OUT_DIR, "preview_collage_20221208.png")
TEX_FILE = os.path.join(OUT_DIR, "figura_cme_20221208.tex")
PANEL_PX = 900  # resolución de cada panel individual, en píxeles


def find_source_id(node, path):
    if not path:
        return node.get("sourceId") if isinstance(node, dict) else None
    name, resto = path[0], path[1:]
    contenedor = node.get("children", node) if isinstance(node, dict) else node
    if name not in contenedor:
        raise KeyError(f"No se encontró '{name}' (disponibles: {list(contenedor.keys())})")
    return find_source_id(contenedor[name], resto)


def fetch_panel(source_id, date, scale, half_fov, size_px):
    params = {
        "date": date,
        "imageScale": scale,
        "layers": f"[{source_id},1,100]",
        "x1": -half_fov, "y1": -half_fov,
        "x2": half_fov, "y2": half_fov,
        "width": size_px, "height": size_px,
        "display": "true",
        "watermark": "false",
    }
    r = requests.get(f"{API}/takeScreenshot/", params=params, timeout=60)
    r.raise_for_status()
    from PIL import Image
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def main():
    os.makedirs(PANEL_DIR, exist_ok=True)
    print("Consultando árbol de fuentes de Helioviewer...")
    datasources = requests.get(f"{API}/getDataSources/", timeout=30).json()

    imgs = []
    for p in PANELS:
        print(f"Descargando panel {p['letter']}: {p['nombre']} ({p['date']})...")
        try:
            sid = find_source_id(datasources, p["path"])
            img = fetch_panel(sid, p["date"], p["scale"], p["half_fov"], PANEL_PX)
        except Exception as e:
            print(f"  -> Falló ({e}); se usa panel gris de reemplazo.")
            from PIL import Image
            img = Image.new("RGB", (PANEL_PX, PANEL_PX), (40, 40, 40))
        panel_path = os.path.join(PANEL_DIR, f"panel_{p['letter']}_{p['nombre'].split('/')[0].lower()}.png")
        img.save(panel_path, dpi=(300, 300))
        imgs.append((p, img, panel_path))

    # ------------------------------------------------------------------
    # Collage "llamativo": fondo oscuro tipo espacio, borde de color por
    # misión, letra de panel, título general y pie con la fecha del evento.
    # ------------------------------------------------------------------
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(15, 10.2), facecolor="#05060c")
    rng = np.random.default_rng(42)
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg_ax.set_facecolor("#05060c")
    bg_ax.scatter(rng.random(220), rng.random(220), s=rng.random(220) * 3,
                  color="white", alpha=0.5, linewidths=0)
    bg_ax.set_xticks([]); bg_ax.set_yticks([])
    bg_ax.axis("off")

    fig.suptitle(EVENT_LABEL, color="white", fontsize=22, fontweight="bold", y=0.965)
    fig.text(0.5, 0.935,
             "Observación simultánea multi-instrumento — SDO, SOHO/LASCO y STEREO-A/SECCHI",
             color="#9fb3c8", fontsize=12, ha="center")

    n = len(imgs)
    cols = 3
    rows = -(-n // cols)
    gs = fig.add_gridspec(rows, cols, left=0.03, right=0.97, top=0.88, bottom=0.06,
                           wspace=0.06, hspace=0.18)

    for i, (p, img, _) in enumerate(imgs):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(np.asarray(img))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(p["color"])
            spine.set_linewidth(3)
        ax.text(0.03, 0.96, p["letter"], transform=ax.transAxes, color=p["color"],
                 fontsize=20, fontweight="bold", va="top", ha="left",
                 bbox=dict(boxstyle="circle", facecolor="#05060c", edgecolor=p["color"], pad=0.35))
        ax.text(0.5, -0.045, f"{p['nombre']}  ·  {p['date'][11:16]} UT",
                 transform=ax.transAxes, color="white", fontsize=10.5, ha="center", va="top")

    fig.text(0.5, 0.015,
              "Datos: Helioviewer Project (SOHO/LASCO, SDO/AIA, STEREO-A/SECCHI)",
              color="#6c7a89", fontsize=9, ha="center")

    fig.savefig(PREVIEW_FILE, dpi=200, facecolor=fig.get_facecolor())
    print(f"\nPreview guardado en: {PREVIEW_FILE}")

    # ------------------------------------------------------------------
    # Snippet de LaTeX (memoir + subcaption) para insertar en la tesis.
    # ------------------------------------------------------------------
    tex = []
    tex.append(r"\begin{figure}[htbp]")
    tex.append(r"    \centering")
    for p, _, panel_path in imgs:
        rel = os.path.relpath(panel_path, OUT_DIR)
        tex.append(r"    \begin{subfigure}[b]{0.32\textwidth}")
        tex.append(r"        \centering")
        tex.append(rf"        \includegraphics[width=\textwidth]{{{rel}}}")
        tex.append(rf"        \caption{{{p['caption']}}}")
        tex.append(rf"        \label{{fig:cme20221208_{p['letter']}}}")
        tex.append(r"    \end{subfigure}")
        if p["letter"] in ("b", "d"):
            tex.append(r"    \\[0.6em]")
        else:
            tex.append(r"    \hfill")
    tex.append(rf"    \caption[{EVENT_LABEL}]{{{EVENT_LABEL}, observado simultáneamente por SDO, "
               r"SOHO/LASCO y STEREO-A/SECCHI. Datos obtenidos mediante la API de Helioviewer "
               r"(\url{{https://api.helioviewer.org}}).}}")
    tex.append(r"    \label{fig:cme20221208}")
    tex.append(r"\end{figure}")

    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Snippet LaTeX guardado en: {TEX_FILE}")


if __name__ == "__main__":
    main()