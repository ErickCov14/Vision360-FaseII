#!/usr/bin/env python3
# CARGA DE DATOS
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
BASE = 'https://raw.githubusercontent.com/ErickCov14/Actividad5-Vision360/8d5e9ec6150446b539ebf07cd3e8ab1de3c64eed/'
EJEMPLOS = {
    '01_grieta.png': 'datos/datos_ini/muestra_original/testing/Difetto1/RRT-101R_Img2_A80_S2_[17][9].png',
    '02_porosidad.png': 'datos/datos_ini/muestra_original/testing/Difetto2/RRT-102R_Img4_A80_S2_[17][12].png',
    '03_falta_penetracion.png': 'datos/datos_ini/muestra_original/testing/Difetto4/RRT-107R_Img2_A80_S5_[6][108].png',
    '04_sin_defecto.png': 'datos/datos_ini/muestra_original/testing/NoDifetto/RRT-101R_Img4_A80_S6_[18][32].png',
}
DESTINO = Path(__file__).resolve().parents[1] / 'ejemplos'
DESTINO.mkdir(parents=True, exist_ok=True)
for filename, ruta in EJEMPLOS.items():
    request = Request(BASE + quote(ruta, safe='/'), headers={'User-Agent': 'Vision360-FaseII/1.0'})
    with urlopen(request, timeout=60) as r:
        data = r.read(3 * 1024 * 1024)
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise RuntimeError('Ejemplo no disponible o no es PNG: ' + filename)
    (DESTINO / filename).write_bytes(data)
    print('Descargado:', DESTINO / filename)
