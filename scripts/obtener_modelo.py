#!/usr/bin/env python3
# CARGA DE DATOS
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

MODELO_URL = ('https://raw.githubusercontent.com/ErickCov14/Actividad5-Vision360/'
              '8d5e9ec6150446b539ebf07cd3e8ab1de3c64eed/resultados/random_forest_vision360.joblib')
MODELO_SHA_GIT = 'bfa6a08fab0f88ffd57f102692a7d7aaa71c3896'
MODELO_BYTES = 10_132_306

def comprobar(destino: Path) -> bool:
    if not destino.is_file():
        return False
    data = destino.read_bytes()
    if len(data) != MODELO_BYTES:
        return False
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == MODELO_SHA_GIT

def obtener(destino: Path):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if comprobar(destino):
        print('Modelo validado, descarga omitida:', destino)
        return
    temporal = destino.with_suffix('.download')
    try:
        request = Request(MODELO_URL, headers={'User-Agent': 'Vision360-FaseII/1.0'})
        with urlopen(request, timeout=120) as res, temporal.open('wb') as fout:
            while True:
                parte = res.read(1024 * 1024)
                if not parte:
                    break
                fout.write(parte)
        if not comprobar(temporal):
            raise RuntimeError('La descarga no coincide con el tamaño y SHA de GitHub. No se utilizará.')
        os.replace(temporal, destino)
    finally:
        temporal.unlink(missing_ok=True)
    print('Modelo original descargado y verificado:', destino)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=Path, default=Path(__file__).resolve().parents[1] / 'backend/model/random_forest_vision360.joblib')
    args = parser.parse_args()
    obtener(args.dest)
