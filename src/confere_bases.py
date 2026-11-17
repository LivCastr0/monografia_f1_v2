"""
Confere as bases da versão 2 contra as da versão 1, linha a linha.

Uso:
  python src/confere_bases.py CAMINHO_V1
onde CAMINHO_V1 é a pasta data/processed do repositório monografia_f1 (versão 1).

Lê data/bases/ (rode `python src/bases.py` antes). Para cada base, diz quantas
linhas existem só na v1, só na v2, e em quantas linhas comuns algum descritor
difere. Diferença esperada é só a que vem de correção declarada no CHANGELOG.
"""
import sys
from pathlib import Path

import pandas as pd

V2 = Path("data/bases")
COMPARAR = {
    "decisao_parada": ["ano", "round", "piloto", "volta"],
    "duelos": ["ano", "round", "volta", "defensor", "atacante"],
}


def confere(nome, chave, pasta_v1):
    v1 = pd.read_csv(pasta_v1 / f"base_{nome}.csv")
    v2 = pd.read_csv(V2 / f"{nome}.csv")
    print(f"\n== {nome}: v1 {len(v1):,} linhas, v2 {len(v2):,} linhas")
    print(f"   corridas: v1 {v1.groupby(['ano', 'round']).ngroups}, "
          f"v2 {v2.groupby(['ano', 'round']).ngroups}")
    m = v1.merge(v2, on=chave, how="outer", suffixes=("_v1", "_v2"), indicator=True)
    so_v1, so_v2 = (m["_merge"] == "left_only").sum(), (m["_merge"] == "right_only").sum()
    print(f"   só na v1: {so_v1}   só na v2: {so_v2}")
    if so_v1 or so_v2:
        dif = m[m["_merge"] != "both"]
        print("   por corrida:")
        print(dif.groupby(["ano", "round", "_merge"], observed=True).size()
              .unstack(fill_value=0).to_string())
    comuns = m[m["_merge"] == "both"]
    alvo = [c for c in v1.columns if c not in chave and c in v2.columns]
    for c in alvo:
        a, b = comuns[f"{c}_v1"], comuns[f"{c}_v2"]
        if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
            dif = ~((a - b).abs().le(1e-9) | (a.isna() & b.isna()))
        else:
            dif = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna()))
        if dif.any():
            print(f"   {c}: {int(dif.sum())} linhas diferentes")
    return so_v1 + so_v2


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    pasta_v1 = Path(sys.argv[1])
    for nome, chave in COMPARAR.items():
        confere(nome, chave, pasta_v1)


if __name__ == "__main__":
    main()
