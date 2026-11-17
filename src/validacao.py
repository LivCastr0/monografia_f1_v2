"""
Etapa 2 - Validação dos subgrupos encontrados por busca.py.

1. Elegível: 4 ou mais circuitos e nenhum circuito com mais de 40% das linhas
   (a mesma exigência vale dentro das permutações).
2. Limiar por permutação. Cada repetição embaralha ao mesmo tempo:
     - o alvo entre as linhas da mesma corrida;
     - os perfis entre os 26 circuitos (se o perfil é descritor);
     - a era entre as corridas do mesmo circuito (se a era é descritor).
   Guarda o maior |z| entre todos os subgrupos elegíveis de cada repetição; o
   limiar é o quantil de 95% desses máximos. Isso já desconta o número de testes.
3. Firme: passa do limiar; mantém o sentido nas duas eras (quando a era não é
   condição do subgrupo) e sem o circuito mais frequente.
4. Redundância: um firme fica se muda a média em pelo menos `min_ganho` e
   `min_z` erros-padrão contra cada versão mais simples dele (inclusive a base).

Os sorteios seguem a mesma ordem da versão 1, para o limiar ser reproduzível.
A diferença é de velocidade: os subgrupos sem perfil e sem era não mudam de
cobertura entre permutações, então todas as permutações deles saem numa conta só.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from busca import matriz, mascara, medidas


def elegivel(res, min_circuitos, max_circuito):
    return (res["circuitos"] >= min_circuitos) & (res["frac_maior_circuito"] <= max_circuito)


def sorteios(df, descritores, alvo, rng, permutacoes):
    """Sorteia as permutações na ordem da versão 1."""
    y = df[alvo].to_numpy(dtype=float)
    circ = df["circuito"].to_numpy()
    corrida = (df["ano"].astype(str) + "_" + df["round"].astype(str)).to_numpy()
    por_corrida = list(pd.Series(np.arange(len(df))).groupby(corrida).indices.values())
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    eras = pd.DataFrame({"corrida": corrida, "circuito": circ, "d_era": df["d_era"].to_numpy()}) \
        .drop_duplicates("corrida").set_index("corrida")
    Y = np.empty((len(df), permutacoes), dtype=np.float32)
    perfis, eras_perm = [], []
    for i in range(permutacoes):
        v = y.copy()
        for linhas in por_corrida:
            v[linhas] = y[rng.permutation(linhas)]
        Y[:, i] = v - v.mean()
        if "d_perfil" in descritores:
            novo = pd.Series(rng.permutation(mapa.to_numpy()), index=mapa.index)
            perfis.append(pd.Series(circ).map(novo).to_numpy())
        if "d_era" in descritores:
            nova = eras.groupby("circuito")["d_era"].transform(lambda s: rng.permutation(s.to_numpy()))
            eras_perm.append(pd.Series(corrida).map(nova).to_numpy())
    return Y, perfis, eras_perm


def limiar(df, descritores, alvo, res, rng, permutacoes, suporte, min_circuitos,
           max_circuito, quantil=0.95):
    conds, idx = res.attrs["conds"], res.attrs["idx"]
    dp = df[alvo].std(ddof=1)
    circ_id, circs = pd.factorize(df["circuito"])
    Y, perfis, eras_perm = sorteios(df, descritores, alvo, rng, permutacoes)

    def z_ok(n, soma, por_circ):
        ok = ((n >= suporte) & ((por_circ > 0).sum(axis=1) >= min_circuitos)
              & (por_circ.max(axis=1) <= max_circuito * np.maximum(n, 1)))
        z = np.zeros_like(soma)
        z[ok] = soma[ok] / np.sqrt(n[ok])[:, None] / dp
        return z

    mexe = np.array([i < len(conds) and conds[i][0] in ("d_perfil", "d_era")
                     for i in range(len(conds) + 1)])
    variavel = mexe[idx].any(axis=1)
    maximos = np.zeros(permutacoes)

    # Subgrupos fixos: todas as permutações de uma vez.
    colunas = {c: df[c].to_numpy() for c in descritores}
    m = matriz(colunas, conds, len(df))
    if (~variavel).any():
        n, soma, pc = medidas(m, idx[~variavel], Y, circ_id, len(circs))
        maximos = np.abs(z_ok(n, soma, pc)).max(axis=0)

    # Subgrupos com perfil ou era: refeitos em cada permutação.
    if variavel.any():
        for i in range(permutacoes):
            col = dict(colunas)
            if perfis:
                col["d_perfil"] = perfis[i]
            if eras_perm:
                col["d_era"] = eras_perm[i]
            mi = matriz(col, conds, len(df))
            n, soma, pc = medidas(mi, idx[variavel], Y[:, i:i + 1], circ_id, len(circs))
            maximos[i] = max(maximos[i], np.abs(z_ok(n, soma, pc)).max())
    return np.quantile(maximos, quantil), maximos


def firmeza(df, alvo, res, lim):
    """Acima do limiar, mesmo sentido nas duas eras e sem o circuito mais frequente."""
    y = df[alvo].to_numpy(dtype=float)
    media = y.mean()
    era, circ = df["d_era"].to_numpy(), df["circuito"].to_numpy()
    linhas = []
    for d, z in zip(res["descricao"], res["z"]):
        m = mascara(df, d)
        s = np.sign(y[m].mean() - media)
        por_era = {e: y[m & (era == e)].mean() for e in ("2018-21", "2022+") if (m & (era == e)).any()}
        if "era=" in d:
            ok_eras = True
        else:
            ok_eras = len(por_era) == 2 and all(np.sign(v - media) == s for v in por_era.values())
        maior = pd.Series(circ[m]).value_counts().index[0]
        sem = m & (circ != maior)
        ok_sem = sem.any() and np.sign(y[sem].mean() - media) == s
        acima = abs(z) > lim
        linhas.append({"acima_do_limiar": acima, "ok_eras": ok_eras, "ok_sem_maior": ok_sem,
                       "firme": acima and ok_eras and ok_sem,
                       "media_2018_21": por_era.get("2018-21", np.nan),
                       "media_2022_mais": por_era.get("2022+", np.nan)})
    return pd.DataFrame(linhas, index=res.index)


def redundancia(df, alvo, firmes, min_ganho, min_z):
    """Compara cada firme com todas as versões mais simples dele, inclusive a base.

    O erro-padrão usa p(1-p)/n, como na versão 1 (alvo 0/1).
    """
    y = df[alvo].to_numpy(dtype=float)
    media = y.mean()
    linhas = []
    for d in firmes["descricao"]:
        conds = d.split(" E ")
        m = mascara(df, d)
        n, valor = m.sum(), y[m].mean()
        sinal = np.sign(valor - media)
        pior, pior_z, com = np.inf, np.inf, None
        for k in range(len(conds)):
            for sub in combinations(conds, k):
                base = y[mascara(df, " E ".join(sub))].mean() if sub else media
                ganho = sinal * (valor - base)
                z = ganho / np.sqrt(max(base * (1 - base), 1e-12) / n)
                if ganho < pior:
                    pior, pior_z, com = ganho, z, " E ".join(sub) or "base inteira"
        linhas.append({"menor_ganho": pior, "z_do_menor_ganho": pior_z, "comparado_com": com,
                       "fica": pior >= min_ganho and pior_z >= min_z})
    return pd.DataFrame(linhas, index=firmes.index)
