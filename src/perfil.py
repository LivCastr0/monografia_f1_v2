"""
Etapa 2 - Camada de perfil, aplicada igual aos resultados de todas as análises.

1. media_por_perfil: o alvo muda entre os perfis? Teste por permutação dos perfis
   entre os 26 circuitos. É o único teste em que o urbano-técnico pode aparecer:
   com 2 circuitos, ele nunca cumpre a exigência de 4 circuitos da busca.
2. perfil_acrescenta: para cada subgrupo firme com condição de perfil, compara a
   média dele com a do mesmo subgrupo sem a condição de perfil. O limiar vem de
   embaralhar os perfis entre os circuitos e guardar a maior diferença entre todos
   os subgrupos testados (já desconta o número de testes). Existe porque um
   subgrupo "perfil E condição" pode ser firme só por herdar o efeito da condição.
"""
import numpy as np
import pandas as pd

from busca import mascara


def media_por_perfil(df, alvo, rng, permutacoes):
    g = df.groupby("circuito")[alvo].agg(["sum", "size"])
    perfil = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"].reindex(g.index)
    media = g["sum"].sum() / g["size"].sum()

    def separacao(rotulos):
        s = g.groupby(np.asarray(rotulos)).sum()
        return ((s["sum"] / s["size"] - media) ** 2 * s["size"]).sum()

    obs = separacao(perfil)
    maiores = sum(separacao(rng.permutation(perfil.to_numpy())) >= obs for _ in range(permutacoes))
    tabela = df.groupby("d_perfil").agg(linhas=(alvo, "size"), circuitos=("circuito", "nunique"),
                                        media=(alvo, "mean"))
    por_circuito = g.assign(media=g["sum"] / g["size"], perfil=perfil) \
        .rename(columns={"size": "linhas"})[["perfil", "linhas", "media"]] \
        .sort_values("media", ascending=False)
    return (maiores + 1) / (permutacoes + 1), tabela, por_circuito


def perfil_acrescenta(df, alvo, firmes_com_perfil, rng, permutacoes, quantil=0.95):
    y = df[alvo].to_numpy(dtype=float)
    mapa = df.drop_duplicates("circuito").set_index("circuito")["d_perfil"]
    circ = df["circuito"]
    casos = []
    for d in firmes_com_perfil["descricao"]:
        valor = d.split("perfil=")[1].split(" E ")[0]
        resto = mascara(df, d, ignorar=("perfil",))
        casos.append((valor, resto, y[resto].mean(), y[resto].std(ddof=1)))

    def estatisticas(perfis):
        z = np.zeros(len(casos))
        for k, (valor, resto, base, dp) in enumerate(casos):
            m = resto & (perfis == valor)
            if m.any() and dp > 0:
                z[k] = (y[m].mean() - base) * np.sqrt(m.sum()) / dp
        return z

    obs = estatisticas(df["d_perfil"].to_numpy())
    maximos = np.empty(permutacoes)
    for j in range(permutacoes):
        novo = pd.Series(rng.permutation(mapa.to_numpy()), index=mapa.index)
        maximos[j] = np.abs(estatisticas(circ.map(novo).to_numpy())).max()
    lim = np.quantile(maximos, quantil)
    return pd.DataFrame({"media_sem_perfil": [c[2] for c in casos], "z_perfil": obs,
                         "perfil_acrescenta": np.abs(obs) > lim},
                        index=firmes_com_perfil.index), lim
