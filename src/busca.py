"""
Etapa 2 - Motor de descoberta de subgrupos.

Enumera todas as conjunções de até `profundidade` condições "descritor = valor", de
descritores diferentes, com pelo menos `suporte` linhas. Para cada subgrupo calcula
tamanho, média do alvo, diferença para a média geral e o z:

    z = (média do subgrupo - média geral) * raiz(n) / desvio-padrão do alvo

que é a qualidade padrão da SD para alvo numérico (StandardQFNumeric com a = 0,5,
do pysubgroup) dividida pelo desvio-padrão. Ordenar por z ou pela qualidade dá a
mesma ordem; o z tem a vantagem de ser comparável entre análises.

Também guarda, para a validação: número de circuitos do subgrupo e a fração de
linhas no circuito mais frequente.

A conta é matricial: uma matriz com uma linha por condição (1 = a linha da base
satisfaz a condição); a cobertura de um subgrupo é o E das linhas das condições
dele, e a soma do alvo em todos os subgrupos sai de uma multiplicação de matrizes.
"""
import numpy as np
import pandas as pd

BLOCO = 512   # subgrupos calculados de cada vez, para não estourar a memória


def condicoes(df, descritores):
    """Lista de (descritor, valor), na ordem dos descritores e dos valores."""
    return [(c, v) for c in descritores for v in sorted(df[c].dropna().unique(), key=str)]


def matriz(colunas, conds, n):
    """Uma linha por condição, mais uma linha de 1 no fim (condição vazia)."""
    m = np.ones((len(conds) + 1, n), dtype=bool)
    for i, (c, v) in enumerate(conds):
        m[i] = colunas[c] == v
    return m


def enumera(m, conds, suporte, profundidade):
    """Todas as conjunções com suporte mínimo, de descritores diferentes.

    Busca em largura: um subgrupo só é estendido se já tem o suporte mínimo (o
    suporte nunca aumenta ao acrescentar condição). Devolve uma matriz de índices
    de condição, completada com o índice da linha de 1.
    """
    vazia = len(conds)
    attr = [c for c, _ in conds]
    nivel = [(i,) for i in range(len(conds)) if m[i].sum() >= suporte]
    todos = list(nivel)
    for _ in range(1, profundidade):
        novo = []
        for g in nivel:
            cob = np.logical_and.reduce(m[list(g)])
            usados = {attr[i] for i in g}
            for j in range(g[-1] + 1, len(conds)):
                if attr[j] in usados:
                    continue
                if np.count_nonzero(cob & m[j]) >= suporte:
                    novo.append(g + (j,))
        nivel = novo
        todos += novo
    idx = np.full((len(todos), profundidade), vazia)
    for k, g in enumerate(todos):
        idx[k, :len(g)] = g
    return idx


def cobertura(m, idx):
    c = m[idx[:, 0]]
    for j in range(1, idx.shape[1]):
        c = c & m[idx[:, j]]
    return c


def medidas(m, idx, Y, circ_id, n_circ):
    # Y em float32 (permutações) ou float64 (alvo observado); a cobertura segue o tipo de Y.
    """Para cada subgrupo: n, soma do alvo centrado em cada coluna de Y, contagem por
    circuito. Y tem uma coluna por versão do alvo (a original e as permutadas)."""
    n = np.empty(len(idx), dtype=np.int64)
    soma = np.empty((len(idx), Y.shape[1]))
    por_circ = np.empty((len(idx), n_circ), dtype=np.int32)
    um_circ = np.zeros((m.shape[1], n_circ), dtype=np.float32)
    um_circ[np.arange(m.shape[1]), circ_id] = 1
    for a in range(0, len(idx), BLOCO):
        c = cobertura(m, idx[a:a + BLOCO]).astype(Y.dtype)
        n[a:a + BLOCO] = c.sum(axis=1)
        soma[a:a + BLOCO] = c @ Y
        por_circ[a:a + BLOCO] = np.rint(c @ um_circ)
    return n, soma, por_circ


def descreve(conds, g, vazia):
    return " E ".join(f"{conds[i][0].removeprefix('d_')}={conds[i][1]}" for i in g if i != vazia)


def busca(df, descritores, alvo, suporte, profundidade):
    """Todos os subgrupos com suporte mínimo, sem nenhum filtro de validação."""
    conds = condicoes(df, descritores)
    colunas = {c: df[c].to_numpy() for c in descritores}
    m = matriz(colunas, conds, len(df))
    idx = enumera(m, conds, suporte, profundidade)

    y = df[alvo].to_numpy(dtype=float)
    circ_id, circs = pd.factorize(df["circuito"])
    n, soma, por_circ = medidas(m, idx, (y - y.mean())[:, None],
                                circ_id, len(circs))
    soma = soma[:, 0]
    res = pd.DataFrame({
        "descricao": [descreve(conds, g, len(conds)) for g in idx],
        "condicoes": [sum(i != len(conds) for i in g) for g in idx],
        "linhas": n,
        "circuitos": (por_circ > 0).sum(axis=1),
        "frac_maior_circuito": por_circ.max(axis=1) / n,
        "media": y.mean() + soma / n,
        "diferenca": soma / n,
        "z": soma / np.sqrt(n) / y.std(ddof=1),
        "tem_perfil": [any(conds[i][0] == "d_perfil" for i in g if i != len(conds)) for g in idx],
    })
    res.attrs.update(conds=conds, idx=idx, media_geral=y.mean())
    return res


def mascara(df, descricao, ignorar=()):
    """Linhas do subgrupo descrito no formato 'a=x E b=y'."""
    m = np.ones(len(df), dtype=bool)
    if not descricao:
        return m
    for cond in descricao.split(" E "):
        c, v = cond.split("=", 1)
        if c not in ignorar:
            m &= df["d_" + c].astype(str).to_numpy() == v
    return m
