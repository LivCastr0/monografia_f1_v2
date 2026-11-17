"""
Etapa 2 - Bases de análise, montadas a partir das coletas.

Tudo sai de duas tabelas centrais:
  corridas()     uma linha por corrida: circuito, perfil, era, chuva, distância real,
                 e as marcas que decidem se a corrida entra nas análises
  piloto_volta() uma linha por piloto por volta: posição, distâncias, pneu, box,
                 parada, bandeira no ponto de decisão

As bases de cada análise são recortes da piloto-volta:
  decisao_parada()  piloto-volta em corrida seca; alvo `parou`
  duelos()          volta de duelo (atacante a até 1,0 s); alvo `passou`

Não usa internet. Roda tudo a partir de data/interim/ e dos arquivos da Etapa 1.
As bases não são versionadas: rodar `python src/bases.py` grava em data/bases/.

Regras herdadas da versão 1 (ver CHANGELOG):
  - "terminou" inclui "+N Laps" e "Lapped"
  - distância da corrida = voltas do vencedor, não a distância programada
  - composto em dureza relativa à escolha do fim de semana
  - chuva em três faixas pela fração do tempo
  - corrida sem disputa: mais de 90% do pelotão sem parar (Spa 2021)
  - corrida com parada incompleta: menos de 80% dos pilotos com as paradas
    conferidas contra os stints
  - SC/VSC medido no ponto de decisão; bandeira vermelha no ponto de decisão sai
"""
from pathlib import Path

import numpy as np
import pandas as pd

VOLTAS = Path("data/interim/voltas")
CORRIDAS = Path("data/interim/corridas")
SESSOES = Path("data/raw/sessoes_incluidas.csv")
CLUSTERS = Path("data/processed/clusters_circuitos.csv")
FEATURES = Path("data/processed/features_completas_por_circuito.csv")
DESTINO = Path("data/bases")

# A coluna certa da partição da Etapa 1 (cluster_final NÃO é a adotada).
COLUNA_CLUSTER = "cluster_kmeans_so_dinamicas_k4"
NOMES_PERFIL = {0: "alto downforce", 1: "stop-and-go", 2: "potência", 3: "urbano-técnico"}

SECOS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]
VERDE, VERMELHA = "1", "5"
NEUTRALIZACAO = {"4", "6", "7"}          # SC, VSC, VSC terminando
TERMINOU = r"^(Finished|Lapped|\+\d+ Laps?)$"

LIMIAR_CHUVA_PONTUAL = 0.15
MAX_SEM_PARADA = 0.90
MIN_CONFERIDOS = 0.80


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
def _arquivo(pasta, ano, rnd, tipo):
    return pasta / f"{ano}_{rnd:02d}_{tipo}.csv"


def _le_todos(tipo):
    partes = [pd.read_csv(a) for a in sorted(CORRIDAS.glob(f"*_{tipo}.csv"))]
    return pd.concat([p for p in partes if not p.empty], ignore_index=True)


def _le_voltas(ano, rnd):
    v = pd.read_csv(_arquivo(VOLTAS, ano, rnd, "voltas"), dtype={"bandeira": str})
    v = v.dropna(subset=["volta"]).copy()
    v["volta"] = v["volta"].astype(int)
    return v


def _faixa(x, cortes, rotulos):
    """Primeiro rótulo cujo corte é >= x. Valor vazio fica vazio."""
    if pd.isna(x):
        return np.nan
    for c, r in zip(cortes, rotulos):
        if x <= c:
            return r
    return rotulos[-1]


def dureza(compostos):
    """Composto seco -> (rótulo relativo, ordem dentro do fim de semana)."""
    p = [c for c in SECOS if c in compostos]
    if len(p) >= 3:
        r = ["macio"] + ["médio"] * (len(p) - 2) + ["duro"]
    elif len(p) == 2:
        r = ["macio", "duro"]
    else:
        r = ["médio"] * len(p)
    return {c: (rot, i) for i, (c, rot) in enumerate(zip(p, r))}


def fase(volta, total):
    if volta <= total / 3:
        return "primeiro terço"
    return "segundo terço" if volta <= 2 * total / 3 else "último terço"


def acha_paradas(v):
    """Uma linha por parada: entrou no box na volta L e saiu na volta L+1.

    Sem volta seguinte é quem abandonou nos boxes, não uma parada. `trocou_pneu`
    exige stint diferente na volta seguinte (stint vazio não conta).
    """
    v = v.sort_values(["piloto", "volta"])
    seg = v.groupby("piloto")
    v = v.assign(prox_volta=seg["volta"].shift(-1), prox_stint=seg["stint"].shift(-1),
                 prox_saida=seg["saida_box_s"].shift(-1))
    par = v[v["entrada_box_s"].notna() & (v["prox_volta"] == v["volta"] + 1)
            & v["prox_saida"].notna()]
    trocou = par["prox_stint"].notna() & par["stint"].notna() & (par["stint"] != par["prox_stint"])
    return pd.DataFrame({"piloto": par["piloto"].to_numpy(), "volta": par["volta"].to_numpy(),
                         "trocou_pneu": trocou.to_numpy()})


# ---------------------------------------------------------------------------
# Tabelas centrais
# ---------------------------------------------------------------------------
def piloto_corrida():
    """Uma linha por piloto por corrida (resultado e estratégia pelos stints)."""
    p = _le_todos("pilotos")
    s = _le_todos("stints")
    n_paradas = (s.groupby(["ano", "round", "piloto"]).size() - 1).rename("n_paradas")
    p = p.merge(n_paradas.reset_index(), on=["ano", "round", "piloto"], how="left")
    p["terminou"] = p["status"].str.match(TERMINOU, na=False)
    p["nao_largou"] = p["status"].isin(["Did not start", "Withdrew"])
    p["voltas_corrida"] = p.groupby(["ano", "round"])["voltas_completadas"].transform("max")
    return p


def corridas(pc=None):
    """Uma linha por corrida, com as marcas que decidem se ela entra nas análises."""
    pc = piloto_corrida() if pc is None else pc
    clima = _le_todos("clima")
    chuva = clima.groupby(["ano", "round"])["chuva"].apply(lambda c: c.astype(bool).mean())

    c = pc.groupby(["ano", "round"]).agg(
        voltas_corrida=("voltas_completadas", "max"),
        sem_parada=("n_paradas", lambda x: (x.fillna(0) == 0).mean()),
    ).reset_index()
    c["pct_tempo_chuva"] = c.set_index(["ano", "round"]).index.map(chuva)
    c["faixa_chuva"] = [
        "seco" if pd.isna(x) or x == 0 else
        "chuva pontual" if x < LIMIAR_CHUVA_PONTUAL else "chuva significativa"
        for x in c["pct_tempo_chuva"]]
    c["sem_disputa"] = c["sem_parada"] > MAX_SEM_PARADA

    sess = pd.read_csv(SESSOES)[["ano", "round", "circuito"]]
    clus = pd.read_csv(CLUSTERS)[["circuito", COLUNA_CLUSTER]]
    perda = pd.read_csv(FEATURES)[["circuito", "pit_lane_time_loss_s"]]
    c = c.merge(sess, on=["ano", "round"], how="left") \
         .merge(clus, on="circuito", how="left") \
         .merge(perda, on="circuito", how="left")
    c["perfil"] = c[COLUNA_CLUSTER].map(NOMES_PERFIL)
    c["era"] = np.where(c["ano"] >= 2022, "2022+", "2018-21")
    c["tem_voltas"] = [_arquivo(VOLTAS, a, r, "voltas").exists()
                       for a, r in zip(c["ano"], c["round"])]

    # Paradas conferidas: fração de pilotos cujas paradas medidas na volta a volta
    # batem com o número de stints menos 1.
    s = _le_todos("stints")
    esperadas = s.groupby(["ano", "round", "piloto"]).size() - 1
    conferidas = {}
    for a, r in c.loc[c["tem_voltas"], ["ano", "round"]].itertuples(index=False):
        v = _le_voltas(a, r).dropna(subset=["posicao"])
        par = acha_paradas(v)
        med = par[par["trocou_pneu"]].groupby("piloto").size()
        esp = esperadas.loc[(a, r)]
        conferidas[(a, r)] = (med.reindex(esp.index, fill_value=0) == esp).mean()
    c["paradas_conferidas"] = c.set_index(["ano", "round"]).index.map(conferidas)

    c["entra_secas"] = (c["tem_voltas"] & ~c["sem_disputa"] & (c["faixa_chuva"] == "seco")
                        & (c["paradas_conferidas"] >= MIN_CONFERIDOS))
    return c.drop(columns=[COLUNA_CLUSTER, "sem_parada"])


def _piloto_volta_corrida(ano, rnd, info):
    v = _le_voltas(ano, rnd).dropna(subset=["posicao"])
    par = acha_paradas(v)
    trocas = set(zip(par.loc[par["trocou_pneu"], "piloto"], par.loc[par["trocou_pneu"], "volta"]))
    rot = dureza(set(v["composto"].dropna()))

    b = pd.read_csv(_arquivo(CORRIDAS, ano, rnd, "bandeiras")).sort_values("tempo_s")
    avisos_t, avisos_cod = b["tempo_s"].to_numpy(), b["codigo"].astype(str).to_numpy()

    # Ordem e distâncias no fim de cada volta.
    v = v.sort_values(["volta", "posicao"])
    g = v.groupby("volta")
    v["frente"] = g["piloto"].shift(1)
    v["tras"] = g["piloto"].shift(-1)
    v["dif_pos_frente"] = v["posicao"] - g["posicao"].shift(1)
    v["dist_frente"] = v["fim_volta_s"] - g["fim_volta_s"].shift(1)
    v["dist_tras"] = g["fim_volta_s"].shift(-1) - v["fim_volta_s"]
    # Volta verde: 80% ou mais do pelotão com bandeira verde na volta.
    verde = g["bandeira"].agg(lambda x: (x.astype(str) == VERDE).mean() >= 0.8)
    v["volta_verde"] = v["volta"].map(verde)

    v["parou"] = [(p, l) in trocas for p, l in zip(v["piloto"], v["volta"])]
    v["box"] = v["entrada_box_s"].notna() | v["saida_box_s"].notna()
    v["saida_box"] = v["saida_box_s"].notna()
    v = v.sort_values(["piloto", "volta"])
    v["paradas_feitas"] = v.groupby("piloto")["parou"].cumsum() - v["parou"]

    # Ponto de decisão: entrada do box para quem parou, fim da volta para quem não parou.
    v["t_decisao"] = np.where(v["parou"] & v["entrada_box_s"].notna(),
                              v["entrada_box_s"], v["fim_volta_s"])
    i = np.searchsorted(avisos_t, v["t_decisao"].to_numpy(), side="right") - 1
    cod = avisos_cod[np.clip(i, 0, None)]
    v["bandeira_decisao"] = np.where(v["t_decisao"].isna(), VERDE, cod)

    v["composto_rel"] = v["composto"].map(lambda c: rot.get(c, (np.nan, np.nan))[0])
    v["ordem_composto"] = v["composto"].map(lambda c: rot.get(c, (np.nan, np.nan))[1])
    v["ultima_volta"] = int(v["volta"].max())
    for k, val in info.items():
        v[k] = val
    return v


def piloto_volta(c=None, so_secas=True):
    """Uma linha por piloto por volta. Por padrão, só as corridas que entram nas
    análises de corrida seca (`entra_secas`)."""
    c = corridas() if c is None else c
    sel = c[c["entra_secas"]] if so_secas else c[c["tem_voltas"]]
    partes = []
    for r in sel.itertuples(index=False):
        info = {"circuito": r.circuito, "perfil": r.perfil, "era": r.era,
                "voltas_corrida": r.voltas_corrida, "perda_box": r.pit_lane_time_loss_s}
        partes.append(_piloto_volta_corrida(r.ano, r.round, info))
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------
# Recortes por análise
# ---------------------------------------------------------------------------
def decisao_parada(pv):
    """Piloto em cada volta L. Alvo `parou`: entrou no box na volta L e trocou o pneu.

    Os descritores descrevem o fim da volta L-1, antes da decisão. A exceção é a
    neutralização, medida no ponto de decisão da volta L.
    Saem: volta 1, a última volta, volta de saída do box (L ou L-1), pneu de chuva,
    tempo de vida do pneu vazio e bandeira vermelha no ponto de decisão.
    """
    pv = pv.sort_values(["ano", "round", "piloto", "volta"])
    chave = ["ano", "round", "piloto"]
    ant = pv[chave + ["volta", "posicao", "frente", "tras", "dist_frente", "dist_tras",
                      "composto", "composto_rel", "vida_pneu", "saida_box", "parou",
                      "paradas_feitas"]].copy()
    ant["volta"] += 1
    ant = ant.rename(columns={c: c + "_ant" for c in ant.columns if c not in chave + ["volta"]})
    d = pv[chave + ["volta", "parou", "saida_box", "bandeira_decisao", "ultima_volta",
                    "voltas_corrida", "perda_box", "circuito", "perfil", "era"]] \
        .merge(ant, on=chave + ["volta"])
    d = d[(d["volta"] >= 2) & (d["volta"] < d["ultima_volta"]) & ~d["saida_box"]
          & ~d["saida_box_ant"] & d["composto_ant"].isin(SECOS)
          & (d["bandeira_decisao"] != VERMELHA) & d["vida_pneu_ant"].notna()]

    paradas = set(zip(pv.loc[pv["parou"], "ano"], pv.loc[pv["parou"], "round"],
                      pv.loc[pv["parou"], "piloto"], pv.loc[pv["parou"], "volta"]))

    def rival_parou(col):
        return [(a, r, p, l - 1) in paradas for a, r, p, l in
                zip(d["ano"], d["round"], d[col], d["volta"])]

    return pd.DataFrame({
        "ano": d["ano"], "round": d["round"], "circuito": d["circuito"],
        "piloto": d["piloto"], "volta": d["volta"], "parou": d["parou"].astype(int),
        "idade_pneu": d["vida_pneu_ant"],
        "d_idade_pneu": [_faixa(x, [9, 19, 29], ["0–9 voltas", "10–19 voltas",
                                                 "20–29 voltas", "30+ voltas"])
                         for x in d["vida_pneu_ant"]],
        "d_composto": d["composto_rel_ant"],
        "d_neutralizacao": np.where(d["bandeira_decisao"].isin(NEUTRALIZACAO),
                                    "safety car ou VSC", "bandeira verde"),
        "d_rival_frente": np.where(rival_parou("frente_ant"), "rival da frente parou",
                                   "rival da frente não parou"),
        "d_rival_tras": np.where(rival_parou("tras_ant"), "rival de trás parou",
                                 "rival de trás não parou"),
        "d_janela": np.where(d["dist_tras_ant"].fillna(np.inf) > d["perda_box"],
                             "cabe a parada", "não cabe a parada"),
        "d_dist_frente": [_faixa(x, [1, 3], ["até 1 s", "1 a 3 s", "mais de 3 s"])
                          if pd.notna(x) else "mais de 3 s" for x in d["dist_frente_ant"]],
        "d_paradas_feitas": [_faixa(x, [0, 1], ["0 paradas", "1 parada", "2+ paradas"])
                             for x in d["paradas_feitas_ant"] + d["parou_ant"]],
        "d_fase": [fase(l, t) for l, t in zip(d["volta"], d["voltas_corrida"])],
        "d_posicao": [_faixa(x, [5, 10], ["top 5", "6.º a 10.º", "11.º em diante"])
                      for x in d["posicao_ant"]],
        "d_perfil": d["perfil"], "d_era": d["era"],
    }).reset_index(drop=True)


def duelos(pv, pc, distancia=1.0, volta_minima=3, voltas_relargada=3):
    """Volta de duelo: ao fim da volta L-1 o atacante está logo atrás do defensor, a
    até `distancia` s; as voltas L-1 e L são verdes; nenhum dos dois passa pelo box
    nelas; os dois completam a volta L. Alvo `passou`: atacante à frente ao fim de L.
    """
    chave = ["ano", "round"]
    # Largada: quem não largou sai; largada do box (grid 0) conta como a última.
    pc = pc.copy()
    pc["n"] = pc.groupby(chave)["piloto"].transform("size")
    pc["grid_ef"] = np.where(pc["grid"] == 0, pc["n"] + 1, pc["grid"])
    pc.loc[pc["nao_largou"], "grid_ef"] = np.nan
    grid = pc.set_index(chave + ["piloto"])["grid_ef"]
    equipe = pc.set_index(chave + ["piloto"])["equipe"]

    # Voltas que não são verdes marcam neutralização, para a "relargada".
    neutras = pv.loc[~pv["volta_verde"].astype(bool), chave + ["volta"]].drop_duplicates()
    neutras = neutras.groupby(chave)["volta"].apply(set).to_dict()

    cols = ["posicao", "fim_volta_s", "box", "composto", "ordem_composto", "vida_pneu",
            "volta_verde"]
    idx = pv.set_index(chave + ["piloto", "volta"])[cols + ["dist_frente"]]
    # Atacante na volta L-1, com o defensor imediatamente à frente.
    a = pv[(pv["dif_pos_frente"] == 1) & (pv["dist_frente"] > 0)
           & (pv["dist_frente"] <= distancia)].copy()
    a = a.rename(columns={"piloto": "atacante", "frente": "defensor", "volta": "antes"})
    a["volta"] = a["antes"] + 1
    a = a[a["volta"] >= volta_minima]

    def junta(df, quem, volta, suf):
        j = idx.add_suffix(suf)
        return df.merge(j, left_on=chave + [quem, volta], right_index=True, how="inner")

    a = a[chave + ["circuito", "perfil", "era", "voltas_corrida", "atacante", "defensor",
                   "antes", "volta", "dist_frente"]].rename(columns={"dist_frente": "distancia_s"})
    a = junta(a, "atacante", "antes", "_a0")
    a = junta(a, "defensor", "antes", "_d0")
    a = junta(a, "atacante", "volta", "_a1")
    a = junta(a, "defensor", "volta", "_d1")
    a = a[a["volta_verde_a0"].astype(bool) & a["volta_verde_a1"].astype(bool)
          & ~(a["box_a0"] | a["box_d0"] | a["box_a1"] | a["box_d1"])
          & a["ordem_composto_a1"].notna() & a["ordem_composto_d1"].notna()
          & a["vida_pneu_a1"].notna() & a["vida_pneu_d1"].notna()]
    ga = grid.reindex(pd.MultiIndex.from_frame(a[chave + ["atacante"]])).to_numpy()
    gd = grid.reindex(pd.MultiIndex.from_frame(a[chave + ["defensor"]])).to_numpy()
    a = a.assign(grid_a=ga, grid_d=gd).dropna(subset=["grid_a", "grid_d"])
    ea = equipe.reindex(pd.MultiIndex.from_frame(a[chave + ["atacante"]])).to_numpy()
    ed = equipe.reindex(pd.MultiIndex.from_frame(a[chave + ["defensor"]])).to_numpy()

    dif = a["vida_pneu_d1"] - a["vida_pneu_a1"]
    trem = (a["dist_frente_d0"] > 0) & (a["dist_frente_d0"] <= distancia)
    relarg = [any(antes - voltas_relargada <= n < antes for n in neutras.get((an, r), ()))
              for an, r, antes in zip(a["ano"], a["round"], a["antes"])]
    oa, od = a["ordem_composto_a1"], a["ordem_composto_d1"]

    return pd.DataFrame({
        "ano": a["ano"], "round": a["round"], "circuito": a["circuito"], "volta": a["volta"],
        "defensor": a["defensor"], "atacante": a["atacante"],
        "passou": (a["posicao_a1"] < a["posicao_d1"]).astype(int),
        "posicao_disputada": a["posicao_d0"], "distancia_s": a["distancia_s"], "dif_idade": dif,
        "d_dif_idade": ["atacante 10+ voltas mais novo" if x >= 10 else
                        "atacante 5 a 9 voltas mais novo" if x >= 5 else
                        "diferença de até 4 voltas" if x > -5 else
                        "defensor 5+ voltas mais novo" for x in dif],
        "d_dif_composto": np.where(oa < od, "atacante mais macio",
                                   np.where(oa > od, "atacante mais duro", "mesmo composto")),
        "d_distancia": [_faixa(x, [0.3, 0.6], ["até 0,3 s", "0,3 a 0,6 s", "0,6 a 1,0 s"])
                        for x in a["distancia_s"]],
        "d_trem": np.where(trem, "defensor em trem de DRS", "defensor livre"),
        "d_largada": np.where(a["grid_a"] < a["grid_d"], "atacante largou à frente",
                              "defensor largou à frente"),
        "d_fase": [fase(l, t) for l, t in zip(a["volta"], a["voltas_corrida"])],
        "d_relargada": np.where(relarg, "após relargada", "corrida normal"),
        "d_posicao": [_faixa(x, [5, 10], ["top 5", "6.º a 10.º", "11.º em diante"])
                      for x in a["posicao_d0"]],
        "d_mesma_equipe": np.where(ea == ed, "mesma equipe", "equipes diferentes"),
        "d_perfil": a["perfil"], "d_era": a["era"],
    }).sort_values(["ano", "round", "volta", "posicao_disputada"]).reset_index(drop=True)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    pc = piloto_corrida()
    c = corridas(pc)
    pv = piloto_volta(c)
    bases = {"corridas": c, "piloto_corrida": pc, "piloto_volta": pv,
             "decisao_parada": decisao_parada(pv), "duelos": duelos(pv, pc)}
    for nome, df in bases.items():
        df.to_csv(DESTINO / f"{nome}.csv", index=False)
        print(f"{nome:16s} {len(df):>8,} linhas")
    print(f"\nCorridas com volta a volta: {int(c['tem_voltas'].sum())}; "
          f"entram nas análises de corrida seca: {int(c['entra_secas'].sum())}")
    print(f"Salvo em {DESTINO}/")


if __name__ == "__main__":
    main()
