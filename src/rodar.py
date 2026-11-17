"""
Etapa 2 - Roda uma análise a partir da ficha.

  python src/rodar.py analises/duelos.yaml                  busca + validação + perfil
  python src/rodar.py analises/duelos.yaml --sem-validacao  só a busca

Lê a ficha, monta a base com src/bases.py e grava em results/<nome>/:
  ficha.yaml          cópia da ficha usada
  subgrupos.csv       todos os subgrupos com suporte mínimo (e, com validação,
                      as colunas de elegibilidade, limiar e firmeza)
  finais.csv          firmes que ficam depois da redundância (só com validação)
  resumo.txt          o que foi impresso na tela
Sem validação, grava em results/<nome>/sem_validacao/.

Ordem dos sorteios (para reproduzir a versão 1): média por perfil, limiar da busca,
"o perfil acrescenta?".
"""
import argparse
import io
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import bases       # noqa: E402
import perfil      # noqa: E402
import validacao   # noqa: E402
from busca import busca   # noqa: E402

FMT = lambda x: f"{x:+.3f}"   # noqa: E731
COLS = ["descricao", "linhas", "circuitos", "media", "z"]


def carrega_base(nome):
    """Lê data/bases/<nome>.csv; se não existir, gera todas as bases."""
    arquivo = bases.DESTINO / f"{nome}.csv"
    if not arquivo.exists():
        bases.main()
    return pd.read_csv(arquivo)


def titulo(t):
    print("=" * 78 + f"\n{t}\n" + "=" * 78)


def roda(ficha, sem_validacao):
    f = ficha
    df = carrega_base(f["base"]).sort_values(["circuito", "ano", "round"], kind="stable") \
        .reset_index(drop=True)
    alvo, D = f["alvo"], f["descritores"]
    print(f"Análise: {f['nome']} — {f['pergunta']}")
    print(f"Base: {len(df):,} linhas, {df.groupby(['ano', 'round']).ngroups} corridas, "
          f"{df['circuito'].nunique()} circuitos. Média geral do alvo: {df[alvo].mean():.4f}\n")

    rng = np.random.default_rng(f["validacao"]["semente"])
    v = f["validacao"]
    if not sem_validacao:
        titulo("1. O ALVO POR PERFIL")
        p, tab, por_circ = perfil.media_por_perfil(df, alvo, rng, v["permutacoes_perfil"])
        print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\nO perfil muda o alvo? p = {p:.4f} (permutação entre "
              f"{df['circuito'].nunique()} circuitos)\n")

    titulo(f"2. A BUSCA ({len(D)} descritores, profundidade {f['profundidade']}, "
           f"suporte {f['suporte']})")
    t = time.time()
    res = busca(df, D, alvo, f["suporte"], f["profundidade"])
    print(f"Subgrupos com suporte mínimo: {len(res):,} ({time.time() - t:.1f} s)")
    res = res.sort_values("z", key=abs, ascending=False)
    if sem_validacao:
        print("\nMaiores |z|:")
        print(res.head(25)[COLS].to_string(index=False, float_format=FMT, max_colwidth=70))
        return res, None

    ok = validacao.elegivel(res, v["min_circuitos"], v["max_circuito"])
    t = time.time()
    lim, _ = validacao.limiar(df, D, alvo, res, rng, v["permutacoes"], f["suporte"],
                              v["min_circuitos"], v["max_circuito"])
    print(f"Elegíveis ({v['min_circuitos']}+ circuitos, nenhum acima de "
          f"{v['max_circuito']:.0%}): {int(ok.sum()):,}")
    print(f"Limiar |z| (5%, {v['permutacoes']} permutações): {lim:.2f} "
          f"({time.time() - t:.1f} s)")
    res["elegivel"] = ok
    fir = validacao.firmeza(df, alvo, res[ok], lim)
    res = res.join(fir)
    firmes = res[res["firme"] == True]   # noqa: E712
    print(f"Acima do limiar: {int(res['acima_do_limiar'].sum())}   firmes: {len(firmes)}"
          f"   firmes com perfil: {int(firmes['tem_perfil'].sum())}\n")
    print("Maiores |z| entre os elegíveis:")
    print(res[ok].head(20)[COLS + ["firme"]].to_string(index=False, float_format=FMT,
                                                       max_colwidth=70))

    titulo("3. O PERFIL ACRESCENTA?")
    com_perfil = firmes[firmes["tem_perfil"]]
    if len(com_perfil):
        ac, lim_p = perfil.perfil_acrescenta(df, alvo, com_perfil, rng, v["permutacoes"])
        res = res.join(ac)
        print(f"Testados: {len(ac)}. Limiar |z| corrigido ({v['permutacoes']} permutações): "
              f"{lim_p:.2f}")
        print(f"O perfil acrescenta em {int(ac['perfil_acrescenta'].sum())} de {len(ac)}.\n")
    else:
        print("Nenhum subgrupo firme com condição de perfil.\n")

    titulo("4. REDUNDÂNCIA")
    red = validacao.redundancia(df, alvo, firmes, v["min_ganho"], v["min_z"])
    res = res.join(red)
    finais = res[res["fica"] == True]   # noqa: E712
    print(f"Firmes: {len(firmes)}   ficam: {len(finais)}   saem: {len(firmes) - len(finais)}\n")
    media = df[alvo].mean()
    for nome, parte in [("Acima da média", finais[finais["media"] > media]),
                        ("Abaixo da média", finais[finais["media"] < media])]:
        print(f"{nome}: {len(parte)}")
        print(parte.head(15)[COLS + ["comparado_com"]].to_string(
            index=False, float_format=FMT, max_colwidth=60))
        print()
    return res, finais


def main():
    p = argparse.ArgumentParser(description="Roda uma análise da Etapa 2 a partir da ficha")
    p.add_argument("ficha", type=Path)
    p.add_argument("--sem-validacao", action="store_true")
    a = p.parse_args()
    ficha = yaml.safe_load(a.ficha.read_text(encoding="utf-8"))
    destino = Path("results") / ficha["nome"]
    if a.sem_validacao:
        destino = destino / "sem_validacao"
    destino.mkdir(parents=True, exist_ok=True)

    tela = io.StringIO()

    class Ambos(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            tela.write(s)
            return len(s)

    with redirect_stdout(Ambos()):
        res, finais = roda(ficha, a.sem_validacao)
    shutil.copy(a.ficha, destino / "ficha.yaml")
    res.to_csv(destino / "subgrupos.csv", index=False)
    if finais is not None:
        finais.to_csv(destino / "finais.csv", index=False)
    (destino / "resumo.txt").write_text(tela.getvalue(), encoding="utf-8")
    print(f"\nSalvo em {destino}/")


if __name__ == "__main__":
    main()
