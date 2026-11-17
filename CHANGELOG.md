# Changelog

Registro cronológico das decisões e marcos do projeto.

---

## Etapa 1 — Caracterização dos circuitos

### Concluído

- Definição das 9 features do grão circuito (dinâmicas, manuais, externas)
- Escolha de sessões de classificação como fonte do fingerprint
- Exclusão de 2022 por indisponibilidade de `session_info` na API
- Substituição de `nanmin`/`nanmax` por `nanpercentile(1/99)` para robustez ao ruído
- Substituição de carga lateral (G indisponível na API) por proxy de velocidade mínima de curva
- Pipeline resiliente com cache FastF1 e registro de falhas
- 26 circuitos canônicos × 10 features, zero nulos

- Features manuais e externas integradas (pit lane, altitude, Pirelli); EDA formal (`01_eda_features.ipynb`)
- Clusterização (`02_clusterizacao_circuitos.ipynb`): KMeans, `so_dinamicas` (5 features de telemetria), k = 4; silhouette 0,270, Davies-Bouldin 1,000, ARI bootstrap 0,818
- Partição adotada: coluna `cluster_kmeans_so_dinamicas_k4` de `clusters_circuitos.csv` (11/8/5/2). A coluna `cluster_final` não é a partição adotada

### Pendente

- Conferir o sinal do heatmap de centroides do cluster de potência (o pico de desaceleração é negativo nos dados brutos)

---

## Etapa 2 — Descoberta de subgrupos nas corridas (versão 2)

Este repositório recomeça a Etapa 2 a partir das coletas já feitas. A primeira
versão da Etapa 2 (bases, buscas, hipóteses H1–H11, validações e fichas) segue
preservada no repositório original, `LivCastr0/monografia_f1`.

### Coletas trazidas da versão 1

- Coleta de corridas (`coleta_corridas.py`): pilotos, stints, bandeiras e clima por corrida em `data/interim/corridas/`; falhas em `data/interim/falhas_coleta_corridas.csv`
- Coleta volta a volta (`coleta_voltas.py`): `data/interim/voltas/`, 163 de 164 corridas de 2018–2025 (Monza 2018 sem dados de tempo); 2022 incluído
- Teste do script de coleta (`testa_coleta_corrida.py`)
- Contagem manual de zonas de DRS por circuito (`data/external/zonas_drs_por_circuito.csv`)

### Bases (`src/bases.py`)

- Tudo sai de duas tabelas centrais, `corridas` e `piloto_volta`; as bases das análises são recortes delas
- 163 corridas com volta a volta; 122 entram nas análises de corrida seca (secas, com disputa e com 80% ou mais das paradas conferidas)
- Conferência contra a versão 1 (`src/confere_bases.py`):
  - duelos: as mesmas 19.475 linhas, todos os descritores iguais
  - decisão de parar: 120.989 linhas contra 121.075; todos os descritores iguais nas linhas comuns
- Correção em relação à versão 1: na decisão de parar, 86 voltas sem idade do pneu (Baku 2018 e Spielberg 2022, uma delas com parada) caíam na faixa "30+ voltas" porque a função de faixas não tratava valor vazio. Agora saem da base

### Pendente

- Motor de busca, validação e camada de perfil
