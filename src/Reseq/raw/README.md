# Pré-processamento a partir da matriz raw (P01–P07)

Este guia explica, em ordem, o que foi feito nos notebooks `P01`–`P07`, por que
cada decisão foi tomada e como usar o resultado. Os números vêm das tabelas em
`resultsReseq/tables/P0*`; o relatório completo está em
`resultsReseq/RELATORIO_reseq.md`, seção 4.

## A pergunta

A análise anterior (`analise/A01`–`A06`) partiu da matriz **filtrada** do Cell
Ranger, ou seja, das células que ele já tinha escolhido. A pergunta aqui foi
outra: partindo da matriz **raw** (todos os barcodes, inclusive as gotas
vazias) e seguindo o
[sc-best-practices](https://www.sc-best-practices.org/introduction/raw-data-processing/),
dá para recuperar células que o Cell Ranger deixou de fora? E qual seria um
critério único, o mesmo para as 14 bibliotecas?

## Resposta curta

- **A matriz raw quase não esconde células com conteúdo útil.** O que o
  Cell Ranger deixou de fora é, na grande maioria, gota vazia ou fragmento de
  célula morta.
- **O que muda o número final é o controle de qualidade.** O critério
  adotado é um **mínimo de 200 UMIs em todas as bibliotecas**. Fora do Cell
  Ranger, só entram barcodes que tenham perfil de célula. Os doublets saem por
  um critério calibrado pela taxa física de multipletos.
- **Resultado: 35.247 células** (o pré-processamento antigo dava 35.064).
  HNP44_NS passa de 70 para 111 células.
- **Integração e clusterização:** Harmony por biblioteca e Leiden na resolução
  0,8, que dá 14 clusters. Quatro deles são técnicos e devem sair antes da
  anotação.

## O caminho, passo a passo

```text
raw_feature_bc_matrix.h5 (≈1–2 milhões de barcodes por biblioteca)
   │
   ├─ P01  emptyDrops (DropletUtils) ─┐
   ├─ P02  CellBender (GPU) ──────────┼─ quem é célula? → candidatos
   │                                  │
   ├─ P03  QC do best practices + valor único de 200 UMIs → 36.979 células
   ├─ P04  SoupX (RNA ambiente) + scDblFinder (doublets)  → 35.247 células
   ├─ P05  filtro de genes + PCA + Harmony + UMAP          → objeto principal
   ├─ P06  balanço: antigo × novo, por biblioteca
   └─ P07  avaliação da integração + escolha da resolução + clusters técnicos
```

### P01 — Gotas vazias com o emptyDrops

**O que é.** Cada barcode é uma gota. A maioria tem só o "caldo" de RNA solto
na suspensão (RNA ambiente), e poucas têm uma célula. O `emptyDrops` compara
cada barcode com o perfil desse caldo e chama de célula quem é diferente dele.

**O que foi feito.**
- `emptyDropsCellRanger`, a versão que imita o Cell Ranger, reproduziu a
  chamada dele (37.783 de 37.898 células). Isso confirmou a montagem.
- O `emptyDrops` clássico, sem o piso de 500 UMIs que o Cell Ranger aplica,
  acrescentou **38.005 barcodes**.

**O que se encontrou.** Esses barcodes extras têm, em mediana, 153 UMIs (as
células têm ~9 mil). Metade tem perfil de núcleo de célula morta (muita leitura
intrônica): 21 mil vêm só de PB21_NS.

**O teste de "perfil de célula".** Como saber se um barcode com 150 UMIs é uma
célula pequena ou só caldo? O teste mede, molécula a molécula, se o conteúdo
se parece mais com as células da biblioteca ou com as gotas vazias (log-razão
de verossimilhança por UMI). Ele usa dois controles na mesma profundidade:
- **positivo:** células reais, subamostradas até ~150 UMIs. Elas continuam
  reconhecíveis;
- **negativo:** gotas vazias somadas até a mesma profundidade.

O corte deixa passar só 1% das gotas vazias. Por esse critério, ~2.300 dos
38.005 extras têm perfil de célula, quase todos com menos de 500 UMIs. Em
HNP44_NS, os 732 extras se comportam como gotas vazias e ainda carregam genes
do cromossomo Y numa doadora.

### P02 — CellBender

Segunda opinião, em Python e na GPU. O CellBender ajusta um modelo da matriz
inteira, que separa célula de gota vazia e estima o RNA ambiente.
- Concordou com o Cell Ranger.
- Como o emptyDrops, tomou o material nuclear de PB21_NS por célula.
- Estimou de 0,9% a 1,9% de RNA ambiente dentro das células.
- PB20_NS e HNP44_NS precisaram de menos gotas analisadas para o treino não
  abortar.

### P03 — Controle de qualidade e o valor único

**As regras do capítulo de QC**, por biblioteca:
- UMIs, genes e % de UMIs nos 20 genes mais expressos: 5 desvios absolutos
  da mediana (MAD) para cima e para baixo;
- fração mitocondrial: 3 MADs acima em escala logit **e** acima de 8%;
- mais uma regra nossa: tirar barcodes com perfil de núcleo (fração intrônica
  alta).

**O achado que mudou a abordagem.** O limite inferior de 5 MADs varia demais
entre bibliotecas: de ~150 UMIs (CTH92_ML, CTH97_ML) até ~3.800 (CTH94_NS).
Nas bibliotecas profundas, ele descartava células razoáveis. Por isso foi
testado um **valor único**, o mesmo como chamada e como mínimo do QC:

| mínimo de UMIs | células | extras ao Cell Ranger | extras com perfil de célula |
| --- | --- | --- | --- |
| 200 | 39.544 | 3.354 | 789 (24%) |
| 300 | 37.936 | 1.746 | 471 |
| 500 | 36.940 | 750 | 17 |
| 1.000 | 35.514 | 499 | 8 |
| 5 MADs por biblioteca | 35.671 | 743 | 227 |

**O padrão adotado (200 UMIs):**
- **células do Cell Ranger** entram se passarem nas demais regras;
- **barcodes fora do Cell Ranger** entram só se tiverem perfil de célula. São
  789, marcados como `baixa_profundidade` (mediana de 327 UMIs e 253 genes);
- **o teste de perfil não é aplicado às células do Cell Ranger.** Ele foi
  calibrado para barcodes de ~150 UMIs e compara com o perfil médio. Aplicado a
  células profundas, reprovaria ~8.500 células reais, inclusive as ativadas,
  que são até mais positivas para citocina.

> **Armadilha medida aqui.** Recalcular os limiares do capítulo sobre todos os
> candidatos (inclusive as gotas vazias) puxa o limite inferior para baixo.
> Em HNP44_NS, 723 barcodes "passariam". Com piso de 100 UMIs, ~8.000 células
> reais sairiam como outliers superiores.

### P04 — RNA ambiente e doublets

- **SoupX (RNA ambiente):** a contaminação estimada fica no mínimo da busca
  (1%) na maioria das bibliotecas, ou seja, ≤ 1%. A correção é aplicada às
  contagens e mexe sobretudo nos eventos de 1 UMI de citocina de CTH97_ML.
- **scDblFinder (doublets):** com o limiar padrão, marcaria 7,6% das
  células, contra ~4,5% esperados pela física do carregamento. Por isso o
  limiar foi **calibrado pela taxa do 10x**, biblioteca a biblioteca.
- **Removidos 1.732 doublets:** os calibrados, mais os doublets T + B
  validados em A03. PB21_ML, a biblioteca mais carregada, perde 950. Os
  métodos não separam um doublet de uma célula grande; o calibrado remove só o
  que o carregamento permite.

### P05 — Integração

- Filtro de genes do capítulo: gene em ≥ 20 células **no conjunto**, não por
  biblioteca. Assim as citocinas não somem, como acontecia com o `inner join`
  antigo. Ficam 26.053 genes e 14 de 16 citocinas.
- Processamento V3 (A04): 2.000 genes variáveis por biblioteca, PCA sobre a
  lognorm escalonada, **Harmony por biblioteca**, vizinhança em 15 dimensões,
  UMAP.

### P06 — Balanço

| biblioteca | antigo | novo |
| --- | --- | --- |
| CTH92_NS | 809 | 860 |
| CTH92_ML | 1.531 | 1.569 |
| CTH94_NS | 5.576 | 5.669 |
| CTH94_ML | 499 | 535 |
| CTH97_NS | 1.513 | 1.545 |
| CTH97_ML | 4.696 | 4.658 |
| PB20_NS | 592 | 626 |
| PB20_ML | 777 | 841 |
| PB21_NS | 3.665 | 3.790 |
| PB21_ML | 10.629 | 9.929 |
| MB34_NS | 1.917 | 2.114 |
| MB34_ML | 920 | 992 |
| HNP44_NS | 70 | 111 |
| HNP44_ML | 1.870 | 2.008 |
| **total** | **35.064** | **35.247** |

De onde vem a diferença no total:
- **+1.262** células do Cell Ranger que o MAD descartava;
- **+789** de baixa profundidade;
- **−1.732** doublets.

As bibliotecas pequenas ganham de 2% a 10%.

### P07 — Avaliação da integração e clusterização

**O Harmony mistura bem.**
- Os vizinhos da mesma biblioteca caem de 41% para 11%.
- O número efetivo de pacientes na vizinhança vai de 1,4 para 2,9 (o máximo
  possível é 4,3).
- Ele também aproxima as duas bibliotecas de cada paciente (0,54 → 0,22),
  porque estímulo e canal estão confundidos: cada biblioteca é um canal e uma
  condição. **O efeito do estímulo deve ser testado nas contagens**
  (pseudobulk pareado por paciente), não pela presença de clusters.

**Resolução.** O Leiden foi rodado em 8 resoluções com 3 sementes cada. A
resolução 0,5 do pipeline original é das menos estáveis entre sementes (ARI
0,54). Ficam duas escalas:

| escala | resolução | clusters | ARI entre sementes |
| --- | --- | --- | --- |
| **fina (principal)** | 0,8 | 14 | 0,62 |
| grossa | 0,2 | 6 | 0,76 |

**Clusters técnicos na escala fina**, a tirar antes da anotação (3.064
células):

| cluster | células | motivo |
| --- | --- | --- |
| 6 | 2.444 | baixo conteúdo de RNA (mediana de 1.099 UMIs); é onde caem as células de baixa profundidade |
| 12 | 232 | baixo conteúdo de RNA (1.272 UMIs); instável |
| 11 | 295 | não-T (marcadores de células B) |
| 13 | 93 | não-T |

Sem eles, a profundidade explica 22% da variância de UMIs dentro de cada
biblioteca (no pipeline original, com a PCA sobre contagens cruas, eram 57%).
O cluster 7 tem mais células nas bibliotecas estimuladas nos 7 pacientes e é
47% de CTH97. É uma hipótese para a anotação, não um resultado (p exato 0,016;
0,11 depois da correção).

## Como usar o resultado

```python
import anndata as ad
import pandas as pd

adata = ad.read_h5ad("dataReseq/P05_integracao/reseq_raw_integrado__v2.h5ad")
rot = pd.read_parquet("dataReseq/P07_integracao_clusterizacao/clusters__v3.parquet")
adata.obs = adata.obs.join(rot.drop(columns="library_id").set_index("cell_key"), on="cell_key")

limpo = adata[~adata.obs["cluster_tecnico"].astype(bool)].copy()   # sem os clusters técnicos
```

O que tem no objeto principal:

| onde | o quê |
| --- | --- |
| `layers["counts"]` | contagens corrigidas pelo SoupX (inteiras) |
| `layers["counts_raw"]` | contagens brutas do Cell Ranger |
| `X` | lognorm (normalize_total 10⁴ + log1p) das contagens SoupX |
| `obsm["X_pca"]`, `obsm["X_emb_harmony"]`, `obsm["X_umap"]` | PCA, Harmony, UMAP |
| `obs["baixa_profundidade"]` | os 789 barcodes de 200–500 UMIs que entraram pelo teste de perfil |
| `obs["so_no_padrao"]` | células do Cell Ranger que o MAD por biblioteca descartaria |
| `obs["cellranger"]`, `obs["emptydrops"]`, `obs["cellbender"]` | quem chamou cada barcode |
| `obs["classification"]`, `obs["stimulation"]`, `obs["patient"]` | desenho (corrigido; HNP44 = NP) |

E nos rótulos (`clusters__v3.parquet`):

| coluna | o quê |
| --- | --- |
| `leiden_final` | resolução 0,8 (principal) |
| `leiden_grossa` | resolução 0,2 |
| `leiden_r*` | todas as resoluções testadas |
| `cluster_tecnico` | marca dos clusters 6, 11, 12 e 13 |

**Cuidados na análise:**
- para DE e citocinas, use `layers["counts"]`;
- se o ruído das células pobres pesar, exclua `baixa_profundidade`;
- trate citocinas sempre por biblioteca: CTH97_ML concentra boa parte do
  sinal;
- MB e NP têm um paciente cada.

## Como refazer

```bash
cd src/Reseq
python raw/run_cellbender.py                        # ~4–5 h de GPU, 14 bibliotecas
python raw/run_cellbender.py PB20_NS --total-droplets 4000
python raw/run_cellbender.py HNP44_NS --total-droplets 2000
for nb in P01_gotas_vazias P02_cellbender P03_qc_best_practices P04_ambiente_e_doublets \
          P05_integracao P06_balanco_final P07_integracao_clusterizacao; do
  ./build_notebook.sh raw/$nb 10800
done
```

Ambientes usados, todos fora do `.venv` e do `renv` do projeto:

| ambiente | para quê |
| --- | --- |
| `~/venvs/reseq-r` | R 4.5 via conda: DropletUtils, SoupX, scDblFinder. O R do sistema não tem gfortran |
| `~/venvs/cellbender` | CellBender 0.4.0 |

As saídas brutas dessas ferramentas (≈14 GB) ficam no NVMe, em
`/home/pedrovp/singleCellTaru_claude/reseq_raw/`. O parâmetro central está em
`functions/config.py` (`QC_MIN_UMIS = 200`).

## Limites

- Não há FASTQ nem BAM. As etapas anteriores à matriz (alinhamento, correção
  de barcode, UMIs) só podem ser auditadas pelo que o Cell Ranger registrou
  (`comparativa/C01`, `C03`).
- O teste de perfil de célula tem sensibilidade limitada abaixo de ~300 UMIs.
  Algumas células reais muito pobres ficaram de fora, mas nenhuma delas teria
  informação suficiente para anotação.
- A clusterização da massa central de células T é um recorte de um contínuo.
  Só 4 dos 14 clusters reaparecem quase iguais em todas as sementes.
- Nada foi anotado.
