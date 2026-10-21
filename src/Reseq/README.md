# Reseq — avaliação do ressequenciamento

Avaliação preliminar das 14 bibliotecas de células T CD4⁺ (7 pacientes ×
estimulado ML / não estimulado NS) ressequenciadas com mais profundidade
(`data/allOuts`, Cell Ranger 10.1.0). O ressequenciamento foi feito para tentar
resolver a baixa detecção de citocinas. Esta análise também testa a hipótese de
que a baixa recuperação de células venha de excesso de doublets.

O trabalho vai **até a clusterização dos dados integrados**. Nenhum cluster é
anotado.

O guia da análise a partir da matriz raw (P01–P07) está em [`raw/README.md`](raw/README.md).

O resultado, com todos os números, está em
[`resultsReseq/RELATORIO_reseq.md`](../../resultsReseq/RELATORIO_reseq.md).

## Perguntas e onde cada uma é respondida

| pergunta | notebook |
| --- | --- |
| O que os web summaries e os demais arquivos do Cell Ranger mostram? | `comparativa/C01_websummary_metricas` |
| O que mudou entre a corrida antiga e o ressequenciamento, célula a célula? | `comparativa/C02_antigo_vs_novo` |
| Mais profundidade recuperaria citocinas? | `comparativa/C03_rarefacao_citocinas` |
| O pré-processamento original, aplicado aos dados novos, dá o quê? | `analise/A01_preprocessamento` |
| Falta algum pré-processamento? | `analise/A02_preprocessamento_adicional` |
| A baixa recuperação de células vem de doublets? | `analise/A03_doublets_recuperacao` |
| O processamento original, aplicado aos dados novos, dá o quê? E corrigido? | `analise/A04_processamento_integracao` |
| Os clusters descrevem estados celulares ou artefatos? Os problemas antigos se repetem? | `analise/A05_avaliacao_clusterizacao` |
| Outros detectores de doublets (Solo, cxds) mudam a conclusão? | `analise/A06_doublets_solo_cxds` |
| Partindo da matriz raw, como no sc-best-practices, dá para recuperar células? | `raw/P01`–`P06` |

## O que cada notebook faz

**C01 — web summary e métricas.** Lê o JSON embutido nos 14
`web_summary.html`, `metrics_summary.csv`, logs e a composição automática
(Azimuth). Confere proveniência e parâmetros, lista os alarmes do Cell Ranger,
mede o retorno da profundidade pelas curvas de saturação e a pureza do sorting.

**C02 — corrida antiga × nova.** Confirma que são as mesmas bibliotecas
(identidade de barcodes, sexo do doador). Compara chamada de células, UMIs e
genes por célula pareada, reprodutibilidade das contagens, genes que
"colapsaram" entre corridas e células positivas para citocinas.

**C03 — rarefação e citocinas.** A partir do `molecule_info.h5`: reads por
molécula, posição da corrida antiga na curva de rarefação, testes de artefato
(troca de índice, erros de UMI e de barcode), genes sustentados só por reads
isolados (com os genes do Y em doadoras como controle negativo) e, para cada
citocina, quantas células positivas apareceriam com menos ou mais reads.

**A01 — pré-processamento.** Réplica do `src/scripts/preprocessing.ipynb`:
filtro MAD combinado (k = 5) do spatools, flag de MT, Scrublet com
`expected_doublet_rate=0.06`. A réplica é validada antes nos dados antigos e
reproduz os números do notebook original nas 14 bibliotecas.

**A02 — pré-processamento adicional.** Avalia, sem alterar a linha principal,
células de baixo conteúdo, RNA ambiente, fração intrônica e barcodes de
material nuclear, contaminantes não-T, o universo de genes (interseção ×
união) e a composição dos genes variáveis. Fecha com uma recomendação por item.

**A03 — doublets e recuperação.** Quatro linhas de evidência independentes:
aritmética do carregamento, régua de RNA (mistura em log-UMI), contaminantes
B como segunda "espécie" e Scrublet com threshold automático e calibrado.

**A04 — processamento e integração.** Réplica do `src/scripts/processing.ipynb`
em três variantes sobre as mesmas células, mudando uma coisa por vez:

| variante | universo de genes | matriz da PCA |
| --- | --- | --- |
| V1 fiel | interseção entre bibliotecas | contagens cruas (como no original) |
| V2 | interseção | lognorm escalonada |
| V3 | união + `min_cells=3` global | lognorm escalonada |

Antes, a V1 é rodada sobre a corrida antiga e reproduz exatamente o
`data/adata_harmony.h5ad` (mesmos genes variáveis, componentes e clusters).
Inclui duas análises de sensibilidade: genes técnicos entre os genes variáveis
e Harmony por paciente em vez de por biblioteca.

**A05 — avaliação da clusterização.** Composição dos clusters por biblioteca e
paciente, mistura antes e depois do Harmony, relação com a profundidade,
estabilidade (sementes e resolução), QC por cluster, concordância com a corrida
antiga, citocinas por cluster e o veredito "repete / não repete" para cada
problema da análise anterior.

**A06 — Solo e cxds.** Dois detectores de doublets com lógicas diferentes do
Scrublet: o Solo (scvi-tools; scVI treinado em todas as células na GPU e um
classificador por biblioteca) e o cxds (pares de genes mutuamente exclusivos,
sem simulação; reimplementado em Python e conferido contra o pacote `scds` do
R). Avaliados contra os 70 doublets T + B validados em A03, contra células B
sozinhas (controle negativo), pelo conteúdo de RNA e pela taxa esperada do 10x.

## Pré-processamento a partir da matriz raw (`raw/`)

Segue os capítulos *raw data processing* e *quality control* do
[sc-best-practices](https://www.sc-best-practices.org/introduction/raw-data-processing/),
a partir de `raw_feature_bc_matrix.h5`.

| notebook | o que faz |
| --- | --- |
| `P01_gotas_vazias` | checklist do capítulo; knee plots; `emptyDrops` e `emptyDropsCellRanger` (DropletUtils); teste de "perfil de célula" com controles de profundidade para os barcodes extras |
| `P02_cellbender` | lê o CellBender (treinado antes por `raw/run_cellbender.py`); convergência, chamadas e RNA ambiente |
| `P03_qc_best_practices` | QC do capítulo (5 MADs nos dois lados, top-20, MT em logit + 8%) + filtro de material nuclear; piso de chamada; **valor único de 200 UMIs** (padrão adotado) |
| `P04_ambiente_e_doublets` | SoupX (ρ e contagens corrigidas) e scDblFinder (limiar calibrado pela taxa do 10x), por biblioteca |
| `P05_integracao` | filtro global de genes (≥ 20 células), V3 sobre as células aprovadas sem doublets; grava `reseq_raw_integrado__v2.h5ad` |
| `P06_balanco_final` | funil antigo × novo por biblioteca, alternativas, citocinas, veredito |
| `P07_integracao_clusterizacao` | avaliação do Harmony (por biblioteca) no objeto principal; varredura de resolução com 3 sementes; escala fina 0,8 (14 clusters, principal) e grossa 0,2; clusters técnicos; citocinas e estímulo por cluster; grava `clusters__v3.parquet` |

Padrão adotado (`config.QC_MIN_UMIS = 200`):
- **valor único:** 200 UMIs para chamada e QC em todas as bibliotecas, no
  lugar dos 5 MADs por biblioteca, com as demais regras do capítulo;
- **células do Cell Ranger** (≡ `emptyDropsCellRanger`) entram se passarem
  nessas regras;
- **barcodes fora do Cell Ranger** só entram com perfil de célula (teste de
  P01) e ficam marcados em `obs["baixa_profundidade"]`;
- **doublets removidos:** scDblFinder com limiar calibrado pela taxa do 10x,
  mais os T + B validados.

Resultado: 35.247 células. As de baixa profundidade formam um cluster
técnico; marcá-lo na anotação.

### Ambientes externos (fora do `.venv` e do `renv` do projeto)

| ambiente | caminho | uso |
| --- | --- | --- |
| R 4.5 + DropletUtils, scDblFinder, SoupX (bioconda) | `~/venvs/reseq-r` (`config.RESEQ_RSCRIPT`) | P01, P04, via `functions/droplets.R`, `soupx.R`, `scdblfinder.R` |
| CellBender 0.4.0 (Python, GPU) | `~/venvs/cellbender` (`config.CELLBENDER_BIN`) | `raw/run_cellbender.py` |
| R do sistema + scds | `~/R/reseq-scds` | conferência do cxds em A06 |

O R do sistema não tem gfortran, por isso o ambiente conda. As saídas brutas
de DropletUtils, CellBender, SoupX e scDblFinder ficam no NVMe, em
`/home/pedrovp/singleCellTaru_claude/reseq_raw/` (`config.HEAVY_OUT`).

Ordem: `python raw/run_cellbender.py` (~4–5 h na GPU; PB20_NS e HNP44_NS com
`--total-droplets 4000` e `2000`) → P01 → P02 → P03 → P04 → P05 → P06 → P07.

### spatools

As correções dos plots e do pré-processamento do spatools estão no branch
`reseq-qc-melhorias` (worktree em `~/spatools-reseq`, commit local, sem push e
sem instalar no `.venv`). Inclui a pipeline `BEST_PRACTICES`, os limiares
gravados em `uns`, a ordem e a classificação corrigidas no `outlier_quality`,
`barcode_rank`, `qc_scatter` e `tests/test_qc.py` (9/9). As versões usadas
aqui estão em `functions/qcplots.py` e `functions/bpqc.py`.

## Estrutura

```text
src/Reseq/
├── README.md
├── build_notebook.sh       # .py (jupytext) → .ipynb executado
├── functions/              # código compartilhado
│   ├── config.py           # caminhos, mapa de nomes, desenho, painéis de genes, parâmetros
│   ├── io.py               # leitores (web summary, 10x, molecule_info) e escrita versionada
│   ├── compare.py          # pareamento antiga × nova
│   ├── rarefaction.py      # rarefação analítica a partir de molecule_info.h5
│   ├── qc.py               # réplica do QC, RNA ambiente
│   ├── doublets.py         # testes da hipótese de doublets; cxds
│   ├── droplets.py         # DropletUtils, CellBender, SoupX, scDblFinder; teste de perfil de célula
│   ├── droplets.R, soupx.R, scdblfinder.R
│   ├── bpqc.py             # QC do sc-best-practices
│   ├── qcplots.py          # plots de QC (versões corrigidas dos do spatools)
│   ├── cxds.R              # cxds do pacote scds, só para conferência
│   ├── pipeline.py         # variantes V1–V3, seleção de HVG, métricas de avaliação
│   ├── plotting.py, fmt.py # estilo das figuras e números em português
│   ├── issues.py           # registro de problemas por notebook
│   └── legacy.py           # acesso às funções de src/scripts/claude/functions
├── comparativa/            # C01–C03 (.py + .ipynb executado)
├── analise/                # A01–A06 (.py + .ipynb executado)
└── raw/                    # P01–P06 + run_cellbender.py

dataReseq/                  # objetos (parquet, h5ad) + MANIFEST.jsonl — fora do git
resultsReseq/
├── RELATORIO_reseq.md      # o resumo dos problemas encontrados
├── figures/<notebook>/     # PNG + PDF
└── tables/<notebook>/      # CSV; cada figura tem a sua tabela fig_<nome>.csv,
                            # cada notebook tem o seu issues.csv
```

### Objetos principais em `dataReseq/`

| arquivo | conteúdo |
| --- | --- |
| `A01_preprocessamento/new/<biblioteca>.h5ad` | as 14 bibliotecas após QC e Scrublet, no mesmo formato de `data/new` |
| `A01_preprocessamento/celulas_qc.parquet` | todas as células do Cell Ranger, com o motivo de saída de cada uma |
| `A02_preprocessamento_adicional/flags_adicionais.parquet` | baixo conteúdo, fração intrônica, rótulo automático, por célula |
| `A03_doublets_recuperacao/flags_doublets.parquet` | sinal de célula B, conjugado T–B, Scrublet calibrado, por célula |
| `A04_processamento_integracao/reseq_integrado.h5ad` | **o objeto integrado** (ver abaixo) |
| `A04_processamento_integracao/antiga_variantes.parquet` | a corrida antiga processada pelas três variantes (clusters por célula) |
| `C03_rarefacao_citocinas/mol_*.parquet` | resumos do `molecule_info.h5` (a passada leva ~6 min) |
| `A06_doublets_solo_cxds/flags_solo_cxds*.parquet` | probabilidade do Solo, score do cxds, chamadas e consenso, por célula |
| `P03_qc_best_practices/qc_candidatos*.parquet` | todo barcode com ≥ 100 UMIs ou chamado: chamadas, métricas, regras; `passa` = padrão de 500 UMIs |
| `P07_integracao_clusterizacao/clusters__v3.parquet` | rótulos do Leiden do objeto principal: `leiden_final` (0,8), `leiden_grossa` (0,2), `leiden_r*`, `cluster_tecnico`; chave `cell_key` |
| `P05_integracao/reseq_raw_integrado__v2.h5ad` | **o objeto do pré-processamento novo**: 35.247 células (padrão de 200 UMIs, sem doublets), contagens SoupX e brutas, V3; a versão sem sufixo é a anterior (500 UMIs) |

O objeto integrado tem 35.064 células e o universo completo de genes. `X` é a
lognorm e `layers["counts"]` as contagens, conferidas entrada por entrada
contra as matrizes do Cell Ranger. Cada variante entra com sufixo:
`obs["leiden_V1"]`, `obsm["X_umap_V2"]`, `obsp["connectivities_V3"]`,
`var["highly_variable_V3"]`. As chaves sem sufixo (`leiden`, `X_umap`,
`X_emb_harmony`, `neighbors`, `highly_variable`) apontam para a V3, a linha de
base proposta. As flags de A02 e A03 ficam fora do objeto, como deltas; a
chave de junção é `obs["cell_key"]` = `library_id|barcode`.

```python
import anndata as ad, scanpy as sc
adata = ad.read_h5ad("dataReseq/A04_processamento_integracao/reseq_integrado.h5ad")
sc.pl.umap(adata, color="leiden")                       # V3
sc.tl.leiden(adata, resolution=0.8, neighbors_key="neighbors_V3", key_added="leiden_0_8")
```

## Como executar

```bash
cd src/Reseq
./build_notebook.sh comparativa/C01_websummary_metricas      # [timeout em s, padrão 3600]
```

A ordem importa, porque cada etapa lê o que a anterior gravou:
C01 → C02 → C03 → A01 → A02 → A03 → A04 → A05 → A06. A cadeia inteira leva
perto de uma hora e meia. Ambiente: `/mnt/SATA/singleCellTaru/.venv` (scanpy
1.11, anndata 0.12, harmonypy 2.0, spatools 0.38, scvi-tools 1.4 com GPU).
O A06 confere o cxds contra o `scds` 2.0.0, instalado numa biblioteca R
isolada em `~/R/reseq-scds` (fora do `renv` do projeto); sem ela, o notebook
roda só com a reimplementação.

* `RESEQ_OUT=<pasta>` grava em `<pasta>/dataReseq` e `<pasta>/resultsReseq`
  (modo de teste; nada vai para o projeto).
* `RESEQ_RECOMPUTE=1` força o C03 a refazer a passada pelo `molecule_info.h5`.

## Regras de escrita

* Nada em `data/` é escrito ou apagado, inclusive `data/allOuts`. O código
  recusa gravar fora de `dataReseq/` e `resultsReseq/`.
* Objetos em `dataReseq/` nunca são substituídos: conteúdo igual não é
  regravado; conteúdo diferente vira `nome__v2`, `nome__v3`… Todo arquivo
  gravado entra no `dataReseq/MANIFEST.jsonl` (append-only), com assinatura,
  notebook de origem e versões dos pacotes. Cada h5ad é relido e conferido
  logo depois de gravado.
* Figuras e tabelas em `resultsReseq/` são regeneradas a cada execução.

## Convenções

* Nomes de biblioteca canônicos, iguais aos da corrida antiga. As pastas
  `MB_34_SN` e `PB_20_SN` são **não estimuladas** (o `endswith("NS")` do código
  original as rotularia como estimuladas); o mapa está em
  `config.RESEQ_TO_CANONICAL`.
* `obs["classificacao"]` preserva a regra original (HNP44 → "MB");
  `obs["classification"]` é a corrigida (HNP44 → "NP").
* Toda figura e tabela declara o número de pacientes e de bibliotecas. MB e NP
  têm um paciente cada: são descritos, não testados. O único contraste testado
  é o pareado estimulado × não estimulado (n = 7).
* A corrida antiga é lida de `data/raw/<biblioteca>/` (mtx).
  `data/raw_h5ad/CTH94_NS.h5ad` está corrompido e não é usado.

## Desvios em relação aos notebooks originais

* Símbolos de genes vêm da tabela de features do Cell Ranger, não do `mygene`
  (rede). Não muda genes variáveis, PCA nem clusters.
* A seleção de genes variáveis (`seurat_v3` por biblioteca) é a do scanpy,
  reimplementada linha a linha em `pipeline.select_hvg` para que o loess de um
  lote numericamente singular seja refeito com span maior em vez de abortar.
  Isso só acontece no universo completo; na interseção o resultado é idêntico
  ao do scanpy, e o notebook confere.
* Não são reproduzidos o bug da célula 36 do `preprocessing.ipynb` (que
  sobrescreveu `data/filtered_h5ad`) nem o `cache=True` da leitura.
