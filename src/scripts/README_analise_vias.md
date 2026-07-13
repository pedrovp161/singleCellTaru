# Análise de vias: Estimulado vs Não estimulado por subgrupo

Relatório técnico do script [`teste.R`](teste.R).

---

## 1. Objetivo

Descobrir **quais vias biológicas mudam quando as células são estimuladas** (`estimulado` vs `nao estimulado`), fazendo essa comparação **separadamente dentro de cada subgrupo de classificação** dos pacientes: `controle`, `PB`, `MB` e `NP`.

O resultado final são tabelas (CSV) e gráficos de pontos (dot plots) que mostram, por subgrupo, quais vias estão ativadas (UP) ou reprimidas (DOWN) sob estímulo — usando três coleções de vias em paralelo: **Hallmark**, **Reactome** e **KEGG**.

---

## 2. Dados de entrada

| Item | Valor |
|------|-------|
| Objeto | `adata_harmony_subset_T_no8_.RDS` (objeto Seurat de single-cell / spatial) |
| Contagens cruas | assay `counts`, camada `data` |
| Coluna de subgrupo | `classificacao` (MB / PB / controle / NP) |
| Coluna de condição | `estimulo` (`nao estimulado` = referência, `estimulado`) |
| Coluna de amostra | `library_id` (1 paciente + 1 condição = 1 réplica pseudobulk) |

Um passo de preparação reclassifica como `NP` toda amostra cujo `library_id` começa com `HNP` (o nível `NP` é adicionado ao fator antes disso).

---

## 3. Decisões metodológicas centrais

### 3.1. Por que pseudobulk (e não teste célula-a-célula)?

Células do mesmo paciente **não são réplicas independentes** entre si. Um teste célula-a-célula (ex.: Wilcoxon) trata cada célula como uma amostra, o que **infla artificialmente o N** (pseudorreplicação) e gera p-valores minúsculos e falsos.

**Solução:** somar as contagens cruas de todas as células de cada `library_id` (pseudobulk). Assim, cada **paciente/amostra** vira uma réplica real, e o número de réplicas passa a refletir o desenho experimental de verdade.

### 3.2. Por que um único modelo DESeq2 combinado?

Alguns subgrupos (tipicamente **MB** e **NP**) têm só **1 paciente por condição**. Um DESeq2 rodado isoladamente dentro desse subgrupo não consegue estimar a dispersão com n=1 e seria descartado.

**Solução:** ajustar **um único modelo DESeq2 com TODAS as amostras juntas** (design `~ group_cond`, onde `group_cond = classificacao × estimulo`). A dispersão é estimada usando o conjunto completo (muito mais estável) e reaproveitada para testar cada subgrupo — inclusive os de 1 réplica.

> ⚠️ **Cautela:** isso impede que o grupo suma da análise, mas **não cria réplicas que não existem**. Contrastes com n=1 por condição continuam com baixo poder estatístico. O N de cada grupo é reportado (`n_estimulado`/`n_nao_estimulado`) para você interpretar com cuidado.

### 3.3. Duas análises complementares: fgsea e ORA

| | **fgsea** (GSEA) | **ORA** (Over-Representation) |
|---|---|---|
| O que usa | O **ranking completo** de todos os genes (estatística Wald do DESeq2) | Apenas os **genes DE** que passam um corte duro |
| Corte | Nenhum | `padj < 0.05` (definido por `padj_cutoff_ora`) |
| Teste | Enriquecimento ao longo da distribuição | Hipergeométrico (`fgsea::fora`) |
| UP vs DOWN | Pelo sinal do NES | Duas listas separadas de genes (up/down) desde a origem |
| Ponto forte | Não perde sinal fraco/difuso | Direção UP/DOWN naturalmente separada, mais interpretável |

As duas são complementares: o fgsea aproveita toda a distribuição do ranking; o ORA foca só nos genes claramente significativos.

> **Consequência prática:** um subgrupo com poucas réplicas (ex.: MB) pode **aparecer no fgsea e sumir no ORA**. O fgsea sempre tem um ranking; o ORA precisa de genes que passem `padj < 0.05`, e se nenhum passar, aquele subgrupo não gera nenhuma linha de ORA.

---

## 4. Fluxo do pipeline

```
RDS (Seurat)
   │
   ├─ prepara metadados (classificacao, estimulo, library_id; reclassifica HNP→NP)
   │
   ├─ carrega coleções de vias (Hallmark, Reactome, KEGG) via msigdbr
   │
   ├─ PSEUDOBULK: soma contagens cruas por library_id
   │
   ├─ filtro de genes de baixa expressão (soma total ≥ 10)
   │
   ├─ 1 MODELO DESeq2 combinado: design ~ group_cond
   │
   └─ para cada subgrupo (controle, PB, MB, NP):
         │
         ├─ extrai contraste estimulado vs não (do modelo combinado)
         │
         ├─ FGSEA  → roda nas 3 coleções, reaproveitando o mesmo ranking (stat)
         │
         └─ ORA    → define genes UP/DOWN (padj<0.05) e roda nas 3 coleções
                     │
                     └─ empilha tudo → fgsea_all / ora_all
                          │
                          ├─ salva CSVs (geral + 1 por coleção)
                          └─ dot plots (1 por coleção, fgsea em cima / ORA embaixo)
```

---

## 5. Detalhamento das etapas

### 5.1. Pseudobulk (`build_pseudobulk`)
Para um conjunto de células, soma (`Matrix::rowSums`) as contagens cruas agrupando por `library_id`. Resultado: matriz gene × amostra.

### 5.2. Modelo DESeq2
- `pb_counts_all`: pseudobulk de **todas** as células.
- `coldata_all`: uma linha por `library_id`, com `classificacao`, `estimulo` e a variável combinada `group_cond`.
- Filtro: gene mantido se `rowSums ≥ min_counts_gene` (10).
- `DESeqDataSetFromMatrix(..., design = ~ group_cond)` → `DESeq()`.

### 5.3. Contraste por subgrupo (`run_pathway_analysis_for_group`)
Para cada subgrupo, extrai o contraste `estimulado vs nao_estimulado` **daquele subgrupo** do modelo combinado, via `results(dds, contrast = c("group_cond", <grupo>__estimulado, <grupo>__nao_estimulado))`. Emite avisos se o grupo tiver menos réplicas que `min_samples_per_condition` (2).

**fgsea:** usa `res$stat` (Wald) como vetor de ranking ordenado, e roda `fgseaMultilevel` nas 3 coleções (`set.seed` fixo antes de cada uma, para reprodutibilidade).

**ORA:** define `genes_up` / `genes_down` por `padj < 0.05` e sinal do `log2FoldChange`; `universe` = todos os genes testados; roda `fora` nas 3 coleções para cada direção não vazia.

### 5.4. Agregação e saídas
- `fgsea_all` e `ora_all`: resultados de todos os subgrupos empilhados, com coluna `classificacao` e `geneset_collection`.
- Tabelas de diagnóstico no console (menor padj, nº de vias significativas, nº de réplicas, nº de genes DE).
- CSVs: um geral (`..._todas_colecoes.csv`) e um por coleção. `leadingEdge`/`overlapGenes` são colapsados em texto separado por `/`.

---

## 6. Como as vias são escolhidas para os dot plots

Isto é importante porque **não é** simplesmente "as N vias com menor padj no geral".

### fgsea (`make_fgsea_dot_plot`)
1. **Por subgrupo E por direção** (UP=NES>0, DOWN=NES<0): pega as `top_n_per_group_per_direction` (**8**) vias com menor padj.
   - *Motivo:* se pegasse só "as N com menor padj" no bolo todo, as vias UP (que tendem a ser mais significativas) dominariam e as DOWN sumiriam do gráfico.
2. **União** das vias de todos os grupos/direções (remove duplicatas).
3. **Corte final só se passar do teto:** se a união ultrapassar `top_n_pathways` (**30**), mantém as 30 com menor padj **mínimo** entre os grupos.

### ORA (`make_ora_dot_plot`)
Mesma lógica, mas a direção (up/down) já vem da própria estrutura do ORA (duas listas de genes).

> O padj é calculado **dentro de cada coleção separadamente** — os dot plots são por coleção, então a comparação de significância é sempre Hallmark-vs-Hallmark, Reactome-vs-Reactome etc., nunca misturando coleções.

**Codificação visual dos dot plots:**
- **Tamanho do ponto** = `-log10(padj)` (maior = mais significativo).
- **Cor (fgsea)** = NES (azul `#2166AC` = DOWN, vermelho `#B2182B` = UP).
- **Cor (ORA)** = direção (up/down).
- **Círculo cheio vs vazio** = padj < `padj_cutoff` (0.05) ou não.
- **Eixo x** inclui o N de réplicas (`n=nao/estim`) ou nº de genes DE.

---

## 7. Parâmetros ajustáveis principais

| Parâmetro | Valor | O que controla |
|-----------|-------|----------------|
| `group_levels` | controle, PB, MB, NP | Ordem/quais subgrupos |
| `cond_levels` | nao estimulado, estimulado | Condições (referência primeiro) |
| `min_samples_per_condition` | 2 | Limiar para **avisar** sobre poucas réplicas (não descarta mais) |
| `min_counts_gene` | 10 | Filtro de expressão mínima do gene |
| `padj_cutoff_ora` | 0.05 | Corte para gene ser "DE" no ORA |
| `lfc_cutoff_ora` | 0 | Corte de log2FC para UP/DOWN no ORA |
| `gene_set_collections` | H, C2:REACTOME, C2:KEGG | Coleções de vias (msigdbr) |
| `fgsea_minSize` / `maxSize` | 15 / 500 | Tamanho mínimo/máximo de via |
| `padj_cutoff` | 0.05 | Significância na visualização |
| `top_n_pathways` | 30 | Teto de vias no dot plot |
| `top_n_per_group_per_direction` | 8 | Vias UP e DOWN por grupo antes da união |
| `map_gene_ids` | FALSE | TRUE se rownames forem Ensembl (mapeia p/ símbolo) |

---

## 8. Arquivos gerados

| Arquivo | Conteúdo |
|---------|----------|
| `fgsea_estimulo_por_classificacao_todas_colecoes.csv` | fgsea, todas as coleções |
| `fgsea_estimulo_por_classificacao_<Coleção>.csv` | fgsea, uma coleção |
| `ora_estimulo_por_classificacao_todas_colecoes.csv` | ORA, todas as coleções |
| `ora_estimulo_por_classificacao_<Coleção>.csv` | ORA, uma coleção |
| `fgsea_estimulo_por_classificacao_<Coleção>_dotplot.png` | Dot plot fgsea |
| `ora_estimulo_por_classificacao_<Coleção>_dotplot.png` | Dot plot ORA |

Ao final, os gráficos fgsea e ORA de cada coleção são combinados (fgsea em cima, ORA embaixo) via `patchwork`.

---

## 9. Como rodar

O script já instala/carrega os pacotes necessários (Seurat, DESeq2, fgsea, msigdbr, tidyverse etc.). Basta ajustar os caminhos no topo (`libdir`, `data_path`, `obj_path`, `setwd`) para o seu ambiente e rodar:

```r
source("script.R")
```

Todos os parâmetros que normalmente precisam mudar estão concentrados no bloco **"PARÂMETROS AJUSTÁVEIS"** no início do arquivo.

---

## 10. Limitações e pontos de atenção

- **Poucas réplicas em MB/NP:** contrastes com n=1 por condição têm baixo poder. Grupos assim podem ter poucos ou nenhum gene DE — e portanto **sumir do ORA** (mas continuam no fgsea). Sempre confira `n_estimulado`/`n_nao_estimulado` e a mensagem de genes UP/DOWN no console.
- **Corte duro do ORA:** depende inteiramente de `padj < 0.05`; sinal biológico difuso pode não aparecer no ORA mesmo sendo real (aí o fgsea complementa).
- **Comparação entre coleções:** o padj é interno a cada coleção — não compare significância de Hallmark diretamente com Reactome/KEGG.