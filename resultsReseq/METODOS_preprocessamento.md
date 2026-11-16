# Métodos — pré-processamento dos dados de scRNA-seq ressequenciados

> Texto no formato da seção de Métodos de um artigo. Os trechos entre
> colchetes (**[preencher]**) dependem de informações de bancada que não estão
> nos dados processados. Os números vêm das tabelas em `resultsReseq/tables/`
> (P01–P08), e o código está em `src/Reseq/raw/`. O esquema geral está na
> Figura 1 (`resultsReseq/figures/P06_balanco_final/esquema_preprocessamento.png`).

## Desenho do estudo e geração das bibliotecas

Foram analisadas 14 bibliotecas de scRNA-seq de células T CD4⁺ de sangue
periférico de sete indivíduos: três controles (CTH92, CTH94, CTH97), dois
pacientes paucibacilares (PB20, PB21), um multibacilar (MB34) e um contato
domiciliar (HNP44). De cada indivíduo foram geradas duas bibliotecas: células
não estimuladas (NS) e células estimuladas com **[preencher: antígeno de
*M. leprae*, concentração e tempo]** (ML). As células CD4⁺ foram isoladas por
**[preencher: método de separação]**. As bibliotecas foram preparadas com o
Chromium Single Cell 3′ v3 (10x Genomics) e ressequenciadas em
**[preencher: plataforma e configuração de leitura]**. Como os grupos MB e NP
têm um único indivíduo cada, as comparações entre classes clínicas são
descritivas. As comparações com inferência se restringem ao efeito pareado do
estímulo dentro do indivíduo.

## Alinhamento e quantificação

As leituras foram alinhadas e quantificadas com o Cell Ranger 10.1.0
(`cellranger count`) contra a referência GRCh38-2024-A, contando também as
leituras intrônicas (`include-introns`, o padrão da versão). A análise partiu
da matriz completa de barcodes (`raw_feature_bc_matrix`), que tinha 18.534.951
barcodes com pelo menos 1 UMI (de 801.357 a 1.869.031 por biblioteca). A
chamada de células do próprio Cell Ranger identificou 37.898 células, de 117
(HNP44_NS) a 11.374 (PB21_ML) por biblioteca. A fração intrônica de cada
barcode foi calculada a partir do `molecule_info.h5` como 1 − (moléculas
exônicas / moléculas totais).

## Chamada de células e avaliação de gotas vazias

A chamada do Cell Ranger foi avaliada com dois métodos independentes, aplicados
à matriz completa de cada biblioteca:

1. **emptyDrops** (DropletUtils 1.30.0; Lun et al., 2019). Os barcodes com até
   100 UMIs definiram o perfil de RNA ambiente. Cada barcode acima disso foi
   testado contra esse perfil (10.000 iterações de Monte Carlo), e foram
   chamados os barcodes com FDR ≤ 0,001. Como controle, rodou-se também o
   `emptyDropsCellRanger`, que reproduz o algoritmo do Cell Ranger.
2. **CellBender** 0.4.0 (`remove-background`; Fleming et al., 2023), na GPU,
   com o número de células do Cell Ranger como `--expected-cells`, FPR de 0,01
   e 150 épocas. Foram considerados células os barcodes com probabilidade
   posterior > 0,5.

O `emptyDropsCellRanger` reproduziu 37.783 das 37.898 células do Cell Ranger.
O emptyDrops clássico chamou 74.569 barcodes, e o CellBender, 59.543. Os
barcodes chamados por eles e não pelo Cell Ranger eram, em grande parte,
material nuclear (fração intrônica alta). Esse material se concentrava em
PB21_NS, onde o CellBender aceitou 18.584 barcodes a mais. Por isso nenhum dos
dois métodos foi usado como filtro. A chamada do Cell Ranger foi mantida como
base, e os barcodes fora dela só foram admitidos pelo teste descrito a seguir.

### Teste de perfil de célula

Para decidir se um barcode raso fora da chamada do Cell Ranger contém uma
célula ou só RNA ambiente, foi usado um teste de razão de verossimilhança,
implementado neste trabalho (`droplets.cell_likeness`). Para cada biblioteca:

- **Perfis.** O perfil de célula, *p*<sub>c</sub>, é a soma das contagens das
  células do Cell Ranger. O perfil ambiente, *p*<sub>a</sub>, é a soma dos
  barcodes com 1 a 100 UMIs. Os dois recebem pseudocontagem de 0,5 e são
  normalizados para frações.
- **Score.** Cada gene recebe o peso
  *w*<sub>g</sub> = log *p*<sub>c,g</sub> − log *p*<sub>a,g</sub>. O score de
  um barcode é a média de *w* sobre os seus UMIs, ou seja, a log-razão de
  verossimilhança por UMI entre os dois perfis multinomiais. O valor esperado
  do score não depende da profundidade; a variância depende.
- **Calibração na profundidade dos barcodes testados.** A distribuição de UMIs
  usada na calibração foi a dos barcodes chamados pelo emptyDrops e não pelo
  Cell Ranger.
  - *Controle negativo:* 600 pseudo-barcodes, montados somando gotas vazias de
    20 a 100 UMIs e subamostrando a soma (binomialmente) até uma profundidade
    sorteada daquela distribuição. O limiar é o percentil 99 dos scores do
    controle negativo, ou seja, 1% de falsos positivos por construção.
  - *Controle positivo:* 500 células do Cell Ranger, subamostradas duas vezes
    até profundidades sorteadas da mesma distribuição. A fração delas acima do
    limiar estima a sensibilidade, que variou de 57% a 96% entre as bibliotecas.
- **Decisão.** Um barcode tem perfil de célula quando o score passa do limiar
  da sua biblioteca.

Dos barcodes chamados só pelo emptyDrops, 9,9% dos citoplasmáticos e 0,5% dos
nucleares tinham perfil de célula. O teste não foi aplicado às células do Cell
Ranger. Nelas, ele reprovaria 8.512 de 34.458 células que passam nas demais
regras de QC, com mais citocinas detectadas (11,2% contra 9,2%) e menos UMIs
(mediana 7.243 contra 9.778). Isso indica que o teste penaliza perfis que se
afastam da célula média da biblioteca, e não só gotas vazias.

## Controle de qualidade

O controle de qualidade seguiu o capítulo correspondente do *Single-cell best
practices* (Heumos et al., 2023), calculado sobre as contagens brutas com o
Scanpy 1.11.5 (`calculate_qc_metrics`). Os limiares foram estimados por
biblioteca, sobre a distribuição das células do Cell Ranger, com o MAD sem
fator de escala:

- número de UMIs e de genes (em log1p): no máximo 5 MADs acima da mediana;
- porcentagem de UMIs nos 20 genes mais expressos: até 5 MADs da mediana, nos
  dois lados;
- fração mitocondrial: reprovação quando passa de 3 MADs acima da mediana em
  escala logit, com logit((*n*<sub>MT</sub> + 1)/(*n* + 2)), **e** quando a
  porcentagem mitocondrial passa de 8%;
- material nuclear: reprovação quando a fração intrônica passa de 5 MADs acima
  da mediana das células do Cell Ranger da biblioteca.

O limite inferior de 5 MADs para o número de UMIs, como recomenda o capítulo,
ia de ~150 UMIs (CTH92_ML, CTH97_ML) a ~3.800 UMIs (CTH94_NS). Nas bibliotecas
mais profundas, ele descartava células com perfil normal. Por isso foi
substituído por um mínimo fixo de **200 UMIs** em todas as bibliotecas,
aplicado ao mesmo tempo como chamada e como limite inferior do QC. Foram
mantidos:

- os barcodes do Cell Ranger que passaram em todas as regras: 36.190 (saíram
  1.708);
- os barcodes fora do Cell Ranger com ≥ 200 UMIs que passaram nas mesmas regras
  **e** no teste de perfil de célula: 789. Eles ficaram anotados como
  `baixa_profundidade`.

Ao todo, 36.979 células passaram no QC.

## RNA ambiente

A contaminação por RNA ambiente foi estimada e corrigida com o SoupX 1.6.2
(Young & Behjati, 2020), biblioteca a biblioteca. Em cada uma, as células
aprovadas foram agrupadas rapidamente, como no capítulo de QC: normalização
para 10⁴ UMIs, log1p, PCA de 50 componentes, grafo de 15 vizinhos e Leiden.
A matriz completa, as células e os grupos foram passados ao SoupX
(`autoEstCont`), e as contagens foram corrigidas com `adjustCounts`, com
arredondamento para inteiros. A fração estimada de contaminação (ρ) ficou
entre 1,0% e 1,7%. As contagens corrigidas foram usadas em todas as etapas
seguintes. As brutas foram mantidas em `layers["counts_raw"]`.

## Remoção de doublets

Os doublets foram identificados com o scDblFinder 1.24.10 (Germain et al.,
2021) nas contagens corrigidas, com parâmetros padrão e semente 123. O limiar
padrão marcaria 7,6% das células, acima da taxa física esperada (~4,5%). Por
isso o limiar foi recalibrado por biblioteca: marca exatamente a fração
esperada de multipletos do Chromium, 0,8% por 1.000 células recuperadas
(quantil 1 − taxa esperada do score).

Além disso, foram removidos os doublets heterotípicos T + B. A identificação
usou a pequena população de células B que escapou da separação como "segunda
espécie":
- **com B:** barcode com ≥ 5 UMIs e ≥ 2 genes distintos de marcadores de célula
  B que não são imunoglobulina;
- **com T:** barcode com ao menos um terço dos UMIs de marcadores de T de uma
  célula T típica da mesma biblioteca, em ≥ 3 genes distintos.

Saíram 1.732 doublets, e restaram **35.247 células** (de 111 em HNP44_NS a
9.929 em PB21_ML; Tabela 1).

## Filtro de genes, normalização e redução de dimensão

As bibliotecas foram concatenadas pela união dos genes. Foram mantidos os genes
detectados em ≥ 20 células **no conjunto** (não por biblioteca): 26.053 genes,
com 14 das 16 citocinas de interesse. O filtro por biblioteca seguido de
interseção, usado no pipeline original, eliminava 12 dessas 16 citocinas.

As contagens corrigidas foram normalizadas para 10⁴ UMIs por célula e
transformadas em log1p. Os 2.000 genes mais variáveis foram selecionados pelo
método `seurat_v3` (Stuart et al., 2019) sobre as contagens, com a biblioteca
como lote. Quando o ajuste loess de um lote era numericamente singular com
`span = 0,3`, ele foi refeito com um span maior. A matriz log-normalizada
desses genes foi escalonada (média zero, variância unitária, valores limitados
a 10) e resumida por PCA de 50 componentes (`random_state = 42`). O objeto
resultante, com 35.247 células × 26.053 genes, é
`dataReseq/P05_integracao/reseq_raw_integrado__v2.h5ad`.

## Integração e clusterização

O método de integração foi escolhido por um benchmark no formato do scIB
(Luecken et al., 2022; scib 1.1.7). Foram 18 execuções, todas sobre as mesmas
2.000 HVG:
- PCA sem integração;
- Harmony, Scanorama, BBKNN e ComBat, com e sem escalonamento;
- scVI, Seurat v5 CCA e RPCA;
- scANVI e scGen. Esses dois foram treinados com o rótulo do Azimuth e ficaram
  fora da escolha.

O lote foi a biblioteca. A conservação biológica foi medida com o Azimuth do
Cell Ranger agrupado em sete classes. O score geral foi 0,4 × remoção de lote
+ 0,6 × conservação biológica, com escalonamento min-max entre as execuções.
A conservação da separação NS × ML dentro do indivíduo serviu de critério de
veto e de desempate. Venceu o **ComBat** (Johnson et al., 2007;
`scanpy.pp.combat` por biblioteca) aplicado à matriz log-normalizada e
escalonada. Ele ficou entre os três primeiros também quando o benchmark foi
refeito com a anotação por marcadores e com o atlas de Yasumizu et al. (2024)
como rótulos.

A PCA de 50 componentes da matriz corrigida foi usada para o grafo de 15
vizinhos, a UMAP e o Leiden. A resolução foi escolhida entre 0,1 e 1,0 pela
concordância entre três sementes (ARI). A resolução 1,0 deu 19 clusters, dos
quais 4 foram marcados como técnicos.

## Software

Python 3.11 com Scanpy 1.11.5, AnnData 0.12.16, NumPy 2.4.6, SciPy 1.17.1,
pandas 2.3.3, scikit-learn 1.8.0, scikit-misc 0.5.2, leidenalg 0.12.0,
igraph 1.0.0, umap-learn 0.5.12, scib 1.1.7, scvi-tools 1.4.2, harmonypy 2.0.0,
Scanorama 1.7.4 e bbknn 1.6.0. R 4.5.3 com DropletUtils 1.30.0, SoupX 1.6.2,
scDblFinder 1.24.10, Seurat 5.5.1 e harmony 2.0.5. CellBender 0.4.0.

## Tabela 1 — Células por biblioteca em cada etapa

| biblioteca | Cell Ranger | após QC (200 UMIs) | final (sem doublets) |
| --- | ---: | ---: | ---: |
| CTH92_NS | 851 | 866 | 860 |
| CTH92_ML | 1.570 | 1.592 | 1.569 |
| CTH94_NS | 6.090 | 5.967 | 5.669 |
| CTH94_ML | 547 | 545 | 535 |
| CTH97_NS | 1.628 | 1.566 | 1.545 |
| CTH97_ML | 4.737 | 4.849 | 4.658 |
| PB20_NS | 635 | 636 | 626 |
| PB20_ML | 841 | 850 | 841 |
| PB21_NS | 4.266 | 3.913 | 3.790 |
| PB21_ML | 11.374 | 10.879 | 9.929 |
| MB34_NS | 2.161 | 2.152 | 2.114 |
| MB34_ML | 1.008 | 1.006 | 992 |
| HNP44_NS | 117 | 112 | 111 |
| HNP44_ML | 2.073 | 2.046 | 2.008 |
| **total** | **37.898** | **36.979** | **35.247** |

A coluna "após QC" inclui as 789 células de baixa profundidade de fora do Cell
Ranger, por isso pode passar do número do Cell Ranger.

## Referências

- Fleming SJ et al. Unsupervised removal of systematic background noise from
  droplet-based single-cell experiments using CellBender. *Nat Methods*
  20:1323–1335 (2023).
- Germain PL et al. Doublet identification in single-cell sequencing data
  using scDblFinder. *F1000Research* 10:979 (2021).
- Heumos L et al. Best practices for single-cell analysis across modalities.
  *Nat Rev Genet* 24:550–572 (2023).
- Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray
  expression data using empirical Bayes methods. *Biostatistics* 8:118–127
  (2007).
- Luecken MD et al. Benchmarking atlas-level data integration in single-cell
  genomics. *Nat Methods* 19:41–50 (2022).
- Lun ATL et al. EmptyDrops: distinguishing cells from empty droplets in
  droplet-based single-cell RNA sequencing data. *Genome Biol* 20:63 (2019).
- Stuart T et al. Comprehensive integration of single-cell data. *Cell*
  177:1888–1902 (2019).
- Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
  well-connected communities. *Sci Rep* 9:5233 (2019).
- Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression
  data analysis. *Genome Biol* 19:15 (2018).
- Yasumizu Y et al. Single-cell transcriptome landscape of circulating CD4⁺ T
  cell populations in autoimmune diseases. *Cell Genomics* 4:100473 (2024).
- Young MD, Behjati S. SoupX removes ambient RNA contamination from
  droplet-based single-cell RNA sequencing data. *GigaScience* 9:giaa151
  (2020).
