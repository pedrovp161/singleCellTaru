# Reseq — avaliação do ressequenciamento

As 14 bibliotecas de células T CD4⁺ (7 pacientes × estimulado com *M. leprae*,
ML / não estimulado, NS) foram ressequenciadas com muito mais profundidade
(`data/allOuts`, Cell Ranger 10.1.0). O objetivo era recuperar as citocinas,
que apareciam em poucas células.

Este guia conta, em ordem, tudo o que foi feito com esses dados: a
escolha do método de integração, a clusterização, a anotação dos tipos
celulares, o método de clusterização alternativo (programas de NMF de Yasumizu
et al., 2024) e o benchmark que compara os dois. Para cada etapa, diz por que
ela foi feita, o que se achou e onde está o notebook.

Os números completos estão no relatório,
[`resultsReseq/RELATORIO_reseq.md`](../../resultsReseq/RELATORIO_reseq.md).
Figuras e tabelas ficam em `resultsReseq/figures/<notebook>/` e
`resultsReseq/tables/<notebook>/`.

> **Limite de inferência.** 7 pacientes, 14 bibliotecas. MB e NP têm um
> paciente cada, e 41% das células vêm de um paciente (PB21). O único
> contraste testado é o pareado estimulado × não estimulado (n = 7), e mesmo
> esse é exploratório.

## Resposta curta

1. **Mais sequenciamento não resolve as citocinas.** As bibliotecas estão
   saturadas (89–99,5%). Cada célula ganhou de +4% a +48% de UMIs, e as
   citocinas continuam em menos de 1% das células, exceto IFNG (3%) e TNF
   (6%). O limite está no experimento: poucas células respondem, e a resposta
   se concentra numa biblioteca (CTH97_ML).
2. **A baixa recuperação de células não vem de doublets.** Isso foi testado com
   quatro linhas de evidência e com três detectores (Scrublet, Solo, cxds). As
   bibliotecas pequenas têm poucas células porque poucas células entraram.
3. **O pipeline original tem dois defeitos, e eles se repetem nos dados
   novos.** O `inner join` entre bibliotecas joga fora 12 de 16 citocinas. A
   PCA sobre contagens cruas faz os clusters seguirem a profundidade.
4. **Partir da matriz raw quase não recupera células úteis.** O critério final
   é um mínimo único de **200 UMIs** para as 14 bibliotecas, mais o perfil de
   célula para quem está fora da chamada do Cell Ranger e os doublets
   calibrados. Resultado: **35.247 células**, contra 35.064 antes.
5. **Integração: ComBat.** No benchmark no formato do scIB (Luecken et al.,
   2022), com 10 métodos e 18 execuções, o ComBat teve o maior score geral
   entre os métodos que não usam rótulo. Ele também preserva mais a diferença
   estimulado × não estimulado que o Harmony usado antes.
6. **Clusterização: Leiden 1,0 sobre o ComBat**, com 19 clusters.
7. **Anotação por marcadores canônicos.** Os 19 clusters são CD4⁺ naive (33%),
   de memória (37%), ativadas (9%), Treg (5%) e citotóxicas (4%), mais 1,2% de
   contaminantes (B, monócito, progenitor CD34⁺, basófilo) e 11% de baixa
   qualidade. A anotação concorda com um atlas CD4⁺ independente em 79% das
   células.
8. **Clusterização alternativa: programas de NMF** (Yasumizu et al., 2024).
   Num benchmark com réguas independentes, ela supera o Leiden: concorda mais
   com o atlas (ARI 0,23 contra 0,14) e é muito mais estável (0,94 contra 0,47
   em subamostras). Separa estados que o Leiden mistura, como a resposta a
   interferon e a Treg efetora. Metade das células, porém, é mistura de
   programas.
9. **O benchmark de integração refeito com rótulos melhores mantém o ComBat**,
   mas não como vencedor isolado. Com a anotação e com o atlas, ComBat,
   Seurat RPCA e Harmony (50 dimensões) ficam a poucos centésimos um do outro.
   O ComBat fica entre os 3 primeiros em todas as variantes e é o único do grupo
   que devolve a matriz de genes que a NMF exige. O Harmony de 15 dimensões do
   pipeline original fica sempre para trás, e a vantagem do scGen no P07 vinha
   do rótulo com que foi treinado.

## O caminho completo

```text
data/allOuts (Cell Ranger 10.1.0, 14 bibliotecas)
 │
 ├─ Parte 1 · comparativa/   o ressequenciamento ajudou?
 │     C01 arquivos do Cell Ranger → C02 antigo × novo → C03 rarefação
 │
 ├─ Parte 2 · analise/       o pipeline original nos dados novos (matriz filtrada)
 │     A01 QC → A02 o que falta → A03 doublets → A04 processamento (V1/V2/V3)
 │     → A05 clusters → A06 Solo e cxds
 │
 ├─ Parte 3 · raw/           refazer do zero a partir da matriz raw
 │     P01 emptyDrops → P02 CellBender → P03 QC + 200 UMIs → P04 SoupX + doublets
 │     → P05 objeto principal → P06 balanço
 │     → P07 benchmark de integração (scIB) → ComBat
 │     → P08 clusterização (Leiden)          → objeto final
 │
 └─ Parte 4 · raw/           anotação e o método alternativo
       atlas de Yasumizu et al. (2024) → régua independente (Symphony)
       P09 anotação por marcadores → P10 programas de NMF (o método do artigo)
       → P11 benchmark Leiden × NMF → P12 benchmark de integração com os rótulos novos
```

Cada parte responde a uma pergunta e alimenta a seguinte. A Parte 2 mostrou o
que estava errado no pipeline antigo. A Parte 3 refaz o pré-processamento com
essas correções e termina no objeto que deve ser usado daqui para frente. A
Parte 4 anota esse objeto e compara o Leiden com o método de clusterização do
artigo.

---

## Parte 1 — O ressequenciamento ajudou? (`comparativa/`)

### C01 — Arquivos do Cell Ranger

**O que faz.** Lê o JSON embutido nos 14 `web_summary.html`, o
`metrics_summary.csv`, os logs e a composição automática (Azimuth). Confere
parâmetros e proveniência, lista os alarmes e mede o retorno da profundidade
pelas curvas de saturação.

**O que se achou:**

- saturação de 89,0% a 99,5% (≥ 95% em 12 de 14), com 70 reads por molécula
  na mediana;
- 5 bibliotecas com fração de reads em células abaixo do ideal (alarme do
  Cell Ranger);
- CTH92_ML e CTH97_ML com complexidade muito baixa (1.832 e 1.009 genes por
  célula, contra ~3.000 nos pares NS);
- duas pastas com o sufixo trocado (`MB_34_SN`, `PB_20_SN`): o
  `endswith("NS")` do código original as rotularia como estimuladas;
- sem BAM: não há como genotipar nem reconstruir TCR.

### C02 — Corrida antiga × nova, célula a célula

**O que faz.** Pareia as células das duas corridas. Compara chamada de
células, UMIs, genes e células positivas para citocina.

**O que se achou:**

- são as mesmas bibliotecas (sobreposição de barcodes de 0,991 a 1,000; sexo
  dos doadores consistente);
- células chamadas: 37.161 → 37.898 (+2%). Os barcodes novos estão na
  fronteira do limiar, não são células recuperadas;
- citocinas quase iguais: IFNG 977 → 1.121 células, TNF 1.973 → 2.381. Das
  células que ganharam positividade, 95% têm exatamente 1 UMI;
- CTH97_ML concentra a resposta (56% das IFNG⁺, 91% das IL17F⁺);
- o estímulo não aumenta a fração de células com citocina na maioria dos
  pacientes (IFNG: ML > NS em 3 de 7).

### C03 — Rarefação e citocinas

**O que faz.** A partir do `molecule_info.h5`, simula o que cada citocina
mostraria com menos ou mais reads.

**O que se achou:**

- com metade dos reads atuais já apareciam 85–97% das células positivas;
- dobrar a profundidade de novo renderia, no máximo, +2,6% de UMIs;
- 988 genes são sustentados quase só por moléculas intrônicas lidas 1–2 vezes.
  Os genes do cromossomo Y em doadoras mostram o mesmo perfil e servem de
  controle negativo. Esses genes não devem ser usados como marcadores.

**Conclusão da Parte 1:** não vale a pena sequenciar mais. Mais sinal de
citocina exige mais células respondedoras.

---

## Parte 2 — O pipeline original nos dados novos (`analise/`)

Aqui o ponto de partida é a matriz **filtrada** do Cell Ranger, como no
projeto original.

### A01 — Pré-processamento original

É a réplica do `src/scripts/preprocessing.ipynb`: filtro MAD combinado
(k = 5) do spatools, flag de MT e Scrublet com `expected_doublet_rate=0.06`.
Antes de rodar nos dados novos, foi validada nos antigos (14 de 14 bibliotecas
idênticas). Nos dados novos, o Scrublet remove só 0,18% das células.

### A02 — Falta algum pré-processamento?

**O que se achou:**

- **material nuclear de células mortas** em três bibliotecas NS: 21.097
  barcodes em PB21_NS, 4.873 em CTH97_NS e 1.289 em MB34_NS, com ≥ 50% de
  leitura intrônica. Indica morte celular antes da captura;
- **RNA ambiente baixo** dentro das células (0,2% a 3,3%);
- **o `inner join` entre bibliotecas remove citocinas:** depois de
  `filter_genes` por amostra, sobram 18.852 genes e somem 12 das 16 citocinas.
  A correção é a união com um filtro global;
- **118 genes técnicos** entre os genes variáveis (segmentos de TCR,
  imunoglobulinas, genes de reads isolados, genes de sexo).

### A03 — A baixa recuperação vem de doublets?

Quatro linhas de evidência independentes:

| evidência | resultado |
| --- | --- |
| aritmética do carregamento | taxa esperada ≤ 1,5% nas bibliotecas pequenas |
| régua de RNA (mistura em log-UMI) | teto de doublets T + T de 0–4,1%, crescendo com o número de células, o oposto do que a hipótese prevê |
| células B como segunda "espécie" | 70 doublets T + B validados (0,20% dos barcodes), conjugados pré-formados |
| material fora das células | as bibliotecas pequenas quase não têm barcodes de material: entrou pouca célula |

**Conclusão:** a hipótese de doublets não se sustenta. O Scrublet é cego
nesta população: pega 1 dos 70 doublets validados no limiar automático e 19
no limiar calibrado.

### A04 — Processamento original e correções

É a réplica do `src/scripts/processing.ipynb` (validada: reproduz o
`data/adata_harmony.h5ad` exatamente). Roda em três variantes, mudando uma
coisa por vez:

| variante | universo de genes | PCA sobre | citocinas mantidas |
| --- | --- | --- | --- |
| V1 (a original) | interseção (18.852 genes) | contagens cruas | 4 de 16 |
| V2 | interseção | lognorm escalonada | 4 de 16 |
| **V3 (proposta)** | união + `min_cells=3` global (31.756 genes) | lognorm escalonada | 15 de 16 |

**O que se achou:**

- **PCA sobre contagens cruas (V1):** 50 componentes capturam 98% da
  variância, quase toda em 10 genes (MALAT1, imunoglobulinas). Os clusters
  seguem a profundidade: o cluster explica 57% da variância do log de UMIs
  dentro da biblioteca, contra 22% na V3;
- os clusters da V1 não sobrevivem à correção da PCA (ARI V1 × V2 = 0,08) e
  não devem ser anotados;
- **o Harmony por biblioteca também corrige o estímulo:** a separação entre as
  duas bibliotecas de cada paciente cai de 0,61 para 0,22. Isso acontece
  porque estímulo e biblioteca estão confundidos.

### A05 — Os clusters descrevem células ou artefatos?

Na V3:

- 4 clusters técnicos (601 células: células B, conjugados T–B, não-T);
- fronteiras instáveis: ARI de 0,52 entre sementes do Leiden;
- o ressequenciamento não mudou o agrupamento (ARI antigo × novo de 0,52,
  igual ao ARI entre duas sementes).

### A06 — Solo e cxds

Dois detectores de doublets com lógicas diferentes do Scrublet, testados
contra os 70 doublets T + B validados:

| | Scrublet | Solo (scvi-tools) | cxds (scds) |
| --- | --- | --- | --- |
| separação dos validados (AUC) | 0,70 | 0,82 | 0,73 |
| validados pegos no limiar calibrado | 19/70 | 20/70 | 21/70 |
| correlação do score com os UMIs | 0,30 | 0,75 | 0,74 |

O Solo marca 14,7% das células no limiar padrão, e o score acompanha o
conteúdo de RNA: não deve ser usado para remover células. Nenhum dos três
sustenta a hipótese de doublets. O consenso fica como coluna de sensibilidade.

**Conclusão da Parte 2:** a linha de base certa é a V3 (união de genes, PCA
sobre a lognorm). Faltava refazer o pré-processamento de forma mais rigorosa,
partindo da matriz raw. É a Parte 3.

---

## Parte 3 — Do zero, a partir da matriz raw (`raw/`)

A pergunta: partindo de **todos** os barcodes (`raw_feature_bc_matrix.h5`) e
seguindo o
[sc-best-practices](https://www.sc-best-practices.org/introduction/raw-data-processing/),
dá para recuperar células que o Cell Ranger deixou de fora? E qual seria um
critério único para as 14 bibliotecas?

### P01 — Gotas vazias com o emptyDrops

**O que é.** Cada barcode é uma gota. A maioria tem só o "caldo" de RNA solto
na suspensão (RNA ambiente), e poucas têm uma célula. O `emptyDrops`
(DropletUtils) compara cada barcode com o perfil desse caldo e chama de célula
quem é diferente dele.

**O que se achou:**

- o `emptyDropsCellRanger`, que imita o Cell Ranger, reproduziu a chamada
  dele (37.783 de 37.898 células);
- o `emptyDrops` clássico, sem o piso de 500 UMIs do Cell Ranger, acrescentou
  **38.005 barcodes**. Eles têm 153 UMIs na mediana (as células têm ~9 mil), e
  metade tem perfil de núcleo de célula morta (21 mil só em PB21_NS).

**O teste de "perfil de célula".** Como saber se um barcode com 150 UMIs é uma
célula pequena ou só caldo? O teste mede, molécula a molécula, se o conteúdo
se parece mais com as células da biblioteca ou com as gotas vazias. Ele usa
dois controles na mesma profundidade:

- **positivo:** células reais subamostradas até ~150 UMIs;
- **negativo:** gotas vazias somadas até a mesma profundidade.

O corte deixa passar só 1% das gotas vazias. Por ele, ~2.300 dos 38.005 extras
têm perfil de célula, quase todos com menos de 500 UMIs.

### P02 — CellBender

É uma segunda opinião, em Python e na GPU, com um modelo da matriz inteira. O
CellBender concordou com o Cell Ranger. Como o emptyDrops, tomou o material
nuclear de PB21_NS por célula. Estimou de 0,9% a 1,9% de RNA ambiente dentro
das células.

### P03 — Controle de qualidade e o valor único de 200 UMIs

**As regras do capítulo de QC**, por biblioteca:

- UMIs, genes e % de UMIs nos 20 genes mais expressos: 5 desvios absolutos da
  mediana (MAD) para cima e para baixo;
- fração mitocondrial: 3 MADs acima em escala logit **e** acima de 8%;
- mais uma regra própria: tirar barcodes com perfil de núcleo (fração
  intrônica alta).

**O achado que mudou a abordagem.** O limite inferior de 5 MADs varia de ~150
UMIs (CTH92_ML) até ~3.800 (CTH94_NS). Nas bibliotecas profundas, ele
descartava células razoáveis. Por isso foi testado um **valor único**, o mesmo
como chamada e como mínimo do QC:

| mínimo de UMIs | células | extras ao Cell Ranger | extras com perfil de célula |
| --- | --- | --- | --- |
| **200** | 39.544 | 3.354 | 789 |
| 300 | 37.936 | 1.746 | 471 |
| 500 | 36.940 | 750 | 17 |
| 1.000 | 35.514 | 499 | 8 |
| 5 MADs por biblioteca | 35.671 | 743 | 227 |

**O padrão adotado** (`config.QC_MIN_UMIS = 200`):

- as **células do Cell Ranger** entram se passarem nas demais regras;
- os **barcodes fora do Cell Ranger** entram só com perfil de célula. São 789,
  marcados em `obs["baixa_profundidade"]`;
- o teste de perfil **não** é aplicado às células do Cell Ranger. Aplicado a
  células profundas, ele reprovaria ~8.500 células reais, inclusive as
  ativadas.

### P04 — RNA ambiente e doublets

- **SoupX:** a contaminação estimada fica em ≤ 1% na maioria das bibliotecas.
  A correção é aplicada às contagens e mexe sobretudo nos eventos de 1 UMI de
  citocina de CTH97_ML.
- **scDblFinder:** com o limiar padrão, marcaria 7,6% das células, contra
  ~4,5% esperados pela física do carregamento. Por isso o limiar foi
  **calibrado pela taxa do 10x**, biblioteca a biblioteca.
- **Saem 1.732 doublets:** os calibrados mais os T + B validados em A03.

### P05 — O objeto principal

- Filtro de genes do capítulo: gene em ≥ 20 células **no conjunto**, não por
  biblioteca. Ficam 26.053 genes e 14 de 16 citocinas.
- Processamento V3 (A04): 2.000 genes variáveis por biblioteca, PCA sobre a
  lognorm escalonada, Harmony por biblioteca, vizinhança em 15 dimensões e
  UMAP. É a integração herdada do pipeline original; o P07 a põe à prova.
- Grava `dataReseq/P05_integracao/reseq_raw_integrado__v2.h5ad`, o **objeto
  principal**.

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

A diferença vem de **+1.262** células do Cell Ranger que o MAD descartava,
**+789** de baixa profundidade e **−1.732** doublets.

### P07 — A escolha do método de integração (benchmark scIB)

**A pergunta.** O Harmony herdado do pipeline original é a melhor integração
para estes dados? O P07 repete o benchmark de
[Luecken et al. (2022), *Nat Methods* 19:41–50](https://www.nature.com/articles/s41592-021-01336-8):
os mesmos métodos (os que têm versão mantida), as mesmas métricas (o pacote
do artigo, `scib`), a mesma agregação e figuras no mesmo formato.

**Os métodos.** São 10 métodos e 18 execuções. Todos usam as mesmas 2.000 HVG,
com e sem escala quando o método aceita as duas:

| método | saída | observação |
| --- | --- | --- |
| não integrado (PCA) | embedding | referência |
| Harmony | embedding | inclui a configuração do P05 (vizinhança em 15 dimensões) |
| Scanorama | embedding e matriz corrigida | |
| BBKNN | grafo | |
| ComBat | matriz corrigida | |
| scVI | embedding | GPU |
| scANVI\*, scGen\* | embedding / matriz | usam o rótulo no treino: aparecem, mas não são elegíveis |
| Seurat v5 CCA e RPCA | embedding | R isolado (`IntegrateLayers`) |

**As métricas**, calculadas nas 33.588 células com rótulo:

- **remoção de lote:** PCR batch, batch ASW, graph iLISI, graph connectivity
  e kBET;
- **conservação biológica:** NMI, ARI, cell type ASW, isolated label F1 e
  silhouette, graph cLISI, ciclo celular e conservação de HVG.

Cada métrica é escalonada (min-max) entre as execuções, e o **score geral** é
0,4 × lote + 0,6 × bio, como no artigo.

**As adaptações**, decididas antes de ver os resultados:

- **rótulo biológico:** não há anotação. As métricas usam o Azimuth que o Cell
  Ranger rodou em cada biblioteca, agrupado em 7 classes (CD4 naive, CD4
  memória, Treg, CD8 e T não convencional, B, NK/ILC, mieloide);
- **lote = biblioteca.** Como o estímulo está confundido com a biblioteca,
  entra uma métrica do estudo: a **fração da separação NS × ML dentro do
  paciente que o método preserva**. Ela veta quem apaga o estímulo mais que o
  Harmony atual e serve de desempate;
- **regra de escolha:** maior score geral entre os métodos sem rótulo. Quem
  fica a menos de 0,02 do melhor empata, e no empate vence quem preserva mais
  estímulo;
- **kBET:** portado para Python, porque no `scib` ele depende de um pacote R.
  Validado em dados sintéticos: dá 0,97 com lotes misturados e 0 com lotes
  deslocados.

**O resultado:**

| posição (elegíveis) | execução | geral | lote | bio | estímulo preservado |
| --- | --- | --- | --- | --- | --- |
| **1** | **ComBat, escalado** | **0,566** | 0,598 | 0,544 | **53%** |
| 2 | Seurat v5 RPCA | 0,518 | 0,506 | 0,525 | 41% |
| 3 | Seurat v5 CCA | 0,505 | 0,488 | 0,516 | 28% |
| 4 | scVI | 0,501 | 0,683 | 0,379 | 64% |
| 5 | Harmony, escalado (50 dimensões) | 0,498 | 0,543 | 0,468 | 35% |
| 9 | Harmony do P05 (15 dimensões) | 0,448 | 0,474 | 0,432 | 35% |
| — | scGen\* / scANVI\* (não elegíveis) | 0,766 / 0,652 | | | 49% / 68% |

**Por que o ComBat:**

- ficou em 1º sob as 6 formas de agregar testadas (pesos 0,5/0,5, z-score,
  sem escalonar, min-max só entre os elegíveis, sem as métricas de rótulo
  isolado), sem empate;
- remove toda a variância da PCA explicada pela biblioteca (PCR 1,00) e tem o
  maior ARI de todas as execuções (0,51);
- preserva 53% da separação estimulado × não estimulado, contra 35% do
  Harmony;
- só scGen e scANVI pontuaram mais, mas eles treinam com o mesmo rótulo usado
  na avaliação;
- os dois BBKNN foram vetados: misturam tanto que as duas bibliotecas de cada
  paciente ficam mais juntas que o acaso.

A tarefa é simples: um tipo celular separado por citometria, em bibliotecas de
composição parecida. Nesse cenário, um modelo linear por gene como o ComBat
basta. O P12 refaz este benchmark com rótulos melhores: o ComBat continua no
grupo de cima, ao lado do Seurat RPCA e do Harmony com 50 dimensões. Isso é coerente com os benchmarks anteriores que o artigo cita, em que
o ComBat ou o Harmony superavam os métodos não lineares em tarefas simples.

#### Por que não o scGen, que tem a maior conservação biológica?

Pelo score do artigo, sozinho, o scGen venceria: tem o maior score geral
(0,766) e a maior conservação biológica (0,81). Ele ficou de fora pela regra
de elegibilidade, combinada antes de ver os resultados. As razões:

1. **O 0,81 não quer dizer "81% da biologia conservada".** É a média das
   métricas biológicas escalonadas entre as 18 execuções. Diz que o scGen fica
   no topo da faixa na maioria delas, não quanto da biologia ele conserva.
2. **A biologia é medida com o mesmo rótulo que o scGen recebe no treino.**
   Seis das oito métricas biológicas, e três das cinco de lote, são calculadas
   por classe do Azimuth. O scGen corrige alinhando, classe a classe, as
   células de cada classe do Azimuth entre as bibliotecas. Ou seja, ele é
   avaliado com o gabarito na mão. A vantagem dele aparece justamente onde o
   rótulo entra: NMI, cLISI e isolated label F1 ficam no máximo. O F1 de 0,97
   é medido no grupo mieloide, de 42 células, que ele recebeu rotulado.

   | média das métricas biológicas escalonadas | usam o rótulo (6) | não usam (ciclo celular, HVG) |
   | --- | --- | --- |
   | scGen\* | 0,84 | 0,73 |
   | ComBat | 0,56 | 0,49 |

   Nem tudo é circularidade: nas duas métricas que não usam rótulo o scGen
   também vai bem. Mas elas são só duas e, sozinhas, não decidem, porque
   premiam não integrar: o não integrado tira a nota máxima nelas.
3. **Aqui o rótulo confunde estímulo com tipo celular.** O Azimuth chama de
   Treg 11,0% das células estimuladas e 5,3% das não estimuladas, e a fração
   sobe com o estímulo nos 7 de 7 pacientes. Células CD4 ativadas passam a
   expressar CD25 e FOXP3, então parte dessa diferença provavelmente é ativação
   pelo *M. leprae*, e não Treg. O scGen alinharia essas células às Treg das
   bibliotecas não estimuladas e apagaria justamente o efeito que o estudo
   procura. Além disso, as 1.659 células sem rótulo confiável seriam
   corrigidas como se fossem uma classe só.
4. **O próximo passo é a anotação.** Uma integração que impõe o Azimuth ao
   espaço latente tende a devolver clusters iguais às classes do Azimuth. Os
   estados que não estão na referência, como as células ativadas, ficam
   diluídos. Para anotar, a integração precisa ser independente do rótulo.

O próprio artigo marca scANVI e scGen com asterisco e os recomenda quando há
rótulos confiáveis, isto é, curados. **Quando houver uma anotação revisada,
vale refazer o benchmark com ela.** Aí o scGen e o scANVI passam a ser
candidatos legítimos.

**O P12 confirmou o ponto 2.** Avaliado com rótulos que não viu no treino (a
anotação do P09 e o atlas), o scGen cai do 1º para o 6º e o 5º lugar.

**As figuras**, em `resultsReseq/figures/P07_benchmark_integracao/`:

| figura | equivalente no artigo |
| --- | --- |
| `tabela_resumo` | tabela-resumo das Figs. 2a e 3b |
| `lote_x_bio` | remoção de lote × conservação biológica, Fig. 3a |
| `umaps` | UMAPs do não integrado e dos melhores métodos, Fig. 2b–c |
| `tempo_memoria` | tempo e memória por método, Extended Data Fig. 7 |
| `estimulo_x_geral` | métrica do estudo: estímulo preservado × score geral |
| `robustez` | posição de cada método sob as 6 agregações |
| `metricas_brutas` | as métricas antes do escalonamento |

### P08 — Clusterização sobre o ComBat

**A mistura.**

- Os vizinhos da mesma biblioteca caem de 41% para 17%; com o Harmony, eram
  11%.
- A separação entre as duas bibliotecas de cada paciente vai de 0,54 para
  0,33; com o Harmony, ia para 0,22.

O ComBat mistura um pouco menos e apaga menos o estímulo. Ainda assim, **o
efeito do estímulo deve ser testado nas contagens** (pseudobulk pareado por
paciente), não pela presença de clusters.

**A resolução.** O Leiden foi rodado em 8 resoluções, com 3 sementes cada:

| escala | resolução | clusters | ARI entre sementes |
| --- | --- | --- | --- |
| **fina (principal)** | 1,0 | 19 | 0,58 |
| intermediária | 0,2 | 6 | 0,79 |
| grossa | 0,1 | 4 | 0,98 |

Entre 0,6 e 1,0 o ARI quase não muda. A massa central de células T é um
contínuo: só 8 dos 19 clusters reaparecem quase iguais nas três sementes.

**Os clusters técnicos**, a tirar antes da anotação (3.136 células):

| cluster | células | motivo |
| --- | --- | --- |
| 7 | 1.855 | baixo conteúdo de RNA (mediana de 1.353 UMIs) |
| 11 | 914 | 60% de baixa profundidade (mediana de 422 UMIs) |
| 14 | 309 | não-T (marcadores de B ou mieloides) |
| 15 | 58 | não-T |

Sem eles, a profundidade explica 22% da variância de UMIs dentro de cada
biblioteca; com eles, 59%.

Os clusters 4, 7, 10 e 13 têm mais células nas bibliotecas estimuladas em 6
dos 7 pacientes. Nenhum resiste à correção para múltiplos testes, e o 10 é 74%
de CTH97. São hipóteses para a anotação, não resultado.

---

## Parte 4 — Anotação, programas de NMF e o benchmark refeito (`raw/P09`–`P12`)

Três pedidos, sobre os dados corrigidos pelo ComBat:

1. anotar os clusters, com foco em T CD4⁺ e nos contaminantes, por marcadores
   canônicos;
2. testar o método de clusterização de Yasumizu et al. (2024) como alternativa
   ao Leiden;
3. refazer o benchmark para comparar o pipeline novo com o anterior.

### A régua independente: o atlas de Yasumizu et al. (2024)

O artigo indicado (*Cell Genomics* 4:100473,
[doi:10.1016/j.xgen.2023.100473](https://doi.org/10.1016/j.xgen.2023.100473))
é um atlas de células T CD4⁺ do sangue, com 18 tipos de referência (5 no nível
L1: Tnaive, Tcm, Tem, Temra, Treg). Os autores publicam a referência do
Symphony (Figshare, CC BY 4.0). Cada célula nossa que o Azimuth chama de CD4⁺
(33.999) foi mapeada nela (`raw/run_atlas_mapping.py`), com a biblioteca como
lote. Esse mapeamento **não usa a nossa integração nem os nossos clusters**: é
a régua independente da anotação e do benchmark. Dos 1.271 genes da
referência, 1.177 estão nos nossos dados (o atlas é 5′ v2, e os nossos dados,
3′ v3).

Uma adaptação foi necessária para o Seurat 5: o SCTransform roda com a
`vst.flavor = "v1"` da época em que a referência foi montada, e o mapeamento usa
o `symphony::mapQuery` do pacote. A versão dos autores chama uma função que o
harmony 2.x removeu.

### P09 — Anotação por marcadores canônicos

**Como.** Os 19 clusters do Leiden 1,0 (P08) foram anotados com um painel de
marcadores da literatura (`functions/anotacao.py`, com as referências):

- **linhagem:** CD3E/CD3D/CD2/TRAC, CD4/CD40LG, CD8A/CD8B (Hao et al., 2021);
- **naive × memória:** CCR7, SELL, LEF1, TCF7 × FAS, S100A4, ANXA1 (Sallusto
  et al., 1999; Mahnke et al., 2013);
- **Treg:** FOXP3 e Helios/*IKZF2* (Fontenot et al., 2003; Thornton et al.,
  2010);
- **Th17:** CCR6, RORC, CD161/*KLRB1* (Acosta-Rodriguez et al., 2007; Cosmi
  et al., 2008);
- **Tfh:** CXCR5, PDCD1, ICOS (Crotty, 2011);
- **CD4 citotóxica:** GZMB, PRF1, GNLY, CX3CR1 (Patil et al., 2018);
- **ativação pelo TCR:** CD69, TNFRSF4, TNFRSF9, MIR155HG (Szabo et al., 2019);
- **contaminantes:** B, monócito, NK, progenitor CD34⁺ (Hao et al., 2021) e
  basófilo (HDC, CLC, CPA3; Papavasileiou et al., 2025).

Para cada cluster, o notebook mostra a % de células que detectam cada
marcador, os genes diferenciais e o que o atlas diz das mesmas células. A
decisão de cada cluster está escrita, com a evidência, em
`anotacao.LEIDEN_P08`.

**Resultado:**

| classe | clusters | células |
| --- | --- | --- |
| CD4 naive | 0, 2, 12 | 33,4% |
| CD4 memória: Tcm (1), Tcm/Tfh (3), transição naive/Tcm (5), Th17 (6), Tem GZMK⁺ (16) | 1, 3, 5, 6, 16 | 36,8% |
| CD4 ativada pelo TCR | 4, 13 | 9,0% |
| Treg | 8 | 4,9% |
| CD4 citotóxica (Temra) | 9 | 3,7% |
| **contaminantes:** B/plasmócito (14), monócito (15), progenitor CD34⁺ (17), basófilo (18) | 14, 15, 17, 18 | 1,2% (417 células) |
| baixa qualidade (pouco RNA) | 7, 10, 11 | 11,1% |

- **Concordância com o atlas:** 79,2% no tipo L1 (κ = 0,65), em 23.670 células
  CD4⁺. Em naive e Treg, 94,7%, contra 65,7% do Azimuth automático.
- **Onde diverge:** o Tem do atlas cai quase todo em Tcm. Fora do eixo CCR7, o
  painel não separa bem memória central e efetora.
- **Estímulo (descritivo):** as duas classes ativadas têm mais células nas
  bibliotecas estimuladas em 6 de 7 pacientes. O Th17 em repouso vai na direção
  oposta, coerente com parte dele passar ao estado ativado.

### P10 — Programas gênicos por NMF, a alternativa do artigo ao Leiden

**O método.** Os 18 tipos do atlas foram definidos com Leiden. Para descrever
o CD4⁺, porém, os autores defendem outra coisa: como é um contínuo, delimitar
clusters é artificial, e cada célula vira uma **combinação de programas
gênicos** obtidos por fatoração não negativa (NMF). A receita deles, aplicada
aos dados corrigidos pelo ComBat:

1. regressão de UMIs, %MT e ciclo celular;
2. escala, com os negativos zerados;
3. `sklearn` NMF;
4. número de programas depois do cotovelo da variância explicada e antes de a
   correlação máxima entre programas subir.

Para comparar com o Leiden, o "cluster" de cada célula é o **programa
dominante**. As células também foram projetadas nos 12 programas publicados
(NMFproj, CC BY-NC 4.0, uso acadêmico).

**Quantos programas.** Aqui, os dois critérios do artigo só dão "k > 7": a
correlação entre programas fica num platô até k = 20. O desempate foi a
reprodutibilidade entre três sementes, que só k = 9 e k = 14 têm (correlação
de 1,00 e ARI ≥ 0,997). Ficou k = 14, com k = 9 como sensibilidade. Esse
desempate não está no artigo e foi definido depois de ver a tabela de
estabilidade.

**Os 14 programas:**

| programa | genes de maior peso | leitura |
| --- | --- | --- |
| P13 | LEF1-AS1, IGF1R, FHIT | CD4 naive |
| P9 | IFIT3, RSAD2, MX1, ISG15 | CD4 naive com resposta a interferon |
| P5 | S100A6, VIM, MIF | transição naive/Tcm |
| P6 | NR3C1, TGFBR3, SAMSN1 | Tcm (Th0) |
| P10 | MAF, TOX, ITGA4 | Tcm/Tfh |
| P7 | CCR6, KLRB1, SLC4A10 | Th17 |
| P4 | MIR155HG, LTA, CCL20, TNFRSF4 | ativação pelo TCR |
| P3 | IKZF2, FOXP3, RTKN2 | Treg |
| P11 | HLA de classe II, ENTPD1/CD39 | Treg efetora |
| P8 | NKG7, GNLY, GZMH, PRF1 | CD4 citotóxica |
| P0, P1, P12 | BANK1/MS4A1; FCER1G/CD68; GATA2/SOX4 | contaminantes: B, monócito, progenitor/basófilo |
| P2 | MALAT1, NEAT1 | técnico (baixa qualidade) |

- A NMF separa dois estados que o Leiden 1,0 misturava: a resposta a
  interferon e a Treg efetora.
- **O contínuo:** em metade das células (50,5%), o programa dominante tem
  menos de 40% do total. O programa dominante é um resumo, e as análises devem
  usar a atividade de cada programa por célula (H).
- **Os programas publicados** projetam bem nos nossos dados (POH 0,20, acima do
  limite de 0,1 dos autores). Como clusters, porém, não servem: o Naive-F
  domina 61% das células.

### P11 — Leiden × NMF

Cinco partições sobre os mesmos dados: Leiden 1,0 (o pipeline anterior),
Leiden 0,6 (o mesmo grafo com 14 grupos, para separar o método do número de
grupos), NMF k = 14 (o pipeline novo), NMF k = 9 e a NMF projetada. As métricas
vêm em cinco grupos. Cada métrica é escalonada entre as partições, e o geral é
a média dos grupos:

| | Leiden 1,0 (anterior) | Leiden 0,6 | **NMF k = 14 (novo)** | NMF k = 9 | NMF projetada |
| --- | --- | --- | --- | --- | --- |
| ARI com o atlas L2 | 0,14 | 0,15 | **0,23** | 0,24 | 0,21 |
| ARI em subamostras de 80% | 0,47 | 0,41 | **0,94** | 0,98 | — |
| η² da profundidade (menor é melhor) | 0,59 | 0,26 | **0,33** | 0,07 | 0,04 |
| AUROC dos marcadores | 0,77 | 0,80 | **0,77** | 0,78 | 0,78 |
| anotação × atlas L1 (acurácia) | 0,79 | — | **0,80** | — | 0,70 |
| **score geral** | 0,28 | 0,48 | **0,65** | 0,79 | 0,25 |

- **O pipeline novo supera o anterior** em concordância com a régua
  independente, em robustez e no peso técnico. Também supera o Leiden com o
  mesmo número de grupos.
- **Estados isolados:** a NMF isola melhor a maioria dos estados do atlas. Por
  exemplo, F1 do melhor grupo de 0,48 contra 0,23 no naive com interferon, 0,67
  contra 0,50 na Treg efetora e 0,44 contra 0,27 no Tfh.
- **Onde o Leiden ganha:** os marcadores mais específicos são os do Leiden
  0,6. Fronteiras de grafo dão grupos mais "limpos".
- **Anotação:** empate com o atlas (80,4% contra 79,2%).
- **A ressalva da robustez:** a robustez entre sementes da NMF (0,998) é alta
  por construção, porque foi o critério da escolha de k. A das subamostras
  (0,94 contra 0,47) não é.
- **O k = 9:** tira a maior nota geral, mas não foi anotado e é comparado sem
  esse grupo.

### P12 — O benchmark de integração com os rótulos novos

O P07 escolheu o ComBat medindo a conservação biológica com o Azimuth
automático, o único rótulo que havia. Agora há dois melhores: a anotação do
P09, curada, mas nascida dos clusters sobre o ComBat, e o atlas, independente.
As 18 execuções foram reavaliadas com cada um, com as mesmas métricas, a
mesma agregação e a mesma regra de escolha.

**O rótulo isolado.** Nas réguas novas, o rótulo "isolado" é minúsculo:
basófilo (14 células) na anotação e TcmPHLDA3 (18) no atlas. No atlas, nenhuma
integração o isola (F1 < 0,01 em todas), e o escalonamento transformaria ruído
em diferença. A agregação principal deixa de fora as duas métricas de rótulo
isolado, e a completa do artigo aparece ao lado.

| régua | maior score elegível | escolhido pela regra (empate → estímulo) | ComBat entre os elegíveis |
| --- | --- | --- | --- |
| Azimuth (P07) | ComBat (0,566) | ComBat | 1º |
| anotação (P09) | Seurat RPCA (0,717) | Seurat RPCA | 2º (0,695) |
| atlas L2 | Harmony, 50 dimensões (0,726) | Seurat RPCA (empata com o Harmony e preserva mais estímulo) | 3º (0,696) |
| atlas L2, agregação completa | Harmony, 50 dimensões | ComBat (empata com o Harmony e preserva mais estímulo) | 2º |

- **O grupo de cima é o mesmo com qualquer régua:** ComBat, Seurat RPCA e
  Harmony (50 dimensões). A ordem interna muda, e as classificações das três
  réguas se correlacionam (ρ 0,84–0,87).
- **O ComBat continua sendo a integração do pipeline.** Fica entre os 3
  primeiros em todas as variantes (1º, 2º, 3º, 2º, 2º) e é o único do grupo que
  devolve a matriz de genes corrigida, de que a NMF precisa. Para análises só
  por embedding ou grafo, o Seurat RPCA é equivalente.
- **O Harmony do pipeline original (15 dimensões)** fica entre o 9º e o 12º
  lugar com qualquer régua: a troca feita no P07 se sustenta.
- **O scGen** cai do 1º lugar (com o Azimuth, o rótulo com que foi treinado)
  para o 6º e o 5º. A vantagem dele no P07 vinha do gabarito. O scANVI segue
  no alto com o atlas (2º).

---

## Como usar o resultado final

O resultado final são cinco arquivos em `dataReseq/`, unidos pela chave
`cell_key` (`library_id|barcode`): o objeto principal, a integração, os
clusters do Leiden, a anotação e os programas de NMF.

```python
import anndata as ad
import pandas as pd

adata = ad.read_h5ad("dataReseq/P05_integracao/reseq_raw_integrado__v2.h5ad")
integ = ad.read_h5ad("dataReseq/P07_benchmark_integracao/integracao_escolhida.h5ad")
rot = pd.read_parquet("dataReseq/P08_clusterizacao/clusters.parquet")

# troca a integração do P05 (Harmony) pela escolhida no P07 (ComBat)
assert (integ.obs["cell_key"].to_numpy() == adata.obs["cell_key"].to_numpy()).all()
adata.obsm["X_emb_combat"] = integ.obsm["X_emb"]      # PCA da matriz corrigida pelo ComBat
adata.obsm["X_umap"] = integ.obsm["X_umap"]
adata.obsp["connectivities"] = integ.obsp["connectivities"]
adata.obsp["distances"] = integ.obsp["distances"]
adata.uns["neighbors"]["params"] = {"method": "umap", "n_neighbors": 15, "use_rep": "X_emb_combat"}

adata.obs = adata.obs.join(rot.drop(columns="library_id").set_index("cell_key"), on="cell_key")

# anotação (P09) e programas de NMF (P10)
anot = pd.read_parquet("dataReseq/P09_anotacao/anotacao.parquet").set_index("cell_key")
prog = pd.read_parquet("dataReseq/P10_nmf_programas/programas.parquet").set_index("cell_key")
adata.obs = adata.obs.join(anot[["anotacao", "grupo", "atlas_L1", "atlas_L2", "atlas_L2_prob"]], on="cell_key")
adata.obs = adata.obs.join(prog[["nmf14_rotulo", "nmf14_fracao", "proj_rotulo"]], on="cell_key")
adata.obsm["X_nmf14"] = prog.loc[adata.obs["cell_key"], [f"H14_P{k}" for k in range(14)]].to_numpy()

# só CD4⁺: sem contaminantes nem baixa qualidade
cd4 = adata[~adata.obs["grupo"].isin(["contaminante", "baixa qualidade"])].copy()
```

**O objeto principal** (`reseq_raw_integrado__v2.h5ad`, 35.247 células ×
26.053 genes):

| onde | o quê |
| --- | --- |
| `layers["counts"]` | contagens corrigidas pelo SoupX (inteiras) |
| `layers["counts_raw"]` | contagens brutas do Cell Ranger |
| `X` | lognorm (normalize_total 10⁴ + log1p) das contagens SoupX |
| `obsm["X_pca"]`, `obsm["X_emb_harmony"]`, `obsm["X_umap"]` | PCA, Harmony e UMAP do P05 (o trecho acima troca pelo ComBat) |
| `obs["baixa_profundidade"]` | os 789 barcodes de 200–500 UMIs que entraram pelo teste de perfil |
| `obs["so_no_padrao"]` | células do Cell Ranger que o MAD por biblioteca descartaria |
| `obs["cellranger"]`, `obs["emptydrops"]`, `obs["cellbender"]` | quem chamou cada barcode |
| `obs["classification"]`, `obs["stimulation"]`, `obs["patient"]` | desenho (corrigido; HNP44 = NP) |

**A integração** (`integracao_escolhida.h5ad`, sem genes):

- `obsm["X_emb"]`: PCA de 50 componentes da matriz corrigida pelo ComBat;
- `obsm["X_umap"]` e o grafo de 15 vizinhos em `obsp`;
- `uns["integracao"]`: o método e o score.

**Os clusters** (`clusters.parquet`):

| coluna | o quê |
| --- | --- |
| `leiden_final` | resolução 1,0 (principal) |
| `leiden_grossa` | resolução 0,1 |
| `leiden_r*` | todas as resoluções testadas (0,2 = escala intermediária) |
| `cluster_tecnico` | marca dos clusters 7, 11, 14 e 15 |

**A anotação** (`P09_anotacao/anotacao.parquet`):

| coluna | o quê |
| --- | --- |
| `anotacao` | rótulo fino por marcadores canônicos (CD4 naive, CD4 Tcm, Treg, …) |
| `grupo` | classe ampla: CD4 naive, CD4 memória, CD4 ativada, Treg, CD4 citotóxica, contaminante, baixa qualidade |
| `atlas_L1`, `atlas_L2`, `*_prob` | tipo do atlas de Yasumizu et al. (2024) e a fração de votos dos 5 vizinhos |
| `azimuth_fino`, `azimuth_grupo` | o Azimuth do Cell Ranger |

**Os programas de NMF** (`P10_nmf_programas/programas.parquet`):

| coluna | o quê |
| --- | --- |
| `nmf14_programa`, `nmf14_rotulo`, `nmf14_grupo` | programa dominante (k = 14) e o seu rótulo |
| `nmf14_fracao` | fração do programa dominante no total da célula (abaixo de 0,4 = mistura) |
| `H14_P0` … `H14_P13` | atividade de cada programa na célula |
| `nmf9_programa` | programa dominante com k = 9 (sensibilidade) |
| `proj_programa`, `proj_rotulo`, `Hproj_*` | projeção nos 12 programas publicados |

**Cuidados na análise:**

- para expressão diferencial e citocinas, use `layers["counts"]`. A matriz
  corrigida pelo ComBat serve só para a vizinhança e a clusterização;
- se o ruído das células pobres pesar, exclua `baixa_profundidade`;
- trate citocinas sempre por biblioteca, porque CTH97_ML concentra boa parte
  do sinal. Para chamar uma célula positiva, exija ≥ 2 UMIs;
- não remova células por Scrublet, Solo ou cxds;
- tire os contaminantes e a baixa qualidade (`grupo`) antes de qualquer
  análise de CD4⁺;
- o programa dominante da NMF é um resumo: para perguntas biológicas, use as
  atividades `H14_*`;
- MB e NP têm um paciente cada: descreva, não teste.

---

## Como refazer

Ambiente principal: `/mnt/SATA/singleCellTaru/.venv` (scanpy 1.11, anndata
0.12, harmonypy 2.0, scvi-tools 1.4 com GPU, scib 1.1.7, scanorama, bbknn,
scgen). Cada notebook é um `.py` (jupytext) convertido e executado por
`build_notebook.sh`. A ordem importa, porque cada etapa lê o que a anterior
gravou.

```bash
cd src/Reseq

# Partes 1 e 2 (~1h30 no total)
for nb in comparativa/C01_websummary_metricas comparativa/C02_antigo_vs_novo \
          comparativa/C03_rarefacao_citocinas analise/A01_preprocessamento \
          analise/A02_preprocessamento_adicional analise/A03_doublets_recuperacao \
          analise/A04_processamento_integracao analise/A05_avaliacao_clusterizacao \
          analise/A06_doublets_solo_cxds; do
  ./build_notebook.sh $nb
done

# Parte 3
python raw/run_cellbender.py                        # ~4–5 h de GPU, 14 bibliotecas
python raw/run_cellbender.py PB20_NS --total-droplets 4000
python raw/run_cellbender.py HNP44_NS --total-droplets 2000
for nb in P01_gotas_vazias P02_cellbender P03_qc_best_practices P04_ambiente_e_doublets \
          P05_integracao P06_balanco_final; do
  ./build_notebook.sh raw/$nb 10800
done
python raw/run_benchmark.py tudo                    # ajustes (~40 min) + métricas (~6–7 h em série)
./build_notebook.sh raw/P07_benchmark_integracao    # só lê o cache do benchmark
./build_notebook.sh raw/P08_clusterizacao 7200

# Parte 4
python raw/run_atlas_mapping.py                     # Symphony no atlas de Yasumizu et al. (~7 min)
./build_notebook.sh raw/P09_anotacao
python raw/run_nmf.py tudo --workers 4              # 57 ajustes de NMF, estabilidade, projeção (~10 min)
./build_notebook.sh raw/P10_nmf_programas
./build_notebook.sh raw/P11_benchmark_clusterizacao 5400
python raw/run_benchmark.py metricas --rotulo anotacao   # ~6–7 h em série por rótulo
python raw/run_benchmark.py metricas --rotulo atlas_L2
./build_notebook.sh raw/P12_benchmark_integracao_rotulos
```

A referência do atlas (`ref_Reference_Mapping_20220525.RData`, Figshare
25052648) e a matriz dos 12 programas (`NMF.W.CD4T.csv.gz`, repositório
NMFprojection) ficam em `HEAVY_OUT/atlas_yasumizu/`.

- As métricas do benchmark são o passo lento (~20 min por execução). O que já
  está no cache não roda de novo. Dá para paralelizar em conjuntos disjuntos
  de execuções, por exemplo `python raw/run_benchmark.py metricas scvi combat`
  em outro terminal.
- `RESEQ_OUT=<pasta>` grava em `<pasta>/dataReseq` e `<pasta>/resultsReseq`
  (modo de teste; nada vai para o projeto).
- `RESEQ_RECOMPUTE=1` força o C03 a refazer a passada pelo `molecule_info.h5`.

**Ambientes externos**, fora do `.venv` e do `renv` do projeto:

| ambiente | caminho | uso |
| --- | --- | --- |
| R 4.5 + DropletUtils, scDblFinder, SoupX, Seurat 5.5, symphony 0.1.3, glmGamPoi (conda) | `~/venvs/reseq-r` (`config.RESEQ_RSCRIPT`) | P01, P04, P07 e o mapeamento no atlas. O R do sistema não tem gfortran |
| CellBender 0.4.0 (Python, GPU) | `~/venvs/cellbender` (`config.CELLBENDER_BIN`) | `raw/run_cellbender.py` |
| R do sistema + scds | `~/R/reseq-scds` | conferência do cxds em A06 |

O scGen 2.1.0 é de 2022 e não importa com o scvi-tools 1.x. Uma ponte em
memória (`benchmark._scgen_compat`) resolve isso sem alterar o pacote
instalado.

As saídas brutas de DropletUtils, CellBender, SoupX, scDblFinder e do
benchmark (≈17 GB) ficam no NVMe, em
`/home/pedrovp/singleCellTaru_claude/reseq_raw/` (`config.HEAVY_OUT`).

---

## Estrutura

```text
src/Reseq/
├── README.md               # este guia
├── build_notebook.sh       # .py (jupytext) → .ipynb executado
├── functions/              # código compartilhado
│   ├── config.py           # caminhos, mapa de nomes, desenho, painéis de genes, parâmetros
│   ├── io.py               # leitores (web summary, 10x, molecule_info) e escrita versionada
│   ├── compare.py          # pareamento antiga × nova
│   ├── rarefaction.py      # rarefação analítica a partir de molecule_info.h5
│   ├── qc.py               # réplica do QC, RNA ambiente
│   ├── doublets.py         # testes da hipótese de doublets; cxds
│   ├── droplets.py         # DropletUtils, CellBender, SoupX, scDblFinder; teste de perfil de célula
│   ├── droplets.R, soupx.R, scdblfinder.R, cxds.R
│   ├── bpqc.py             # QC do sc-best-practices
│   ├── qcplots.py          # plots de QC (versões corrigidas dos do spatools)
│   ├── pipeline.py         # variantes V1–V3, seleção de HVG, métricas de avaliação
│   ├── benchmark.py        # benchmark scIB: ajustes, kBET portado, métricas, agregação, regra de escolha
│   ├── benchplots.py       # figuras no formato do scIB
│   ├── seurat_integra.R    # Seurat v5 IntegrateLayers (CCA, RPCA)
│   ├── anotacao.py         # painel de marcadores com referências; decisão de cada cluster e programa
│   ├── atlas.py, atlas_symphony.R  # mapeamento no atlas de Yasumizu et al. (Symphony)
│   ├── nmf.py              # NMF de programas (receita do artigo), escolha de k, projeção
│   ├── clustbench.py       # métricas do benchmark Leiden × NMF
│   ├── plotting.py, fmt.py # estilo das figuras e números em português
│   ├── issues.py           # registro de problemas por notebook
│   └── legacy.py           # acesso às funções de src/scripts/claude/functions
├── comparativa/            # Parte 1: C01–C03 (.py + .ipynb executado)
├── analise/                # Parte 2: A01–A06
└── raw/                    # Partes 3 e 4: P01–P12 + run_cellbender.py, run_benchmark.py,
                            #   run_atlas_mapping.py, run_nmf.py

dataReseq/                  # objetos (parquet, h5ad) + MANIFEST.jsonl — fora do git
resultsReseq/
├── RELATORIO_reseq.md      # o relatório, com todos os números
├── figures/<notebook>/     # PNG + PDF
└── tables/<notebook>/      # CSV; cada figura tem a sua tabela fig_<nome>.csv,
                            # cada notebook tem o seu issues.csv
```

**Arquivos em `dataReseq/`:**

| arquivo | conteúdo |
| --- | --- |
| `P05_integracao/reseq_raw_integrado__v2.h5ad` | **o objeto principal** (Parte 3) |
| `P07_benchmark_integracao/integracao_escolhida.h5ad` | **a integração escolhida** (ComBat) |
| `P08_clusterizacao/clusters.parquet` | **os clusters do Leiden** |
| `P09_anotacao/anotacao.parquet` | **a anotação** por marcadores, o atlas e o Azimuth, por célula |
| `P10_nmf_programas/programas.parquet` | **os programas de NMF** por célula |
| `P09_anotacao/rotulos_benchmark.parquet` | os rótulos usados no P12 |
| `P07_benchmark_integracao/benchmark.parquet` | scores e métricas das 18 execuções do benchmark |
| `P03_qc_best_practices/qc_candidatos*.parquet` | todo barcode com ≥ 100 UMIs ou chamado: chamadas, métricas e regras |
| `P07_integracao_clusterizacao/clusters__v3.parquet` | clusters anteriores, sobre o Harmony (superados pelo P08) |
| `A04_processamento_integracao/reseq_integrado.h5ad` | o objeto da Parte 2: as variantes V1–V3 sobre o pré-processamento antigo (35.064 células; chaves com sufixo `_V1`, `_V2`, `_V3`) |
| `A01_preprocessamento/new/<biblioteca>.h5ad` | as 14 bibliotecas após o QC original e o Scrublet |
| `A01_preprocessamento/celulas_qc.parquet` | todas as células do Cell Ranger, com o motivo de saída de cada uma |
| `A02_preprocessamento_adicional/flags_adicionais.parquet` | baixo conteúdo, fração intrônica, rótulo Azimuth, por célula |
| `A03_doublets_recuperacao/flags_doublets.parquet` | sinal de célula B, conjugado T–B, Scrublet calibrado, por célula |
| `A06_doublets_solo_cxds/flags_solo_cxds*.parquet` | Solo, cxds e consenso, por célula |
| `C03_rarefacao_citocinas/mol_*.parquet` | resumos do `molecule_info.h5` |

---

## Regras de escrita

- Nada em `data/` é escrito ou apagado, inclusive `data/allOuts`. O código
  recusa gravar fora de `dataReseq/`, `resultsReseq/` e do NVMe.
- Objetos em `dataReseq/` nunca são substituídos: conteúdo igual não é
  regravado, e conteúdo diferente vira `nome__v2`, `nome__v3`… Todo arquivo
  gravado entra no `dataReseq/MANIFEST.jsonl` (append-only), com assinatura,
  notebook de origem e versões dos pacotes. Cada h5ad é relido e conferido
  logo depois de gravado.
- Figuras e tabelas em `resultsReseq/` são regeneradas a cada execução.

## Convenções

- Os nomes de biblioteca são os canônicos, iguais aos da corrida antiga. As
  pastas `MB_34_SN` e `PB_20_SN` são **não estimuladas**; o mapa está em
  `config.RESEQ_TO_CANONICAL`.
- `obs["classificacao"]` preserva a regra original (HNP44 → "MB");
  `obs["classification"]` é a corrigida (HNP44 → "NP").
- Toda figura e tabela declara o número de pacientes e de bibliotecas.
- A corrida antiga é lida de `data/raw/<biblioteca>/` (mtx).
  `data/raw_h5ad/CTH94_NS.h5ad` está corrompido e não é usado.
- Os símbolos de genes vêm da tabela de features do Cell Ranger, não do
  `mygene` (rede).
- A seleção de genes variáveis (`seurat_v3` por biblioteca) é a do scanpy,
  reimplementada em `pipeline.select_hvg`. Assim, o loess de um lote
  numericamente singular é refeito com span maior em vez de abortar. Na
  interseção, o resultado é idêntico ao do scanpy.
- As correções dos plots e do pré-processamento do spatools estão no branch
  `reseq-qc-melhorias` (worktree em `~/spatools-reseq`, commit local, sem
  push). As versões usadas aqui estão em `functions/qcplots.py` e
  `functions/bpqc.py`.

## Limites

- Não há FASTQ nem BAM. As etapas anteriores à matriz (alinhamento, correção
  de barcode, UMIs) só podem ser auditadas pelo que o Cell Ranger registrou
  (C01, C03).
- O teste de perfil de célula tem sensibilidade limitada abaixo de ~300 UMIs.
  Algumas células reais muito pobres ficaram de fora, mas nenhuma teria
  informação suficiente para anotação.
- O benchmark de integração usa um conjunto de dados só, sem as barras de erro
  entre tarefas do artigo. No P07, o rótulo biológico era o Azimuth
  automático; o P12 o refaz com a anotação e com o atlas.
- A clusterização da massa central de células T é um recorte de um contínuo:
  metade das células é mistura de programas (P10).
- O atlas de Yasumizu et al. é de 5′ v2, em doenças autoimunes. Os estados
  próprios do estímulo pelo *M. leprae* não existem nele e caem no tipo mais
  próximo: a concordância com o atlas valida os estados de repouso, não os
  ativados.
- A escolha de k da NMF depende de um desempate (a reprodutibilidade entre
  sementes) que não está no artigo.
- A anotação é por cluster. Dentro do cluster citotóxico há uma minoria de NK,
  e o painel não separa bem memória central e efetora.
