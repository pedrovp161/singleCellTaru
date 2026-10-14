# Ressequenciamento das 14 bibliotecas — avaliação preliminar

As mesmas 14 bibliotecas de células T CD4⁺ — 7 pacientes, cada um com uma
biblioteca estimulada com *M. leprae* (ML) e uma não estimulada (NS) — foram
ressequenciadas com muito mais profundidade (`data/allOuts`, Cell Ranger
10.1.0, 26,1 bilhões de reads). O objetivo era recuperar as citocinas, que
apareciam em poucas células. Este relatório resume o que os arquivos do Cell
Ranger mostram, o que mudou em relação à corrida antiga, se a hipótese de
excesso de doublets se sustenta, e o que acontece quando o pré-processamento e
o processamento originais são aplicados aos dados novos — até a clusterização
dos dados integrados, **sem anotação**.

Código e notebooks: `src/Reseq/` (ver `src/Reseq/README.md`). Figuras e
tabelas: `resultsReseq/figures/<notebook>/` e `resultsReseq/tables/<notebook>/`.
Objeto integrado: `dataReseq/A04_processamento_integracao/reseq_integrado.h5ad`.

> **Limite de inferência.** 7 pacientes, 14 bibliotecas. MB e NP têm um
> paciente cada; 41% das células vêm de um paciente (PB21). Composições por
> grupo clínico são descritivas. O único contraste testado é o pareado
> estimulado × não estimulado (n = 7), e mesmo esse é exploratório.

---

## Resumo

1. **O ressequenciamento não resolveu as citocinas, e mais sequenciamento não
   resolveria.** As bibliotecas estão saturadas: 89–99,5% de saturação e uma
   mediana de 70 reads por molécula. A corrida antiga já tinha 67–96% das
   moléculas de hoje. Com 3,9× mais reads (mediana), cada célula ganhou de +4%
   a +48% de UMIs, menos de +10% em 9 das 14 bibliotecas. Mesmo assim, as
   citocinas continuam em menos de 1% das células, exceto IFNG (3%) e TNF
   (6%). Dobrar a profundidade de novo renderia, no teto, +2,6% de UMIs.
2. **O limite está no experimento, não na leitura.**
   - As células positivas já estavam detectadas: com metade dos reads atuais
     já havia 85–97% delas.
   - O estímulo não aumenta a fração de células com transcrito de citocina na
     maioria dos pacientes. IFNG é maior no ML em 3 de 7 pacientes.
   - A resposta de citocinas está concentrada numa biblioteca, CTH97_ML.
3. **A hipótese de doublets não se sustenta**, nem com Scrublet, nem com Solo,
   nem com cxds. Nas bibliotecas com poucas
   células, a taxa esperada de doublets é ≤ 1,5%. O teto medido para doublets
   T + T é de 0–4,1%, e cresce com o número de células, o oposto do que a
   hipótese prevê. Os únicos doublets em excesso são conjugados T–B, 0,20% dos
   barcodes. As bibliotecas pequenas têm poucas células porque poucas células
   entraram: quase não há material celular fora das células chamadas.
4. **Os defeitos do processamento antigo se repetem quando o código original é
   aplicado aos dados novos.** A causa está no código, não na profundidade:
   - a interseção entre bibliotecas joga fora 12 das 16 citocinas;
   - a PCA é calculada sobre contagens cruas, e os clusters seguem a
     profundidade (o cluster explica 57% da variância do log de UMIs dentro da
     biblioteca, igual à corrida antiga);
   - o Scrublet remove 0,18% das células.

   A variante corrigida **V3** — união de genes com filtro global e PCA sobre a
   lognorm — resolve os dois primeiros e é a linha de base proposta.
5. **A clusterização da V3 é utilizável para a anotação, com três cuidados:**
   - remover os 4 clusters técnicos (601 células de não-T e de células B);
   - tratar as fronteiras da massa central como instáveis (ARI 0,52 entre
     sementes do Leiden, sem resolução natural);
   - lembrar que CTH97_ML continua ocupando uma região própria.

   O agrupamento é o mesmo da corrida antiga, dentro da variação entre
   sementes.
6. **A integração por biblioteca apaga boa parte da diferença entre estimulado
   e não estimulado do mesmo paciente.** A separação vai de 0,61 para 0,22.
   Estímulo e canal estão confundidos neste desenho. O efeito do estímulo deve
   ser testado nas contagens (pseudobulk pareado por paciente), não pela
   presença ou ausência de clusters.

---

## 1. O que foi avaliado

| etapa | notebook | pergunta |
| --- | --- | --- |
| arquivos do Cell Ranger | C01 | o que os web summaries, métricas e logs mostram |
| corrida antiga × nova | C02 | o que mudou, célula a célula |
| rarefação | C03 | mais profundidade recuperaria citocinas? |
| pré-processamento | A01 | réplica exata do `preprocessing.ipynb` (validada nos dados antigos: 14/14 bibliotecas iguais) |
| pré-processamento adicional | A02 | falta algum filtro ou correção? |
| doublets | A03 | a baixa recuperação vem de doublets? (aritmética, régua de RNA, contaminantes, Scrublet) |
| processamento | A04 | réplica do `processing.ipynb` (validada: reproduz `adata_harmony.h5ad` exatamente) e duas correções |
| clusterização | A05 | os clusters são estados celulares ou artefatos? os problemas antigos se repetem? |
| outros detectores | A06 | Solo (scvi-tools) e cxds (scds) mudam a conclusão sobre doublets? |

As três variantes de processamento, sobre as mesmas 35.064 células:

| variante | universo de genes | PCA sobre | clusters (Leiden 0,5) |
| --- | --- | --- | --- |
| V1 fiel (o original) | interseção: 18.852 genes, 4 de 16 citocinas | contagens cruas | 14 |
| V2 | interseção | lognorm escalonada | 10 |
| V3 (proposta) | união + `min_cells=3`: 31.756 genes, 15 de 16 citocinas | lognorm escalonada | 13 |

---

## 2. Problemas encontrados

Gravidade: **crítico** compromete o objetivo do ressequenciamento ou uma
conclusão a jusante; **maior** limita de forma material o que pode ser
analisado; **menor** exige cuidado, mas não muda conclusões.

### 2.1 Profundidade e citocinas

**[crítico] As bibliotecas estão saturadas.** Saturação final de 89,0% a 99,5%
(≥ 95% em 12/14), 70 reads por molécula (mediana; até 217), 7× a 638× a
profundidade mínima da 10x. 90% dos genes por célula já apareciam com 10% dos
reads. A corrida antiga equivalia a 9–32% dos reads atuais e já tinha 67–96%
dos UMIs. A última duplicação de profundidade rendeu +3,1% de UMIs; a próxima
renderia no máximo +2,6% (mediana). *C01, C03.*

**[crítico] A detecção de citocinas não melhorou de forma relevante.** Em 37.092
células pareadas, células positivas antiga → nova:

| citocina | antiga | nova | % das células (nova) |
| --- | --- | --- | --- |
| IFNG | 977 | 1.121 | 3,0% |
| TNF | 1.973 | 2.381 | 6,4% |
| IL2 | 77 | 100 | 0,27% |
| IL4 | 82 | 105 | 0,28% |
| IL10 | 50 | 60 | 0,16% |
| IL13 | 56 | 69 | 0,19% |
| IL17A | 75 | 80 | 0,22% |
| IL17F | 237 | 317 | 0,85% |
| IL21 | 74 | 90 | 0,24% |
| IL22 | 322 | 359 | 0,97% |

95% das células que ganharam positividade têm exatamente 1 UMI. *C02.*

**[crítico] A profundidade não é o limite das citocinas.** Nas 15 citocinas
com pelo menos 10 células positivas, metade dos reads atuais já mostrava 85–97%
das positivas. Dobrar a profundidade acrescentaria, no teto, 330 pares
célula–citocina sobre 5.590. Entre as positivas, 6–33% (mediana 14%) dependem
só de moléculas lidas 1–2 vezes. Com uma definição conservadora (≥ 2 UMIs) as
positivas caem muito: IFNG de 1.098 para 436, IL4 de 104 para 10, IL10 de 58
para 10. *C03, A05.*

**[maior] A resposta de citocinas está concentrada em CTH97_ML.** A biblioteca
tem 13% das células e 56% das IFNG⁺, 91% das IL17F⁺, 80% das IL22⁺, 96% das
IL9⁺ e 82% das IL17A⁺. É também a biblioteca de menor complexidade (1.009
genes por célula, contra 2.894 no par NS). Qualquer análise agregada de
citocinas descreve essencialmente essa biblioteca. *C02, A02.*

**[maior] O estímulo não aumenta a fração de células com transcrito de
citocina na maioria dos pacientes.** IFNG: ML > NS em 3/7 pacientes, p = 0,94
(Wilcoxon exato pareado); TNF: 4/7, p = 0,38; IL2: 4/7, p = 0,94. *C02.*

### 2.2 Captura de células e doublets

**[maior] A recuperação de células continua baixa e muito desigual.**
37.161 → 37.898 células chamadas (+2%). Os 806 barcodes novos têm mediana de
569 UMIs: são barcodes na fronteira do limiar, não células recuperadas. Após o
QC ficam de 70 (HNP44_NS) a 10.629 células (PB21_ML), uma razão de 152×; 41%
das células vêm de PB21. Cinco bibliotecas têm menos de 1.000 células. *C01,
C02, A01.*

**[maior] Os dados não sustentam a hipótese de doublets.** Quatro linhas de
evidência independentes (A03):

| evidência | resultado |
| --- | --- |
| aritmética do carregamento | taxa esperada ≤ 1,5% nas bibliotecas pequenas; doublets reduziriam a contagem em no máximo 2× |
| régua de RNA (mistura em log-UMI) | teto de doublets T + T de 0–4,1% nas bibliotecas pequenas, correlação de 0,87 com o nº de células — o esperado do carregamento, o oposto da hipótese |
| contaminantes B como segunda "espécie" | 70 doublets T + B validados, 4,4× o acaso, 0,20% dos barcodes; a fração não depende do carregamento (27% nas bibliotecas pequenas, 26% nas grandes) — conjugados pré-formados |
| material fora das células | as seis menores bibliotecas quase não têm barcodes de material nuclear nem RNA livre: entrou pouca célula |

**[maior] O Scrublet é cego nesta população.** O threshold automático chama
0,18% das células (62 de 35.126, contra ~1.554 esperadas pela taxa do 10x) e
pega 1 dos 70 doublets T + B validados; calibrado pela taxa esperada, pega 19.
*A01, A03.*

**[maior] Solo e cxds também não sustentam a hipótese (A06).** Os dois métodos
foram testados a pedido, contra as mesmas referências: os 70 doublets T + B
validados, 196 células B sozinhas como controle negativo, o conteúdo de RNA e
a taxa esperada do 10x.

| | Scrublet | Solo (scvi-tools) | cxds (scds) |
| --- | --- | --- | --- |
| separação dos T + B validados (AUC) | 0,70 | 0,82 | 0,73 |
| validados pegos no limiar calibrado | 19/70 | 20/70 | 21/70 |
| chamadas no limiar padrão | 0,18% | 14,7% (7–34% por biblioteca; 100% em HNP44_NS) | — (sem limiar próprio) |
| correlação do score com os UMIs | 0,30 | 0,75 | 0,74 |

- O Solo chama a mesma fração em bibliotecas pequenas e grandes (mediana de
  12,2% e 11,0%; ρ = −0,29 com o número de células). Sob a hipótese, as
  pequenas deveriam ter mais.
- O limiar padrão do Solo marca 14,7% das células, muito acima do que o
  carregamento permite (≤ 1,5% nas bibliotecas pequenas). O score acompanha o
  conteúdo de RNA; por isso o Solo **não deve ser usado para remover
  células**.
- O cxds, nesta população, não procura linhagens. Os pares que ele considera
  mutuamente exclusivos opõem genes de programas de célula T, como IGF1R ×
  PHACTR2 e S100A4. Ele marca células que coexpressam os dois lados, sem
  separar um doublet de uma célula em estado intermediário.
- Os três concordam pouco: Jaccard de 0,20 a 0,40 entre as chamadas
  calibradas. O consenso (≥ 2 de 3) tem 1.180 células (3,4%) e 20 dos 70
  validados. Fica como coluna de sensibilidade
  (`dataReseq/A06_doublets_solo_cxds/flags_solo_cxds__v2.parquet`), não como
  filtro.

**[maior] Material nuclear de células mortas em três bibliotecas NS.**
CTH97_NS, PB21_NS e MB34_NS têm, respectivamente, 4.873, 21.097 e 1.289
barcodes com 100–499 moléculas, ≥ 50% intrônicas. Em PB21_NS são 4,9 por
célula; nas demais bibliotecas, no máximo 46 no total. Os genes no topo são os
mesmos nas três: SLC26A3, STEAP1B, FOXP1, GPM6A… Indica morte celular antes da
captura. *A02.*

### 2.3 Qualidade das bibliotecas

- **[maior] Fração de reads em células abaixo do ideal** em 5 bibliotecas
  (53–69,5%; alarme do Cell Ranger), correlacionada com o número de células
  (ρ = 0,93). Apesar disso, a contaminação por RNA ambiente dentro das células
  é baixa (ρ de 0,2% a 3,3%, mediana 0,8%) e não precisa de correção para o
  clustering. Para citocinas, 1 UMI não deve contar como positividade na
  biblioteca em que o gene é abundante (em CTH97_ML o ambiente explica boa
  parte dos eventos de 1 UMI). *C01, A02.*
- **[maior] CTH92_ML e CTH97_ML têm complexidade muito baixa** (1.832 e 1.009
  genes por célula, contra 3.121 e 2.894 nos pares NS), com saturação de
  ~99%. CTH97_ML também tem fração intrônica de 8%, contra 14–40% nas demais.
  Diferenças estimulado × não estimulado nesses pacientes confundem biologia
  com eficiência de captura. *C01, A02.*
- **[menor] Reads antisense e intrônicos altos** (antisense > 20% em MB34_NS,
  HNP44_NS e HNP44_ML; íntrons de 13% a 44%) e **fração de TSO elevada** em 6
  bibliotecas (12–22%). *C01.*
- **[menor] Um grupo de 988 genes é sustentado quase só por moléculas
  intrônicas lidas 1–2 vezes** (LINC00486, SLC26A3, STEAP1B, GPM6A…; 41% delas
  em PB21_NS). Não devem ser usados como marcadores; 83 entram entre os genes
  variáveis da V3. O controle negativo confirma o perfil: os genes do Y em
  doadoras somam 15.761 moléculas, 97,8% delas com 1–2 reads e só 0,3%
  exônicas. *C03.*
- **[menor] Parte das detecções de 1 UMI da corrida antiga não se repete**
  (5,4% das entradas). Há também 85 genes que tinham milhares de UMIs na
  corrida antiga e quase somem na nova, entre eles TOX2: 3.219 → 1.058 células
  positivas. Assinaturas que usam esses genes devem ser conferidas contra
  `tables/C02_antigo_vs_novo/genes_que_colapsaram.csv`. *C02.*
- **[verificado]** Mesmas bibliotecas, sem troca de amostra (sobreposição de
  barcodes de 0,991 a 1,000), sexo dos doadores consistente; sem *index
  hopping*; erros de UMI e de barcode explicam só uma minoria das moléculas de
  baixo suporte. *C02, C03.*

### 2.4 Processamento

- **[crítico] A interseção entre bibliotecas remove as citocinas.** O
  `concat(join="inner")` depois de `filter_genes` por amostra mantém 18.852
  genes e perde 12 das 16 citocinas (IL2, IL4, IL9, IL10, IL13, IL17A, IL17F,
  IL21, IL22, CSF2, IL31, IL3). A culpa não é da biblioteca pequena: tirar
  HNP44_NS recupera 8% dos genes e nenhuma citocina. A solução é a união com um
  filtro global, `min_cells=3` (V3): 31.756 genes, perde só IL3 (1 célula).
  Como todas as bibliotecas usam a mesma referência, gene ausente é zero de
  verdade. *A02, A04.*
- **[crítico] A PCA do original roda sobre contagens cruas**, e o defeito se
  repete nos dados novos (V1). 50 componentes capturam 98,1% da variância;
  92% da variância da matriz está em 10 genes, entre eles MALAT1 e três
  imunoglobulinas de poucas células B. As três primeiras dimensões do Harmony
  acompanham o número de UMIs (|ρ| 0,74, 0,70, 0,43). Com a PCA corrigida
  (V2, V3): 17–18% em 50 componentes, e só a primeira dimensão correlaciona
  (0,57). *A04.*
- **[maior] Os clusters do original não sobrevivem à correção da PCA**: ARI
  V1 × V2 = 0,08, como na corrida antiga (0,09). Os clusters da V1 não devem
  ser anotados. *A04.*
- **[maior] O Harmony por biblioteca também corrige o estímulo.** A separação
  entre as duas bibliotecas do mesmo paciente na vizinhança vai de 0,61 (antes)
  para 0,22 (Harmony por biblioteca). Com Harmony por paciente fica em 0,63, e
  os pacientes se misturam quase tanto quanto (iLISI 2,60 contra 2,84). *A04.*
- **[menor] A seleção de genes variáveis do scanpy falha no universo
  completo.** O erro de loess aparece em 3 bibliotecas pequenas;
  `pipeline.select_hvg` refaz só esses lotes com span maior e é idêntica ao
  scanpy na interseção. Entre os 2.000 genes variáveis da V3 há 118 técnicos
  (10 segmentos V de TCR, 22 imunoglobulinas, 83 de reads isolados, 3 de sexo).
  Tirá-los muda pouco a estrutura (53% dos vizinhos em comum, mais que entre V2
  e V3) e é um refinamento recomendado para a anotação. *A02, A04.*
- **[menor] Pré-processamento.** O filtro MAD não corta nada em CTH92_ML (o
  limiar fica abaixo do piso do Cell Ranger). As 730 células de baixo conteúdo
  (< 1.000 UMIs, só em CTH92_ML e CTH97_ML) não formam cluster próprio, e a
  fração intrônica não aponta células danificadas no objeto. Não é necessário
  pré-processamento adicional para a integração. *A01, A02, A05.*

### 2.5 Clusterização (V3)

- **Estrutura compartilhada.** Os 9 maiores clusters têm células de todos os
  pacientes (número efetivo de 3,3 a 6,8 de 7, mediana 5,6).
- **[menor] Quatro clusters técnicos**: 8 (282 células: células B e conjugados
  T–B), 10, 11 e 12 (126, 100 e 93 células, não-T). Somam 601 células (1,7%)
  e devem sair antes da anotação. O 10 é também o único cluster de um paciente
  só (85% de CTH97).
- **[maior] Fronteiras instáveis.** ARI entre sementes do Leiden de 0,52; de 13
  a 14 clusters conforme a semente; de 7 a 20 clusters entre as resoluções 0,1
  e 1,0, sem patamar. Seis clusters são estáveis (Jaccard ≥ 0,75: 5, 7, 8, 9,
  10 e 12); o cluster 2 é um recorte arbitrário (0,41). A resolução 0,5 é
  herdada, não indicada pelos dados.
- **Integração.** A fração de vizinhos da mesma biblioteca cai de 50% para 10%
  (mediana). CTH97_ML continua com 48% de vizinhos próprios contra 13%
  esperados.
- **Profundidade.** O cluster explica 22% da variância do log de UMIs dentro
  da biblioteca, contra 57% na V1.
- **O ressequenciamento não mudou o agrupamento.** O ARI entre a corrida antiga
  e a nova, processadas pelo mesmo código, é 0,52, igual ao ARI entre duas
  sementes; 75% das células caem no cluster pareado.
- **Estímulo, descritivo.** O cluster 6 tem mais células nas bibliotecas
  estimuladas nos 7 pacientes (p exato 0,016; 0,13 após correção para 13
  clusters). Ele concentra as células positivas para IFNG (16% do cluster),
  TNF, IL22 e IL17F, e 48% das suas células vêm de CTH97_ML. Hipótese para a
  anotação, não resultado.

### 2.6 Arquivos e nomes

- **[menor] `data/raw_h5ad/CTH94_NS.h5ad` está corrompido.** Um trecho contíguo
  de 28.672 bytes (7 blocos de 4 KiB) está zerado: são 7.168 entradas e 9.406
  UMIs a menos em 25 genes do cromossomo 16. A fonte confiável é
  `data/raw/CTH94_NS`. Convém regerar o arquivo e verificar a saúde do disco
  (NTFS). Esta análise lê sempre `data/raw`. *C02.*
- **[menor] Duas pastas de `data/allOuts` usam o sufixo SN** (`MB_34_SN`,
  `PB_20_SN`). A regra original `endswith("NS")` as rotularia como estimuladas
  sem dar erro. O código usa nomes canônicos (`config.RESEQ_TO_CANONICAL`).
  *C01.*
- **[info]** As corridas não geraram BAM (`--create-bam=false`): não há como
  reconstruir TCR nem genotipar, então doublets só podem ser avaliados por
  evidência indireta. *C01.*

---

## 3. Os problemas da análise anterior se repetem?

| problema | corrida antiga | ressequenciamento | veredito |
| --- | --- | --- | --- |
| baixa detecção de citocinas | motivo do ressequenciamento | IFNG 2,6% → 3,0%; IL4 0,22% → 0,28%; IL10 0,13% → 0,16%; +8% de UMIs por célula (mediana) | **repete** — não é profundidade |
| citocinas removidas pela interseção | IL4, IL9, IL10, IL13, IL17A, IL17F, IL21 fora | 12 de 16 fora na V1/V2; 1 na V3 | **repete na V1 · resolvido na V3** |
| PCA sobre contagens cruas | 98,2% da variância em 50 PCs | 98,1% na V1; cluster explica 57% dos UMIs (V1) × 22% (V3) | **repete na V1 · resolvido na V2/V3** |
| clusters dependentes do defeito | ARI original × corrigido 0,09 | ARI V1 × V2 0,08 | **repete** |
| Scrublet quase não remove | 38 de 34.969 (0,11%) | 62 de 35.126 (0,18%); Solo e cxds também não indicam excesso | **repete** — e não há excesso de doublets a remover |
| baixa recuperação e desbalanço | 37.161 células; HNP44_NS com 76 | 37.898 células; HNP44_NS com 70; 152× | **repete** — limitação do carregamento |
| células B no objeto de T | 29 células | 4 clusters técnicos, 601 células | **repete** — remover na anotação |
| `classificacao` com HNP44 → "MB" | regra da célula 037 | preservada em `classificacao`; corrigida em `classification` | registrado |

---

## 4. O que fazer a partir daqui

1. **Não sequenciar mais estas bibliotecas.** Mais sinal de citocina exige
   mais células respondedoras: mais células por captura, enriquecimento
   (sorting por marcador de ativação, captura de células secretoras) ou leitura
   direcionada. Também convém rever o estímulo (dose, tempo até a coleta,
   presença de células apresentadoras).
2. **Usar o objeto integrado da V3** (`reseq_integrado.h5ad`, chaves padrão)
   como linha de base. A V1 fica no objeto só como registro do original.
3. **Antes de anotar:**
   - remover os clusters técnicos (8, 10, 11, 12) e os barcodes com sinal de B;
   - não remover células por Scrublet, Solo ou cxds; levar o consenso dos três
     como coluna de sensibilidade;
   - considerar tirar os 118 genes técnicos da seleção de genes variáveis;
   - escolher a resolução por marcadores, conferindo a estabilidade de cada
     cluster interpretado.
4. **Citocinas:** reportar sempre por biblioteca. Para chamar uma célula
   positiva, exigir ≥ 2 UMIs ou uma molécula com ≥ 3 reads.
5. **Estímulo:** testar por pseudobulk pareado por paciente, sobre as
   contagens. Se a composição por cluster for usada, repetir a integração com
   Harmony por paciente e comparar.
6. **Estatística no nível de biblioteca e paciente.** HNP44_NS (70 células)
   fica fora de composição e pseudobulk.
7. **Em novas capturas:** contar células e medir viabilidade imediatamente
   antes de carregar; considerar remoção de células mortas e multiplexação
   (hashing), que mede doublets diretamente.

## 5. Perguntas que só o laboratório pode responder

- Quantas células foram carregadas por canal, e com que viabilidade? Com isso
  dá para calcular a eficiência de captura de cada biblioteca e fechar a
  questão da baixa recuperação.
- O ressequenciamento usou as mesmas bibliotecas finais ou uma nova preparação
  a partir do cDNA? A corrida antiga não é uma subamostra exata dos reads
  atuais: moléculas bem sustentadas hoje faltavam nela (11–30 reads: 18% ×
  6% previstos).
- Houve alguma diferença de protocolo na amostra CTH97_ML (estímulo, tempo,
  manuseio)?
- Regerar `data/raw_h5ad/CTH94_NS.h5ad` a partir de `data/raw/CTH94_NS` e
  verificar o disco.
