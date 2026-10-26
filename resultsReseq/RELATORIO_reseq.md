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
Objeto principal: `dataReseq/P05_integracao/reseq_raw_integrado__v2.h5ad`, com a
integração do P07 e os clusters do P08 (seção 4). O
`dataReseq/A04_processamento_integracao/reseq_integrado.h5ad` é o registro das
variantes sobre o pré-processamento antigo.

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
7. **Partir da matriz raw quase não recupera células com conteúdo útil
   (P01–P06).** Sem o piso de 500 UMIs do Cell Ranger, o emptyDrops chama
   38.005 barcodes a mais. Com controle de profundidade, só ~2.300 têm perfil
   de célula, quase todos abaixo de 500 UMIs. O padrão adotado é um **valor
   único de 200 UMIs** para chamada e QC em todas as bibliotecas:
   - fora do Cell Ranger, só entram barcodes com perfil de célula;
   - saem os doublets do scDblFinder calibrados pela taxa do 10x e os T + B
     validados.

   Resultado: **35.247 células** (antigo: 35.064), e HNP44_NS passa de 70
   para **111**. As 789 células de baixa profundidade formam um cluster
   técnico e ficam marcadas.
8. **No benchmark de integração no formato do scIB (P07), o ComBat vence o
   Harmony.** Foram 10 métodos e 18 execuções, com as métricas e a agregação
   de Luecken et al. (2022). O ComBat sobre a lognorm escalonada teve o maior
   score geral entre os métodos que não usam rótulo (0,566), em 1º sob as 6
   formas de agregar. Ele preserva 53% da separação NS × ML, contra 35% do
   Harmony usado até aqui, que ficou em 9º de 14. A clusterização foi refeita
   sobre ele (P08): 19 clusters na resolução 1,0, dos quais 4 são técnicos.
9. **Anotação e clusterização alternativa (P09–P11).**
   - **Anotação:** por marcadores canônicos, a amostra é 33% CD4 naive, 37%
     memória (Tcm, Tfh-like, Th17, Tem GZMK⁺), 9% ativadas pelo TCR, 5% Treg e
     4% citotóxicas. Há 1,2% de contaminantes (B/plasmócito, monócito,
     progenitor CD34⁺, basófilo) e 11% de células de baixa qualidade. A
     anotação concorda em 79% com o atlas CD4⁺ de Yasumizu et al. (2024),
     mapeado com o Symphony sem usar a nossa integração.
   - **NMF:** o método do mesmo artigo (programas gênicos de NMF) supera o
     Leiden na comparação com réguas independentes. Concorda mais com o atlas
     (ARI 0,23 contra 0,14), é muito mais estável (0,94 contra 0,47 em
     subamostras) e separa a resposta a interferon e a Treg efetora. Metade das
     células, porém, é mistura de programas.
10. **O benchmark de integração refeito com rótulos melhores (P12)** mantém o
    ComBat no grupo de cima, ao lado do Seurat RPCA e do Harmony com 50
    dimensões, a poucos centésimos um do outro. A regra do P07 escolheria o
    RPCA com a anotação e com o atlas. O ComBat continua: fica entre os 3
    primeiros em todas as variantes e é o único do grupo que devolve a matriz
    de genes que a NMF exige. O Harmony de 15 dimensões fica sempre para trás,
    e a vantagem do scGen no P07 vinha do rótulo com que foi treinado.

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
| a partir da matriz raw | P01–P06 | gotas vazias (emptyDrops, CellBender), QC do best practices, SoupX, scDblFinder, valor único de corte |

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

## 4. Pré-processamento a partir da matriz raw (P01–P08)

Refeito a partir de `raw_feature_bc_matrix.h5`, seguindo os capítulos *raw
data processing* e *quality control* do
[sc-best-practices](https://www.sc-best-practices.org/introduction/raw-data-processing/).
Sem FASTQ nem BAM, as etapas anteriores à matriz (alinhamento, correção de
barcode, UMIs) só podem ser auditadas pelo que o Cell Ranger registrou (C01,
C03).

**Gotas vazias (P01, P02).**
- O `emptyDropsCellRanger` (DropletUtils) reproduz a chamada do Cell Ranger:
  37.783 de 37.898 células.
- O `emptyDrops` sem o piso de 500 UMIs acrescenta **38.005** barcodes, com
  mediana de 153 UMIs; 49% deles têm perfil nuclear. PB21_NS sozinha soma
  21.388.
- O CellBender (Python, GPU) concorda com o Cell Ranger e, como o emptyDrops,
  toma o material nuclear de PB21_NS por célula (+18.584).

**Os extras parecem célula?**
- O teste compara o perfil de cada barcode com o das células e com o do
  ambiente, usando dois controles na mesma profundidade: células reais
  subamostradas e gotas vazias somadas. Uma célula reduzida a ~170 UMIs
  continua reconhecível.
- Por esse critério, cerca de 2.260 dos 38.005 extras têm perfil de célula, quase
  todos abaixo de 500 UMIs.
- Em HNP44_NS, os 732 extras (mediana de 170 UMIs, contra 18.121 nas células)
  ficam no nível das gotas vazias. Eles ainda trazem genes do Y numa doadora e
  os genes de moléculas espúrias do C03.

**O valor único (P03).** Com os limiares do capítulo calculados nas células do
Cell Ranger, o piso de chamada quase não pesa. O que decide é o limite
inferior de 5 MADs por biblioteca, que vai de ~150 a ~3.800 UMIs. Um mínimo
fixo, o mesmo para chamada e QC em todas as bibliotecas, antes dos doublets:

| mínimo de UMIs | células (todos os barcodes acima do mínimo) | extras ao Cell Ranger | extras com perfil de célula |
| --- | --- | --- | --- |
| **200 (adotado, só extras com perfil)** | 39.544 → **36.979** | 3.354 → 789 | 789 |
| 300 | 37.936 | 1.746 | 471 |
| 500 | 36.940 | 750 | 17 |
| 1.000 | 35.514 | 499 | 8 |
| 5 MADs por biblioteca | 35.671 | 743 | 227 |

O padrão adotado é **200 UMIs**, com três regras:
- **células do Cell Ranger** entram se passarem nas demais regras do
  capítulo: limites superiores, top-20, MT em logit + 8% e o filtro de
  material nuclear;
- **barcodes fora do Cell Ranger** só entram com perfil de célula pelo teste
  de P01. São 789, marcados em `obs["baixa_profundidade"]`, com mediana de
  327 UMIs e 253 genes;
- **o teste de perfil não é aplicado às células do Cell Ranger.** Ele foi
  calibrado para barcodes de ~150 UMIs e compara com o perfil médio da
  biblioteca. Aplicado a elas, reprovaria **8.512 células reais** (mediana de
  7.242 UMIs), concentradas em alguns clusters (até 75% de um cluster) e mais
  positivas para citocina que as demais (11,2% contra 9,2%).

O padrão também mantém **1.262 células do Cell Ranger** que o MAD descartava
nas bibliotecas profundas (mediana de 1.349 UMIs e 763 genes). Elas se
espalham pelos clusters.

**As células de baixa profundidade não se misturam.** 680 das 789 formam,
com outras células pobres, um **cluster técnico** (cluster 7 de P05: 1.366
células, mediana de 504 UMIs e 342 genes, quase sem citocinas). Na anotação,
ele deve ser marcado como técnico. Sem ele, o objeto tem ~34 mil células com
conteúdo útil.

> **Cuidado com o QC literal.** Recalcular os limiares sobre candidatos não
> filtrados desloca tudo. Em HNP44_NS, 723 barcodes "passariam". Com piso de
> 100 UMIs, ~8.000 células reais sairiam como outliers superiores.

**Ambiente e doublets (P04).**
- **SoupX:** ρ fica no piso da busca (1%) na maioria das bibliotecas, ou
  seja, ≤ 1%. A correção tira sobretudo eventos de 1 UMI de citocina em
  CTH97_ML.
- **scDblFinder:** com o limiar padrão, marca 7,6% das células, contra ~4,5%
  esperados, nos clusters de mais RNA.
- **Removidos 1.732 doublets:** o limiar do scDblFinder calibrado pela taxa
  do 10x, por biblioteca, mais os T + B validados. Todas as variantes testadas
  tiram células com ~2× a mediana de UMIs; não dá para separar doublet de
  célula grande.

**Balanço (P06).**

| | antigo (A01) | novo |
| --- | --- | --- |
| células finais | 35.064 | **35.247** |
| de onde vem a diferença | — | +1.262 do Cell Ranger que o MAD descartava, +789 de baixa profundidade, −1.732 doublets (950 em PB21_ML) |
| HNP44_NS | 70 | **111** |
| bibliotecas pequenas | — | +2% a +10% |
| genes (filtro global ≥ 20 células) | — | 26.053; 14 de 16 citocinas |
| concordância com a V3 de A04 | — | ARI 0,56 (entre sementes: 0,55) |

Objeto: `dataReseq/P05_integracao/reseq_raw_integrado__v2.h5ad`:
- contagens SoupX em `layers["counts"]` e brutas em `layers["counts_raw"]`;
- `X` = lognorm, com a V3 (PCA, Harmony, UMAP, `leiden`);
- marcas em `obs`: `baixa_profundidade`, `so_no_padrao` e `scdbl_doublet`
  (limiar padrão, só informativo).

O `reseq_raw_integrado.h5ad` sem sufixo é a versão anterior, com o padrão de
500 UMIs e os doublets só marcados, preservada.

### Benchmark de métodos de integração (P07)

O Harmony por biblioteca do P05 foi posto contra os métodos do benchmark de
Luecken et al. (2022, *Nat Methods* 19:41–50), com o pacote de métricas do
artigo (`scib` 1.1.7), a mesma agregação (min-max; 0,4 × lote + 0,6 × bio) e
figuras no mesmo formato.

**Métodos.** Não integrado, Harmony, Scanorama (embedding e matriz), BBKNN,
ComBat, scVI, scANVI\*, scGen\*, Seurat v5 CCA e RPCA. Todos usam as mesmas
2.000 HVG, com e sem escala quando o método aceita: 18 execuções.

**Adaptações:**
- o rótulo biológico é o Azimuth do Cell Ranger, agrupado em 7 classes, já
  que não há anotação;
- lote = biblioteca;
- uma métrica do estudo, a separação NS × ML preservada, veta quem apaga o
  estímulo mais que o Harmony atual;
- scANVI e scGen usam o rótulo no treino e não são elegíveis;
- o kBET foi portado para Python.

**Resultado:**

| posição (elegíveis) | execução | geral | lote | bio | estímulo preservado |
| --- | --- | --- | --- | --- | --- |
| **1** | **ComBat, escalado** | **0,566** | 0,598 | 0,544 | **53%** |
| 2 | Seurat v5 RPCA | 0,518 | 0,506 | 0,525 | 41% |
| 3 | Seurat v5 CCA | 0,505 | 0,488 | 0,516 | 28% |
| 4 | scVI | 0,501 | 0,683 | 0,379 | 64% |
| 5 | Harmony, escalado (50 dimensões) | 0,498 | 0,543 | 0,468 | 35% |
| 9 | Harmony do P05 (15 dimensões) | 0,448 | 0,474 | 0,432 | 35% |

- O ComBat remove toda a variância da PCA explicada pela biblioteca (PCR
  1,00) e tem o maior ARI de todas as execuções (0,51).
- Ficou em 1º sob as 6 formas de agregar, entre elas z-score, pesos iguais e
  sem as métricas de rótulo isolado (que aqui medem um grupo de 42 células).
- Só scGen\* (0,766) e scANVI\* (0,652), que usam o rótulo, ficaram acima.
- Os dois BBKNN foram vetados: misturam as duas bibliotecas de cada paciente
  além do acaso.

A tarefa é simples: um tipo celular separado por citometria, em bibliotecas de
composição parecida. Nesse cenário, um modelo linear por gene basta. Isso é
coerente com os benchmarks anteriores citados pelo artigo, em que o ComBat ou o
Harmony superavam os métodos não lineares em tarefas simples.

### Clusterização sobre o ComBat (P08)

**Mistura.**
- Os vizinhos da mesma biblioteca caem de 41% para 17%; com o Harmony, eram
  11%.
- O número efetivo de pacientes vai de 1,4 para 2,6; com o Harmony, 2,9.
- A separação NS × ML fica em 0,33; com o Harmony, 0,22.

**Resolução.**

| escala | resolução | clusters | ARI entre sementes |
| --- | --- | --- | --- |
| **fina (principal)** | 1,0 | 19 | 0,58 |
| intermediária | 0,2 | 6 | 0,79 |
| grossa | 0,1 | 4 | 0,98 |

Entre 0,6 e 1,0 o ARI quase não muda; só 8 dos 19 clusters são estáveis
entre sementes.

**Clusters técnicos na escala fina** (3.136 células):
- 7 (1.855 células) e 11 (914): baixo conteúdo de RNA; no 11, 60% são de baixa
  profundidade;
- 14 e 15: não-T.

Sem eles, a profundidade explica 22% da variância de UMIs dentro da biblioteca.

Os clusters 4, 7, 10 e 13 têm mais células nas bibliotecas estimuladas em 6 de
7 pacientes. Nenhum resiste à correção, e o 10 é 74% de CTH97. São hipóteses
para a anotação.

**Arquivos:**
- integração: `dataReseq/P07_benchmark_integracao/integracao_escolhida.h5ad`
  (embedding, grafo e UMAP do ComBat);
- rótulos: `dataReseq/P08_clusterizacao/clusters.parquet` (`leiden_final`,
  `leiden_grossa`, `leiden_r*`, `cluster_tecnico`);
- os dois se unem ao objeto principal pela chave `cell_key`. O trecho de
  código está em `src/Reseq/README.md`.

Os rótulos anteriores, sobre o Harmony, continuam em
`dataReseq/P07_integracao_clusterizacao/clusters__v3.parquet`.

### Anotação, programas de NMF e o benchmark refeito (P09–P12)

**Régua independente.** O artigo indicado para a clusterização alternativa é
Yasumizu et al. (2024, *Cell Genomics* 4:100473), um atlas de células T CD4⁺
do sangue. As 33.999 células que o Azimuth chama de CD4⁺ foram mapeadas na
referência pública do atlas com o Symphony (lote = biblioteca), sem usar a
nossa integração nem os nossos clusters. Ficaram 5 tipos L1 e 18 L2 por
célula. Dos 1.271 genes da referência, 1.177 estão nos dados (o atlas é 5′ v2).

**Anotação (P09).** Painel de marcadores canônicos com referências (linhagem,
naive/memória, Treg, Th17, Tfh, citotóxica, ativação e contaminantes),
genes diferenciais e o atlas como conferência:

| classe | clusters | células |
| --- | --- | --- |
| CD4 naive | 0, 2, 12 | 33,4% |
| CD4 memória (Tcm, Tcm/Tfh, transição, Th17, Tem GZMK⁺) | 1, 3, 5, 6, 16 | 36,8% |
| CD4 ativada pelo TCR | 4, 13 | 9,0% |
| Treg | 8 | 4,9% |
| CD4 citotóxica | 9 | 3,7% |
| contaminantes (B/plasmócito, monócito, progenitor CD34⁺, basófilo) | 14, 15, 17, 18 | 1,2% |
| baixa qualidade | 7, 10, 11 | 11,1% |

- Concordância com o L1 do atlas: 79,2% (κ 0,65). Em naive e Treg, 94,7%,
  contra 65,7% do Azimuth automático.
- O Tem do atlas cai quase todo em Tcm: o painel não separa memória central
  e efetora fora do eixo CCR7.
- As classes ativadas aumentam com o estímulo em 6 de 7 pacientes, e o Th17
  em repouso diminui. É descritivo.

**Programas de NMF (P10).** É a alternativa do artigo ao Leiden. Os 18 tipos
do atlas foram definidos com Leiden, mas, para descrever o CD4⁺, os autores
usam programas gênicos de NMF, porque delimitar clusters num contínuo seria
artificial.

- **A receita:** regressão de UMIs, %MT e ciclo; escala com os negativos
  zerados; `sklearn` NMF. Aplicada aos dados corrigidos pelo ComBat.
- **O número de programas:** os critérios do artigo só limitam por baixo
  (k > 7). O desempate foi a reprodutibilidade entre sementes, que dá k = 14,
  com k = 9 como sensibilidade.
- **Os programas:** naive, naive com interferon, transição, Tcm, Tfh-like,
  Th17, ativação, Treg, Treg efetora (HLA-DR⁺ CD39⁺), citotóxico, um programa
  técnico (MALAT1/NEAT1) e três de contaminantes.
- **O contínuo:** em 50,5% das células, o programa dominante tem menos de 40%
  do total.
- **Os programas publicados** projetam bem (POH 0,20), mas, como clusters,
  quase tudo cai no Naive-F.

**Leiden × NMF (P11).** Cinco partições, com métricas de concordância com as
réguas independentes, robustez, peso técnico, especificidade de marcadores e
anotação:

| | Leiden 1,0 (anterior) | Leiden 0,6 | **NMF k = 14 (novo)** | NMF k = 9 |
| --- | --- | --- | --- | --- |
| ARI com o atlas L2 | 0,14 | 0,15 | **0,23** | 0,24 |
| ARI em subamostras de 80% | 0,47 | 0,41 | **0,94** | 0,98 |
| η² da profundidade | 0,59 | 0,26 | **0,33** | 0,07 |
| AUROC dos marcadores | 0,77 | 0,80 | **0,77** | 0,78 |
| anotação × atlas L1 | 0,79 | — | **0,80** | — |
| score geral | 0,28 | 0,48 | **0,65** | 0,79 |

- A NMF supera o Leiden mesmo com o mesmo número de grupos.
- O Leiden 0,6 tem os marcadores mais específicos.
- A robustez da NMF entre sementes (0,998) é alta por construção, porque foi
  o critério da escolha de k. A das subamostras não é.

**Benchmark de integração com os rótulos novos (P12).** As 18 execuções do P07
foram reavaliadas com a anotação do P09 e com o tipo L2 do atlas no lugar do
Azimuth. As métricas, a agregação e a regra são as mesmas. A agregação
principal deixa de fora as métricas de rótulo isolado, porque o rótulo isolado
das réguas novas tem 14–18 células e nenhuma integração o isola.

| régua | maior score elegível | escolhido pela regra | ComBat entre os elegíveis |
| --- | --- | --- | --- |
| Azimuth (P07) | ComBat (0,566) | ComBat | 1º |
| anotação (P09) | Seurat RPCA (0,717) | Seurat RPCA | 2º (0,695) |
| atlas L2 | Harmony 50 dims (0,726) | Seurat RPCA (empate com o Harmony; mais estímulo) | 3º (0,696) |
| atlas L2, agregação completa | Harmony 50 dims | ComBat (empate; mais estímulo) | 2º |

- O grupo de cima é o mesmo com as três réguas (ρ 0,84–0,87 entre as
  classificações).
- O ComBat segue como integração do pipeline, porque a NMF precisa de matriz
  de genes, e o Seurat v5 e o Harmony devolvem só embedding.
- O Harmony de 15 dimensões fica entre o 9º e o 12º lugar.
- O scGen cai do 1º para o 5º/6º lugar quando a régua não é o rótulo com que
  foi treinado.

## 5. O que fazer a partir daqui

1. **Não sequenciar mais estas bibliotecas.** Mais sinal de citocina exige
   mais células respondedoras: mais células por captura, enriquecimento
   (sorting por marcador de ativação, captura de células secretoras) ou leitura
   direcionada. Também convém rever o estímulo (dose, tempo até a coleta,
   presença de células apresentadoras).
2. **Partir de `reseq_raw_integrado__v2.h5ad`** (P05), com a integração do
   P07 (`integracao_escolhida.h5ad`, ComBat) e os rótulos do P08
   (`clusters.parquet`): padrão de 200 UMIs, contagens SoupX, sem doublets,
   Leiden 1,0. Tirar os clusters técnicos (`cluster_tecnico`) antes de
   anotar. A matriz corrigida pelo ComBat serve só para vizinhança e
   clusterização; DE e citocinas usam as contagens. O `reseq_integrado.h5ad` (A04) continua sendo o
   registro das três variantes sobre o pré-processamento antigo.
3. **Usar a anotação do P09 e os programas do P10:**
   - tirar contaminantes e baixa qualidade (coluna `grupo`) antes de analisar
     CD4⁺;
   - não remover células por Scrublet, Solo ou cxds; levar o consenso dos três
     como coluna de sensibilidade;
   - para estados contínuos (ativação, interferon, Treg efetora), preferir a
     atividade dos programas de NMF (`H14_*`) a clusters;
   - para separar memória central e efetora, acrescentar marcadores além do
     CCR7 ou usar o tipo do atlas.
4. **Citocinas:** reportar sempre por biblioteca. Para chamar uma célula
   positiva, exigir ≥ 2 UMIs ou uma molécula com ≥ 3 reads.
5. **Estímulo:** testar por pseudobulk pareado por paciente, sobre as
   contagens. O ComBat preserva mais o estímulo que o Harmony (53% contra 35%
   da separação), mas ainda apaga parte dela. Se a composição por cluster for
   usada, conferir contra a integração por paciente.
6. **Estatística no nível de biblioteca e paciente.** HNP44_NS (70 células)
   fica fora de composição e pseudobulk.
7. **Em novas capturas:** contar células e medir viabilidade imediatamente
   antes de carregar; considerar remoção de células mortas e multiplexação
   (hashing), que mede doublets diretamente.

## 6. Perguntas que só o laboratório pode responder

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
