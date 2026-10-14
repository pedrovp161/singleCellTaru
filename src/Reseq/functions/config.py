"""
Configuração central da avaliação do ressequenciamento.

Caminhos, mapa de nomes, desenho experimental e painéis de genes vivem aqui.
Nenhum notebook repete esses valores.

Regra do projeto: nada em `data/` é escrito — isso inclui `data/allOuts`, que
é a entrada desta análise. Toda escrita vai para `dataReseq/` (objetos) e
`resultsReseq/` (figuras, tabelas, relatório).

Durante o desenvolvimento, a variável de ambiente ``RESEQ_OUT`` redireciona as
duas pastas de saída para outro lugar, de modo que as pastas do projeto só
recebam execuções finais.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/mnt/SATA/singleCellTaru")

#: Entradas — SOMENTE LEITURA.
DATA_IN = PROJECT_ROOT / "data"
ALLOUTS = DATA_IN / "allOuts"                 # Cell Ranger 10.1.0, ressequenciamento
OLD_RAW_MTX = DATA_IN / "raw"                 # corrida antiga: matriz filtrada do Cell Ranger (fonte)
OLD_RAW_H5AD = DATA_IN / "raw_h5ad"           # a mesma matriz regravada em h5ad pelo notebook original
OLD_POSTDOUBLET = DATA_IN / "new"             # corrida antiga: pós QC + Scrublet
OLD_HARMONY = DATA_IN / "adata_harmony.h5ad"  # corrida antiga: saída de processing.ipynb
OLD_AUDIT = DATA_IN / "claude" / "data"       # deltas da auditoria anterior

#: Código original do usuário e da auditoria anterior (referência e reuso).
USER_SCRIPTS = PROJECT_ROOT / "src" / "scripts"
LEGACY_CODE = USER_SCRIPTS / "claude"

CODE_ROOT = PROJECT_ROOT / "src" / "Reseq"

_out_override = os.environ.get("RESEQ_OUT")
if _out_override:
    DATA_OUT = Path(_out_override) / "dataReseq"
    RESULTS = Path(_out_override) / "resultsReseq"
else:
    DATA_OUT = PROJECT_ROOT / "dataReseq"
    RESULTS = PROJECT_ROOT / "resultsReseq"

FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MANIFEST = DATA_OUT / "MANIFEST.jsonl"

#: Prefixos sob os quais a escrita é permitida. `io` recusa qualquer outro.
WRITABLE_ROOTS = (DATA_OUT, RESULTS)

for _d in (DATA_OUT, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bibliotecas
# ─────────────────────────────────────────────────────────────────────────────

#: Pasta em `data/allOuts` → library_id canônico (o da corrida antiga).
#:
#: O ressequenciamento chegou com nomes diferentes: sublinhado e zero à esquerda
#: no paciente, e — o que importa — `SN` em vez de `NS` em duas bibliotecas. O
#: código original deriva o estímulo com ``endswith("NS")``; sem canonizar,
#: MB_34_SN e PB_20_SN seriam rotuladas como estimuladas. A equivalência foi
#: verificada pela sobreposição de barcodes entre as duas corridas (C02).
RESEQ_TO_CANONICAL = {
    "CTH92_ML": "CTH92_ML",
    "CTH92_NS": "CTH92_NS",
    "CTH94_ML": "CTH94_ML",
    "CTH94_NS": "CTH94_NS",
    "CTH97_ML": "CTH97_ML",
    "CTH97_NS": "CTH97_NS",
    "HNP_044_ML": "HNP44_ML",
    "HNP_044_NS": "HNP44_NS",
    "MB_34_ML": "MB34_ML",
    "MB_34_SN": "MB34_NS",
    "PB_20_ML": "PB20_ML",
    "PB_20_SN": "PB20_NS",
    "PB_21_ML": "PB21_ML",
    "PB_21_NS": "PB21_NS",
}
CANONICAL_TO_RESEQ = {v: k for k, v in RESEQ_TO_CANONICAL.items()}

#: Ordem canônica das 14 bibliotecas (paciente, depois NS → ML).
LIBRARIES = [
    "CTH92_NS", "CTH92_ML", "CTH94_NS", "CTH94_ML", "CTH97_NS", "CTH97_ML",
    "PB20_NS", "PB20_ML", "PB21_NS", "PB21_ML",
    "MB34_NS", "MB34_ML", "HNP44_NS", "HNP44_ML",
]

PATIENTS = ["CTH92", "CTH94", "CTH97", "PB20", "PB21", "MB34", "HNP44"]

#: Classificação clínica por paciente. HNP44 é hanseníase neural pura (NP); a
#: coluna original `classificacao` o rotula como "MB" e é preservada à parte.
PATIENT_CLASSIFICATION = {
    "CTH92": "controle", "CTH94": "controle", "CTH97": "controle",
    "PB20": "PB", "PB21": "PB", "MB34": "MB", "HNP44": "NP",
}
CLASSIFICATION_ORDER = ["controle", "PB", "MB", "NP"]

STIMULATION_SUFFIX = {"NS": "nao_estimulado", "ML": "estimulado"}
STIMULATION_ORDER = ["nao_estimulado", "estimulado"]
STIMULATION_LABEL = {"nao_estimulado": "não estimulado (NS)", "estimulado": "estimulado (ML)"}

N_PATIENTS = len(PATIENTS)
N_LIBRARIES = len(LIBRARIES)


def patient_of(library_id: str) -> str:
    return library_id.rsplit("_", 1)[0]


def stimulation_of(library_id: str) -> str:
    return STIMULATION_SUFFIX[library_id.rsplit("_", 1)[1]]


def classification_of(library_id: str) -> str:
    return PATIENT_CLASSIFICATION[patient_of(library_id)]


def outs_dir(library_id: str) -> Path:
    """Pasta `outs` do Cell Ranger para um library_id canônico."""
    return ALLOUTS / CANONICAL_TO_RESEQ[library_id] / "outs"


def run_dir(library_id: str) -> Path:
    return ALLOUTS / CANONICAL_TO_RESEQ[library_id]


# ─────────────────────────────────────────────────────────────────────────────
# Painéis de genes
# ─────────────────────────────────────────────────────────────────────────────

#: Citocinas efetoras — o motivo do ressequenciamento.
CYTOKINES = [
    "IFNG", "TNF", "IL2", "IL4", "IL5", "IL9", "IL10", "IL13",
    "IL17A", "IL17F", "IL21", "IL22", "CSF2", "IL26", "IL31", "IL3",
]

#: As 10 que a auditoria anterior precisou recuperar do bruto (functions/recovery.py).
PREVIOUSLY_MISSING = [
    "IL4", "IL9", "IL10", "IL13", "IL17A", "IL17F", "IL21",
    "CCR8", "NRP1", "PTGDR2",
]

#: Fatores de transcrição e receptores de linhagem, como controle de comparação.
LINEAGE_TFS = ["TBX21", "GATA3", "RORC", "FOXP3", "BCL6"]

#: Marcadores usados apenas para QC de linhagem (pureza do sorting e
#: contaminantes) — não é anotação de subtipo T.
MARKERS_T = ["CD3D", "CD3E", "CD3G", "TRAC", "TRBC1", "TRBC2", "CD5", "CD6", "CD28"]
MARKERS_B = ["CD79A", "CD79B", "MS4A1", "CD19", "BANK1", "CD22", "PAX5", "FCRLA"]
MARKERS_PLASMA = ["JCHAIN", "MZB1", "SDC1", "TNFRSF17", "DERL3"]
MARKERS_MYELOID = ["LYZ", "S100A8", "S100A9", "CST3", "FCN1", "AIF1", "TYROBP",
                   "FCER1G", "CD14", "CSF1R", "MS4A7", "LST1"]
MARKERS_ERYTHROID = ["HBB", "HBA1", "HBA2"]
SEX_GENES = {"feminino": ["XIST"], "masculino": ["RPS4Y1", "DDX3Y", "UTY", "EIF1AY", "KDM5D"]}

#: Genes ligados ao Y presentes na referência. Em bibliotecas de doadoras eles
#: não podem ser expressão — servem de controle negativo para moléculas espúrias.
Y_GENES = ["RPS4Y1", "DDX3Y", "UTY", "EIF1AY", "KDM5D", "ZFY", "USP9Y", "TXLNGY",
           "NLGN4Y", "PCDH11Y", "TMSB4Y", "TBL1Y", "TTTY14", "LINC00278"]

#: Sexo do doador, inferido em C02 (XIST × genes do Y) e conferido lá por asserção.
PATIENT_SEX = {"CTH92": "feminino", "CTH94": "feminino", "CTH97": "masculino",
               "PB20": "feminino", "PB21": "feminino", "MB34": "masculino", "HNP44": "feminino"}


# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros do pipeline original (réplica)
# ─────────────────────────────────────────────────────────────────────────────

MAD_K = 5                      # st.pp.Preprocessing.run("MAD_combined", k=5)
MT_DIAGNOSTIC_PCT = 8.0        # flag `mt_outlier`, criado mas nunca aplicado
SCRUBLET_EXPECTED_RATE = 0.06  # fixo para todas as amostras
N_HVG = 2000
N_PCS = 50
N_DIMS_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
GLOBAL_MIN_CELLS = 3           # variante V3: filtro único no conjunto

#: Taxa de multipletos do 10x 3' v3: ~0,8% por 1.000 células recuperadas.
MULTIPLET_RATE_PER_1000 = 0.008

RANDOM_STATE = 42

#: Números gravados nas saídas do notebook original (preprocessing.ipynb), usados
#: para validar a réplica sobre os dados antigos.
OLD_N_RAW = {
    "CTH92_ML": 1562, "CTH92_NS": 847, "CTH94_ML": 546, "CTH94_NS": 5981,
    "CTH97_ML": 4695, "CTH97_NS": 1603, "HNP44_ML": 2043, "HNP44_NS": 108,
    "MB34_ML": 999, "MB34_NS": 2136, "PB20_ML": 838, "PB20_NS": 637,
    "PB21_ML": 11247, "PB21_NS": 3919,
}
OLD_N_AFTER_COMBINED = {
    "CTH92_ML": 1562, "CTH92_NS": 825, "CTH94_ML": 520, "CTH94_NS": 5671,
    "CTH97_ML": 4695, "CTH97_NS": 1528, "HNP44_ML": 1909, "HNP44_NS": 92,
    "MB34_ML": 953, "MB34_NS": 1940, "PB20_ML": 794, "PB20_NS": 613,
    "PB21_ML": 10798, "PB21_NS": 3664,
}
OLD_N_AFTER_MAD = {
    "CTH92_ML": 1527, "CTH92_NS": 812, "CTH94_ML": 499, "CTH94_NS": 5558,
    "CTH97_ML": 4657, "CTH97_NS": 1506, "HNP44_ML": 1862, "HNP44_NS": 88,
    "MB34_ML": 921, "MB34_NS": 1911, "PB20_ML": 777, "PB20_NS": 606,
    "PB21_ML": 10619, "PB21_NS": 3626,
}
OLD_N_AFTER_SCRUBLET = {
    "CTH92_ML": 1523, "CTH92_NS": 809, "CTH94_ML": 496, "CTH94_NS": 5558,
    "CTH97_ML": 4657, "CTH97_NS": 1502, "HNP44_ML": 1859, "HNP44_NS": 77,
    "MB34_ML": 921, "MB34_NS": 1906, "PB20_ML": 777, "PB20_NS": 604,
    "PB21_ML": 10617, "PB21_NS": 3625,
}
OLD_INTEGRATED_SHAPE = (34931, 17761)
