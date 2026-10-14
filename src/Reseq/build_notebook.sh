#!/usr/bin/env bash
# Converte um .py (jupytext percent) em .ipynb e o executa, gravando as saídas.
#
# Uso: ./build_notebook.sh <pasta/nome_sem_extensao> [timeout_seg]
#   ./build_notebook.sh comparativa/C01_websummary_metricas
#   ./build_notebook.sh analise/A04_processamento_integracao 7200
#
# Ordem de execução (cada notebook lê o que os anteriores gravaram em dataReseq/):
#   comparativa/C01 → C02 → C03        (independentes da pasta analise/)
#   analise/A01 → A02 → A03 → A04 → A05
#
# Com RESEQ_OUT=<pasta> as saídas vão para <pasta>/dataReseq e
# <pasta>/resultsReseq em vez das pastas do projeto (modo de desenvolvimento).
set -euo pipefail
RESEQ_DIR="/mnt/SATA/singleCellTaru/src/Reseq"
VENV="/mnt/SATA/singleCellTaru/.venv/bin"
NAME="$1"
TIMEOUT="${2:-3600}"

cd "$RESEQ_DIR"
export RESEQ_NOTEBOOK="$NAME"
export PYTHONDONTWRITEBYTECODE=1
"$VENV/jupytext" --to notebook --output "${NAME}.ipynb" "${NAME}.py"
"$VENV/jupyter" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout="$TIMEOUT" \
    --ExecutePreprocessor.kernel_name=python3 \
    "${NAME}.ipynb"
echo "OK: ${NAME}.ipynb"
