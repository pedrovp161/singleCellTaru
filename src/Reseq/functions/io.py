"""
Leitura das saídas do Cell Ranger e escrita versionada dos resultados.

Leitura
-------
* ``load_web_summary``   — JSON embutido no ``web_summary.html`` (``const data = …``)
* ``read_metrics``       — ``metrics_summary.csv`` com números já convertidos
* ``read_10x_h5``        — matriz 10x (h5) sem passar por AnnData
* ``read_molecule_info`` — campos escolhidos de ``molecule_info.h5``

Escrita
-------
Duas áreas, duas políticas:

* ``dataReseq/`` (``save_data``) — objetos de dados. Nunca são substituídos nem
  apagados: se o conteúdo é o mesmo a gravação é pulada; se mudou, grava-se uma
  nova versão (``nome__v2``) e ``load_data`` passa a devolver a mais recente.
* ``resultsReseq/`` (``save_table``, ``plotting.savefig``) — figuras e tabelas
  de resultado, regeneradas a cada execução do notebook que as produz.

Tudo o que é escrito fica registrado em ``dataReseq/MANIFEST.jsonl``
(append-only). Nenhuma função daqui escreve fora dessas duas pastas.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config

_DEV_MODE = bool(os.environ.get("RESEQ_OUT"))

_TRACKED_PACKAGES = [
    "scanpy", "anndata", "numpy", "scipy", "pandas", "scikit-learn", "harmonypy",
    "leidenalg", "igraph", "scrublet", "spatools", "umap-learn", "h5py",
]


# ─────────────────────────────────────────────────────────────────────────────
# Web summary e métricas
# ─────────────────────────────────────────────────────────────────────────────

def load_web_summary(library_id: str) -> dict:
    """Devolve o dicionário ``summary`` embutido no web_summary.html.

    O HTML do Cell Ranger 10 carrega todos os dados da página num único literal
    JSON atribuído a ``const data``. Lemos esse literal em vez de raspar o HTML
    renderizado: é a mesma fonte que a página usa.
    """
    path = config.outs_dir(library_id) / "web_summary.html"
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "const data = "
    start = text.index(marker) + len(marker)
    data, _ = json.JSONDecoder().raw_decode(text, start)
    return data["summary"]


def parse_metric(value: Any) -> float:
    """'1,570' → 1570.0 · '97.3%' → 0.973 · já numérico → float."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().replace(",", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)


def read_metrics(library_id: str) -> pd.Series:
    """``metrics_summary.csv`` como Series numérica (percentuais em fração)."""
    df = pd.read_csv(config.outs_dir(library_id) / "metrics_summary.csv")
    return df.iloc[0].map(parse_metric).rename(library_id)


def web_summary_alarms(ws: dict) -> pd.DataFrame:
    """Alarmes da página, sem o aviso promocional de anotação automática."""
    rows = []
    for a in ws.get("alarms", {}).get("alarms", []):
        if a.get("level") == "INFO":
            continue
        rows.append({
            "level": a["level"], "title": a["title"],
            "value": a.get("formatted_value"), "message": a.get("message", ""),
            "metric_id": a.get("id"),
        })
    return pd.DataFrame(rows, columns=["level", "title", "value", "message", "metric_id"])


def web_summary_tables(ws: dict) -> dict[str, str]:
    """Tabelas 'Sequencing', 'Mapping', 'Cells' e informações do pipeline."""
    tab = ws["summary_tab"]
    out = {r[0]: r[1] for r in tab["pipeline_info_table"]["rows"]}
    for key in ("sequencing", "mapping", "cells"):
        out.update({r[0]: r[1] for r in tab[key]["table"]["rows"]})
    return out


def web_summary_curves(ws: dict) -> pd.DataFrame:
    """Curvas de subamostragem do próprio Cell Ranger.

    Colunas: ``mean_reads_per_cell``, ``saturation``, ``median_genes``,
    ``median_umis``. São os pontos dos três gráficos da aba *Analysis*.
    """
    a = ws["analysis_tab"]
    sat = a["seq_saturation_plot"]["plot"]["data"][0]
    gen = a["median_gene_plot"]["plot"]["data"][0]
    umi = a["median_umi_plot"]["plot"]["data"][0]
    return pd.DataFrame({
        "mean_reads_per_cell": np.asarray(sat["x"], dtype=float),
        "saturation": np.asarray(sat["y"], dtype=float),
        "median_genes": np.asarray(gen["y"], dtype=float),
        "median_umis": np.asarray(umi["y"], dtype=float),
    })


def read_celltypes(library_id: str) -> pd.DataFrame:
    """Anotação automática do Cell Ranger (Azimuth, beta), por barcode."""
    path = config.outs_dir(library_id) / "cell_types" / "Azimuth" / "cell_types.csv"
    return pd.read_csv(path).set_index("barcode")


# ─────────────────────────────────────────────────────────────────────────────
# Matrizes 10x
# ─────────────────────────────────────────────────────────────────────────────

def _dec(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def read_10x_h5(path: str | Path) -> dict:
    """Matriz 10x como CSR células × genes, mais barcodes e features.

    Devolve ``{"X", "barcodes", "gene_ids", "gene_names"}``. Não cria AnnData:
    é o leitor leve para comparações e para as matrizes brutas (~1,5 M barcodes).
    """
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(int(x) for x in g["shape"][()])       # genes × barcodes
        M = sp.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=shape)
        out = {
            "X": M.T.tocsr(),                                # células × genes
            "barcodes": _dec(g["barcodes"][()]),
            "gene_ids": _dec(g["features/id"][()]),
            "gene_names": _dec(g["features/name"][()]),
        }
    return out


def read_filtered(library_id: str) -> dict:
    return read_10x_h5(config.outs_dir(library_id) / "filtered_feature_bc_matrix.h5")


def read_raw(library_id: str) -> dict:
    return read_10x_h5(config.outs_dir(library_id) / "raw_feature_bc_matrix.h5")


def read_old_raw_h5ad(library_id: str) -> dict:
    """Matriz filtrada do Cell Ranger da corrida ANTIGA (`data/raw_h5ad`).

    Mesmo formato de ``read_10x_h5``. ``gene_names`` são os var_names do h5ad
    (símbolos tornados únicos); ``gene_ids`` são os Ensembl IDs.
    """
    with h5py.File(config.OLD_RAW_H5AD / f"{library_id}.h5ad", "r") as f:
        X = f["X"]
        shape = tuple(int(x) for x in X.attrs["shape"])
        cls = sp.csr_matrix if X.attrs.get("encoding-type") == "csr_matrix" else sp.csc_matrix
        M = cls((X["data"][()], X["indices"][()], X["indptr"][()]), shape=shape)
        var = f["var"]
        idx_name = var.attrs.get("_index", "_index")
        out = {
            "X": M.tocsr(),
            "barcodes": _dec(f["obs"][f["obs"].attrs.get("_index", "_index")][()]),
            "gene_names": _dec(var[idx_name][()]),
            "gene_ids": _dec(var["gene_ids"][()]) if "gene_ids" in var else None,
        }
    return out


def read_old_matrix(library_id: str) -> dict:
    """Matriz filtrada do Cell Ranger da corrida ANTIGA, da fonte (`data/raw/<lib>/`).

    É daqui que o notebook original partiu. A cópia em `data/raw_h5ad` não é
    usada como referência: a de CTH94_NS diverge desta fonte (ver C02).
    Mesmo formato de ``read_10x_h5``.
    """
    from scipy.io import mmread
    d = config.OLD_RAW_MTX / library_id
    M = mmread(str(d / "matrix.mtx.gz"))                    # genes × barcodes
    feats = pd.read_csv(d / "features.tsv.gz", sep="\t", header=None)
    bcs = pd.read_csv(d / "barcodes.tsv.gz", header=None)[0].to_numpy().astype(str)
    return {"X": sp.csr_matrix(M.T), "barcodes": bcs,
            "gene_ids": feats[0].to_numpy().astype(str),
            "gene_names": feats[1].to_numpy().astype(str)}


def reference_features() -> pd.DataFrame:
    """Tabela de features da referência (GRCh38-2024-A), idêntica nas 14 bibliotecas."""
    with h5py.File(config.outs_dir(config.LIBRARIES[0]) / "filtered_feature_bc_matrix.h5", "r") as f:
        g = f["matrix/features"]
        return pd.DataFrame({
            "gene_id": _dec(g["id"][()]),
            "gene_name": _dec(g["name"][()]),
            "feature_type": _dec(g["feature_type"][()]),
        })


def gene_index(names: Iterable[str], genes: Iterable[str]) -> dict[str, int]:
    """Posição de cada gene em `names` (primeira ocorrência). Ausentes ficam de fora."""
    names = np.asarray(list(names))
    out = {}
    for g in genes:
        hit = np.flatnonzero(names == g)
        if hit.size:
            out[g] = int(hit[0])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# molecule_info.h5
# ─────────────────────────────────────────────────────────────────────────────

def read_molecule_info(library_id: str,
                       fields: Iterable[str] = ("count", "barcode_idx", "feature_idx")) -> dict:
    """Campos por molécula de ``molecule_info.h5``.

    Campos disponíveis: ``count`` (reads da molécula), ``barcode_idx``,
    ``feature_idx``, ``umi`` (2 bits por base), ``umi_type`` (1 = exônico,
    0 = intrônico). Sempre devolve também ``barcodes`` (com sufixo ``-1``, como
    nas matrizes), ``gene_names``, ``gene_ids``, ``cell_barcode_idx`` (índices
    dos barcodes chamados como célula) e ``libraries`` (pares de reads).
    """
    path = config.outs_dir(library_id) / "molecule_info.h5"
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for name in fields:
            out[name] = f[name][()]
        bc = _dec(f["barcodes"][()])
        gem = int(f["gem_group"][0]) if f["gem_group"].shape[0] else 1
        out["barcodes"] = np.char.add(bc.astype(str), f"-{gem}")
        out["gene_names"] = _dec(f["features/name"][()])
        out["gene_ids"] = _dec(f["features/id"][()])
        out["cell_barcode_idx"] = f["barcode_info/pass_filter"][:, 0].astype(np.int64)
        mj = f["metrics_json"][()]
        mj = json.loads(mj.decode() if isinstance(mj, bytes) else mj)
        out["libraries"] = mj.get("libraries", {})
        out["chemistry"] = mj.get("chemistry_description")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Manifesto
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _packages() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for p in _TRACKED_PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not-installed"
    return out


def _assert_writable(path: Path) -> None:
    rp = Path(path).resolve()
    for root in config.WRITABLE_ROOTS:
        try:
            rp.relative_to(Path(root).resolve())
            return
        except ValueError:
            continue
    raise PermissionError(
        f"escrita recusada em {rp}: fora de {[str(r) for r in config.WRITABLE_ROOTS]}")


def _source() -> str:
    return os.environ.get("RESEQ_NOTEBOOK", "interativo")


def register(*, stage: str, name: str, kind: str, path: Path, description: str,
             signature: str | None = None, shape=None, extra: dict | None = None) -> dict:
    record = {
        "timestamp": _now(), "stage": stage, "name": name, "kind": kind,
        "path": str(path), "bytes": path.stat().st_size if path.exists() else None,
        "signature": signature, "shape": list(shape) if shape is not None else None,
        "description": description, "source_notebook": _source(),
        "packages": _packages(), **(extra or {}),
    }
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MANIFEST, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    return record


def manifest() -> pd.DataFrame:
    if not config.MANIFEST.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in
            config.MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows).drop(columns=["packages"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Escrita de dados (dataReseq) — idempotente ou versionada
# ─────────────────────────────────────────────────────────────────────────────

def _signature(obj) -> str:
    h = hashlib.sha256()
    if isinstance(obj, pd.DataFrame):
        h.update(repr(list(obj.columns)).encode())
        h.update(repr([str(t) for t in obj.dtypes]).encode())
        h.update(pd.util.hash_pandas_object(obj, index=True).values.tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(repr((obj.shape, str(obj.dtype))).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    else:                                           # AnnData
        X = obj.X
        h.update(repr(obj.shape).encode())
        if sp.issparse(X):
            X = X.tocsr()
            for part in (X.data, X.indices, X.indptr):
                h.update(np.ascontiguousarray(part).tobytes())
        else:
            h.update(np.ascontiguousarray(np.asarray(X)).tobytes())
        h.update("|".join(obj.obs_names).encode())
        h.update("|".join(obj.var_names).encode())
        h.update(repr(sorted(obj.obs.columns)).encode())
        h.update(repr(sorted(obj.var.columns)).encode())
        h.update(repr(sorted(obj.layers.keys())).encode())
        for k in sorted(obj.obsm.keys()):
            h.update(k.encode())
            h.update(np.ascontiguousarray(np.asarray(obj.obsm[k])).tobytes())
        for col in sorted(obj.obs.columns):
            try:
                h.update(pd.util.hash_pandas_object(obj.obs[col], index=False).values.tobytes())
            except TypeError:
                h.update(repr(obj.obs[col].astype(str).tolist()).encode())
    return h.hexdigest()


def _ext(obj) -> str:
    if isinstance(obj, pd.DataFrame):
        return "parquet"
    if isinstance(obj, np.ndarray):
        return "npy"
    return "h5ad"


def _versions(stage: str, name: str, ext: str) -> list[tuple[int, Path]]:
    d = config.DATA_OUT / stage
    out = []
    if (d / f"{name}.{ext}").exists():
        out.append((1, d / f"{name}.{ext}"))
    pat = re.compile(rf"^{re.escape(name)}__v(\d+)\.{re.escape(ext)}$")
    if d.exists():
        for p in d.iterdir():
            m = pat.match(p.name)
            if m:
                out.append((int(m.group(1)), p))
    return sorted(out)


def _last_signature(path: Path) -> str | None:
    if not config.MANIFEST.exists():
        return None
    sig = None
    for line in config.MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("path") == str(path):
            sig = rec.get("signature")
    return sig


def save_data(obj, stage: str, name: str, *, description: str) -> Path:
    """Grava DataFrame (parquet), ndarray (npy) ou AnnData (h5ad gzip) em dataReseq.

    Nunca substitui: conteúdo igual ao da versão mais recente → não grava;
    conteúdo diferente → grava ``nome__vN``. Em modo de desenvolvimento
    (``RESEQ_OUT`` definido) o arquivo é simplesmente regravado.
    """
    ext = _ext(obj)
    d = config.DATA_OUT / stage
    d.mkdir(parents=True, exist_ok=True)
    sig = _signature(obj)
    existing = _versions(stage, name, ext)

    if existing and not _DEV_MODE:
        last_v, last_path = existing[-1]
        if _last_signature(last_path) == sig:
            print(f"[io] inalterado, não regravado: {last_path.name}")
            return last_path
        path = d / f"{name}__v{last_v + 1}.{ext}"
        print(f"[io] conteúdo mudou — nova versão: {path.name} (anterior preservada)")
    else:
        path = d / f"{name}.{ext}"

    _assert_writable(path)
    if ext == "parquet":
        obj.to_parquet(path)
    elif ext == "npy":
        np.save(path, obj)
    else:
        obj.write_h5ad(path, compression="gzip")
        _verify_h5ad(path, obj)
    register(stage=stage, name=name, kind=ext, path=path, description=description,
             signature=sig, shape=getattr(obj, "shape", None))
    return path


def _verify_h5ad(path: Path, obj) -> None:
    """Relê o h5ad recém-gravado e confere a matriz contra o objeto em memória.

    O disco do projeto é NTFS, e um dos h5ad antigos (`data/raw_h5ad/CTH94_NS`)
    tem um trecho de `X` zerado em relação à fonte. Conferir na gravação custa
    segundos e evita propagar um arquivo assim.
    """
    import anndata as ad
    back = ad.read_h5ad(path)
    Xa, Xb = obj.X, back.X
    same_shape = back.shape == obj.shape
    sa = float(Xa.sum()); sb = float(Xb.sum())
    na = Xa.nnz if sp.issparse(Xa) else int(np.count_nonzero(Xa))
    nb = Xb.nnz if sp.issparse(Xb) else int(np.count_nonzero(Xb))
    ok = same_shape and na == nb and np.isclose(sa, sb, rtol=1e-9, atol=1e-6)
    for k in obj.layers.keys():
        la, lb = obj.layers[k], back.layers[k]
        ok = ok and np.isclose(float(la.sum()), float(lb.sum()), rtol=1e-9, atol=1e-6)
    if not ok:
        raise IOError(f"{path}: o arquivo gravado não confere com o objeto em memória")


def data_path(stage: str, name: str, ext: str) -> Path:
    """Caminho da versão mais recente de um artefato de dados."""
    existing = _versions(stage, name, ext)
    if not existing:
        raise FileNotFoundError(f"{config.DATA_OUT / stage / name}.{ext} não existe")
    return existing[-1][1]


def load_data(stage: str, name: str, ext: str = "parquet"):
    path = data_path(stage, name, ext)
    if ext == "parquet":
        return pd.read_parquet(path)
    if ext == "npy":
        return np.load(path)
    import anndata as ad
    return ad.read_h5ad(path)


# ─────────────────────────────────────────────────────────────────────────────
# Escrita de resultados (resultsReseq) — regenerados a cada execução
# ─────────────────────────────────────────────────────────────────────────────

def save_table(df: pd.DataFrame, notebook: str, name: str, *, description: str,
               index: bool = True, float_format: str | None = "%.6g") -> Path:
    """Tabela de resultado em ``resultsReseq/tables/<notebook>/<name>.csv``."""
    d = config.TABLES / notebook
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.csv"
    _assert_writable(path)
    df.to_csv(path, index=index, float_format=float_format)
    register(stage=notebook, name=name, kind="table", path=path,
             description=description, shape=df.shape)
    return path
