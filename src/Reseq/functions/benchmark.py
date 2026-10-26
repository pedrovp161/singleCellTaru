"""
Benchmark de métodos de integração no estilo do scIB.

Luecken et al. (2022), *Benchmarking atlas-level data integration in
single-cell genomics*, Nat Methods 19:41–50. As métricas são as do pacote do
artigo (``scib`` 1.1.7), chamadas uma a uma. A exceção é o kBET: no ``scib`` ele
exige o pacote R kBET via rpy2, e aqui é um porte do mesmo procedimento
(`kbet`). Os ajustes seguem os parâmetros dos wrappers de ``scib.integration``,
mas preservam a ordem das células.

Fluxo (cada etapa grava em ``HEAVY_OUT/P07_benchmark_integracao/``):

1. `prepare` — entradas comuns a partir do objeto principal: as 2.000 HVG do
   P05, lognorm com e sem escala, PCA de cada uma, a lognorm de todos os genes
   (para o ciclo celular) e o rótulo Azimuth agrupado;
2. `integrate` — um ajuste (um método numa preparação); grava as saídas das
   execuções que ele produz, com tempo e memória;
3. `evaluate` — as métricas de uma execução, a UMAP e o grafo de 15 vizinhos;
4. `aggregate` — escalonamento min-max entre execuções e combinação do artigo
   (0,4 × remoção de lote + 0,6 × conservação biológica).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

from . import config, io, pipeline as pl

STAGE = "P07_benchmark_integracao"
#: ``RESEQ_BENCH_CACHE`` desvia o cache (teste em subamostra); tem de ficar sob HEAVY_OUT.
CACHE = Path(os.environ.get("RESEQ_BENCH_CACHE", config.HEAVY_OUT / STAGE))
_REGISTER = "RESEQ_BENCH_CACHE" not in os.environ   # o teste não vai para o MANIFEST
BATCH = "library_id"
LABEL = "rotulo_bench"
LABEL_TRAIN = "rotulo_treino"     # rótulo dado a scANVI e scGen (sem rótulo → categoria própria)
UNLABELED = "UnknownUnknown"      # categoria não rotulada do scANVI (a mesma do wrapper do scib)
N_NEIGHBORS = 15
SEED = 0

# ─────────────────────────────────────────────────────────────────────────────
# Rótulo de tipo celular (Azimuth do Cell Ranger, agrupado)
# ─────────────────────────────────────────────────────────────────────────────

#: Rótulos finos do Azimuth → grupos usados nas métricas biológicas. O que não
#: está aqui (não confiante, sem atribuição, progenitores, plaquetas, células
#: em ciclo, "Naive T cell" sem CD4/CD8) fica sem rótulo: entra na integração,
#: mas não nas métricas.
AZIMUTH_GROUPS = {
    "CD4 naive": ["Naive CD4 T cell"],
    "CD4 memória": ["Memory CD4 T cell", "Tfh cell", "KLRB1 cytotoxic CD4 T"],
    "Treg": ["Treg cell"],
    "CD8 e T não convencional": ["GZMB CD8 T cell", "GZMK CD8 T cell", "Naive CD8 T cell", "MAIT cell",
                                 "CD8aa+ T cell", "gdT cell"],
    "B e plasmócito": ["Naive B cell", "Memory B cell", "Plasma cell", "ProB cell"],
    "NK e ILC": ["CD16 NK cell", "CD56 NK cell", "ILC"],
    "mieloide": ["CD14 monocyte", "CD16 monocyte", "CD14/CD16 monocyte", "mregDC", "LYVE1 macrophage",
                 "Mast cell"],
}
LABEL_ORDER = list(AZIMUTH_GROUPS)


def azimuth_labels(obs: pd.DataFrame) -> pd.DataFrame:
    """Rótulo fino do Azimuth (Cell Ranger, por biblioteca) e o grupo, por ``cell_key``.

    Só as células chamadas pelo Cell Ranger têm Azimuth; as demais ficam sem rótulo.
    """
    rows = []
    for lib in pd.unique(obs[BATCH].astype(str)):
        d = io.read_celltypes(lib).reset_index()
        d["cell_key"] = lib + "|" + d["barcode"].astype(str)
        rows.append(d[["cell_key", "fine_cell_type", "final_level_softmax_prob"]])
    az = pd.concat(rows).set_index("cell_key")
    keys = obs["cell_key"].astype(str).to_numpy()
    fino = az["fine_cell_type"].reindex(keys)
    groups = {f: g for g, fs in AZIMUTH_GROUPS.items() for f in fs}
    return pd.DataFrame({"cell_key": keys, "azimuth_fino": fino.to_numpy(),
                         "azimuth_prob": az["final_level_softmax_prob"].reindex(keys).to_numpy(),
                         LABEL: pd.Categorical(fino.map(groups).to_numpy(), categories=LABEL_ORDER)})


# ─────────────────────────────────────────────────────────────────────────────
# Execuções
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Run:
    key: str                  # diretório do cache
    method: str               # nome exibido
    output: str               # "embedding" | "genes" | "grafo"
    scaled: bool | None       # None: o método não recebe a matriz escalonada
    fit: str                  # ajuste que produz a execução (um ajuste pode dar duas saídas)
    uses_labels: bool = False
    language: str = "Python"
    dims: int | None = None   # só as primeiras dimensões do embedding

    @property
    def label(self) -> str:
        return self.method + ("*" if self.uses_labels else "")

    @property
    def scaling(self) -> str:
        return {True: "escalado", False: "sem escala", None: "—"}[self.scaled]


RUNS = [
    Run("nao_integrado_esc", "Não integrado", "embedding", True, "pca_esc"),
    Run("nao_integrado", "Não integrado", "embedding", False, "pca"),
    Run("harmony_esc", "Harmony", "embedding", True, "harmony_esc"),
    Run("harmony", "Harmony", "embedding", False, "harmony"),
    Run("harmony_esc_15d", "Harmony · 15 dims (P05)", "embedding", True, "harmony_esc", dims=config.N_DIMS_NEIGHBORS),
    Run("scanorama_esc_emb", "Scanorama", "embedding", True, "scanorama_esc"),
    Run("scanorama_emb", "Scanorama", "embedding", False, "scanorama"),
    Run("scanorama_esc_genes", "Scanorama", "genes", True, "scanorama_esc"),
    Run("scanorama_genes", "Scanorama", "genes", False, "scanorama"),
    Run("bbknn_esc", "BBKNN", "grafo", True, "bbknn_esc"),
    Run("bbknn", "BBKNN", "grafo", False, "bbknn"),
    Run("combat_esc", "ComBat", "genes", True, "combat_esc"),
    Run("combat", "ComBat", "genes", False, "combat"),
    Run("scvi", "scVI", "embedding", None, "scvi"),
    Run("scanvi", "scANVI", "embedding", None, "scanvi", uses_labels=True),
    Run("scgen", "scGen", "genes", False, "scgen", uses_labels=True),
    Run("seurat_cca", "Seurat v5 CCA", "embedding", None, "seurat_cca", language="R"),
    Run("seurat_rpca", "Seurat v5 RPCA", "embedding", None, "seurat_rpca", language="R"),
]
RUN = {r.key: r for r in RUNS}
FITS = list(dict.fromkeys(r.fit for r in RUNS))
REFERENCE = "nao_integrado_esc"       # referência das métricas do estudo
CURRENT = "harmony_esc_15d"           # a integração atual (P05)


def runs_table() -> pd.DataFrame:
    return pd.DataFrame([{**asdict(r), "rotulo": r.label, "escala": r.scaling} for r in RUNS]).set_index("key")


# ─────────────────────────────────────────────────────────────────────────────
# Entradas comuns
# ─────────────────────────────────────────────────────────────────────────────

def _signature(obs: pd.DataFrame, genes) -> str:
    h = hashlib.sha1()
    h.update("\n".join(obs["cell_key"].astype(str)).encode())
    h.update("\n".join(map(str, genes)).encode())
    h.update("\n".join(obs[LABEL].astype(str)).encode())
    return h.hexdigest()[:16]


def prepare(adata, out: Path = CACHE / "entradas") -> dict:
    """Entradas comuns de todas as execuções. Idempotente pela assinatura.

    * ``pre``: lognorm (todos os genes → 2.000 HVG do P05), contagens SoupX das
      HVG em ``layers["counts"]`` e PCA de 50 componentes;
    * ``pre_esc``: a mesma lognorm escalonada (``max_value=10``) e a sua PCA —
      é a entrada do Harmony em P05;
    * ``full``: lognorm de todos os genes com símbolos (ciclo celular).
    """
    hv = adata.var["highly_variable"].to_numpy()
    cols = np.flatnonzero(hv)
    obs = pd.DataFrame({
        "cell_key": adata.obs["cell_key"].astype(str).to_numpy(),
        BATCH: pd.Categorical(adata.obs[BATCH].astype(str), categories=config.LIBRARIES),
        "patient": adata.obs["patient"].astype(str).to_numpy(),
        "stimulation": adata.obs["stimulation"].astype(str).to_numpy(),
        "total_counts": adata.obs["total_counts"].to_numpy(),
    }, index=adata.obs_names.astype(str))
    lab = azimuth_labels(adata.obs)
    for c in ("azimuth_fino", "azimuth_prob", LABEL):
        obs[c] = lab[c].to_numpy()
    sig = _signature(obs, adata.var_names[cols])
    meta_path = out / "assinatura.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["assinatura"] == sig:
        return load_inputs(out)
    io._assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)

    ln_all = pl.lognorm(adata.layers["counts"])
    var = pd.DataFrame({"gene_symbol": adata.var["gene_symbol"].astype(str).to_numpy()[cols]},
                       index=adata.var_names[cols].astype(str))
    pre = ad.AnnData(X=ln_all[:, cols].tocsr(), obs=obs, var=var)
    pre.layers["counts"] = sp.csr_matrix(adata.layers["counts"][:, cols], dtype=np.float32)
    pre_esc = ad.AnnData(X=pre.X.copy(), obs=obs.copy(), var=var.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.scale(pre_esc, max_value=10)
    for a in (pre, pre_esc):
        sc.tl.pca(a, n_comps=config.N_PCS, random_state=config.RANDOM_STATE)
    sym = pd.Index(adata.var["gene_symbol"].astype(str))
    full = ad.AnnData(X=ln_all, obs=obs[[BATCH]].copy(),
                      var=pd.DataFrame(index=pd.Index(ad.utils.make_index_unique(sym))))
    pre.write_h5ad(out / "pre.h5ad")
    pre_esc.write_h5ad(out / "pre_esc.h5ad")
    full.write_h5ad(out / "full.h5ad")
    meta_path.write_text(json.dumps({"assinatura": sig, "celulas": int(adata.n_obs), "hvg": int(len(cols)),
                                     "criado": time.strftime("%Y-%m-%d %H:%M:%S")}))
    if _REGISTER:
        io.register(stage=STAGE, name="entradas", kind="cache", path=out, signature=sig,
                    description="Entradas comuns do benchmark (HVG do P05, lognorm com e sem escala, PCA)")
    return {"pre": pre, "pre_esc": pre_esc, "full": full, "assinatura": sig}


def load_inputs(out: Path = CACHE / "entradas", full: bool = True) -> dict:
    d = {"pre": ad.read_h5ad(out / "pre.h5ad"), "pre_esc": ad.read_h5ad(out / "pre_esc.h5ad"),
         "assinatura": json.loads((out / "assinatura.json").read_text())["assinatura"]}
    if full:
        d["full"] = ad.read_h5ad(out / "full.h5ad")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Ajustes (parâmetros dos wrappers de scib.integration)
# ─────────────────────────────────────────────────────────────────────────────

def _dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _train_labels(obs: pd.DataFrame, missing: str) -> pd.Categorical:
    lab = obs[LABEL].astype(object).where(obs[LABEL].notna(), missing)
    return pd.Categorical(lab.astype(str))


def _fit_pca(pre) -> dict:
    return {"emb": np.asarray(pre.obsm["X_pca"], np.float32)}


def _fit_harmony(pre) -> dict:
    Z, conv = pl._harmony(pre.obsm["X_pca"], pre.obs, BATCH)
    return {"emb": np.asarray(Z, np.float32), "info": conv}


def _fit_scanorama(pre) -> dict:
    """``scanorama.correct_scanpy`` por biblioteca (como ``scib.ig.scanorama``)."""
    import scanorama

    parts = [pre[pre.obs[BATCH] == b].copy() for b in pre.obs[BATCH].cat.categories]
    parts = [ad.AnnData(X=p.X.copy(), obs=p.obs[[BATCH]].copy(), var=pd.DataFrame(index=p.var_names)) for p in parts]
    cor = scanorama.correct_scanpy(parts, return_dimred=True)
    names = pd.Index(np.concatenate([np.asarray(c.obs_names) for c in cor]))
    pos = names.get_indexer(pre.obs_names)
    if (pos < 0).any():
        raise RuntimeError("Scanorama perdeu células")
    emb = np.vstack([c.obsm["X_scanorama"] for c in cor])[pos]
    X = np.vstack([_dense(c.X) for c in cor])[pos]
    return {"emb": emb.astype(np.float32), "X": X}


def _fit_bbknn(pre) -> dict:
    import bbknn

    tmp = ad.AnnData(obs=pre.obs[[BATCH]].copy())
    tmp.obsm["X_pca"] = np.asarray(pre.obsm["X_pca"])
    bbknn.bbknn(tmp, batch_key=BATCH, use_rep="X_pca")      # 3 vizinhos por biblioteca (padrão do scib)
    return {"conn": tmp.obsp["connectivities"].tocsr(), "dist": tmp.obsp["distances"].tocsr()}


def _fit_combat(pre) -> dict:
    tmp = ad.AnnData(X=_dense(pre.X), obs=pre.obs[[BATCH]].copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.combat(tmp, key=BATCH)
    return {"X": _dense(tmp.X)}


def _scvi_model(pre):
    import scvi

    scvi.settings.seed = SEED
    net = ad.AnnData(X=pre.layers["counts"].copy(), obs=pre.obs[[BATCH]].copy())
    net.obs[LABEL_TRAIN] = _train_labels(pre.obs, UNLABELED)
    net.layers["counts"] = net.X
    scvi.model.SCVI.setup_anndata(net, layer="counts", batch_key=BATCH)
    vae = scvi.model.SCVI(net, gene_likelihood="nb", n_layers=2, n_latent=30, n_hidden=128)
    n_epochs = int(min(round((20000 / net.n_obs) * 400), 400))
    vae.train(max_epochs=n_epochs, train_size=1.0)
    return vae, n_epochs


def _fit_scvi(pre) -> dict:
    vae, n_epochs = _scvi_model(pre)
    return {"emb": vae.get_latent_representation().astype(np.float32), "info": f"{n_epochs} épocas"}


def _fit_scanvi(pre) -> dict:
    """scVI → scANVI, épocas como em ``scib.ig.scanvi``; células sem rótulo = não rotuladas."""
    import scvi

    vae, n_scvi = _scvi_model(pre)
    n_scanvi = int(min(10, max(2, round(n_scvi / 3.0))))
    m = scvi.model.SCANVI.from_scvi_model(vae, labels_key=LABEL_TRAIN, unlabeled_category=UNLABELED)
    m.train(max_epochs=n_scanvi, train_size=1.0)
    return {"emb": m.get_latent_representation().astype(np.float32), "info": f"scVI {n_scvi} + scANVI {n_scanvi} épocas"}


def _scgen_compat():
    """scGen 2.1.0 (2022) importa nomes que o scvi-tools 1.x removeu.

    A ponte é só em memória, sem tocar no pacote instalado: ``scvi._compat``
    volta com ``Literal``; ``LossRecorder`` vira um construtor de
    ``LossOutput``; e a representação latente usa a média do codificador
    (``qz_m``), como no scvi-tools da época.
    """
    import sys
    import types
    import typing

    import torch
    import scvi.module.base as mb

    if "scvi._compat" not in sys.modules:
        compat = types.ModuleType("scvi._compat")
        compat.Literal = typing.Literal
        sys.modules["scvi._compat"] = compat
    if not hasattr(mb, "LossRecorder"):
        def LossRecorder(loss, reconstruction_loss, kl_local, kl_global=0.0):
            return mb.LossOutput(loss=loss, reconstruction_loss=reconstruction_loss,
                                 kl_local=kl_local, kl_global=kl_global)
        mb.LossRecorder = LossRecorder
    from scgen import SCGEN

    @torch.inference_mode()
    def _latent(self, adata=None, indices=None, give_mean=True, batch_size=None):
        adata = self._validate_anndata(adata)
        out = []
        for t in self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size):
            o = self.module.inference(**self.module._get_inference_input(t))
            out.append((o["qz_m"] if give_mean else o["z"]).cpu())
        return torch.cat(out).numpy()

    SCGEN.get_latent_representation = _latent
    return SCGEN


def _fit_scgen(pre) -> dict:
    """scGen com os parâmetros de ``scib.ig.scgen`` (100 épocas, lote 32, parada antecipada)."""
    import scvi

    SCGEN = _scgen_compat()
    scvi.settings.seed = SEED
    net = ad.AnnData(X=pre.X.copy(), obs=pre.obs[[BATCH]].copy(), var=pd.DataFrame(index=pre.var_names))
    net.obs[LABEL_TRAIN] = _train_labels(pre.obs, "sem rótulo")
    SCGEN.setup_anndata(net, batch_key=BATCH, labels_key=LABEL_TRAIN)
    m = SCGEN(net)
    m.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cor = m.batch_removal()
    pos = pd.Index(cor.obs_names).get_indexer(net.obs_names)
    if (pos < 0).any():
        raise RuntimeError("scGen perdeu células")
    return {"X": _dense(cor.X)[pos], "info": f"{len(m.history['elbo_train'])} épocas" if "elbo_train" in m.history else ""}


def _fit_seurat(pre, method: str) -> dict:
    """Seurat v5 ``IntegrateLayers`` (CCA ou RPCA) no R isolado (`seurat_integra.R`)."""
    import scipy.io

    from . import droplets

    d = CACHE / "_seurat" / method
    io._assert_writable(d)
    d.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(str(d / "m.mtx"), sp.csr_matrix(pre.X).T.tocoo())     # genes × células, lognorm
    pd.DataFrame({"gene": pre.var_names}).to_csv(d / "genes.csv", index=False)
    pd.DataFrame({"cell": [f"c{i}" for i in range(pre.n_obs)],
                  BATCH: pre.obs[BATCH].astype(str).to_numpy()}).to_csv(d / "obs.csv", index=False)
    r = droplets.run_r("seurat_integra.R", [d, method], timeout=6 * 3600)
    emb = pd.read_csv(d / "emb.csv", index_col=0)
    emb = emb.reindex([f"c{i}" for i in range(pre.n_obs)]).to_numpy(np.float32)
    for f in ("m.mtx", "genes.csv", "obs.csv", "emb.csv"):
        (d / f).unlink(missing_ok=True)
    return {"emb": emb, "info": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""}


_FIT_FUNCS = {
    "pca_esc": ("pre_esc", _fit_pca), "pca": ("pre", _fit_pca),
    "harmony_esc": ("pre_esc", _fit_harmony), "harmony": ("pre", _fit_harmony),
    "scanorama_esc": ("pre_esc", _fit_scanorama), "scanorama": ("pre", _fit_scanorama),
    "bbknn_esc": ("pre_esc", _fit_bbknn), "bbknn": ("pre", _fit_bbknn),
    "combat_esc": ("pre_esc", _fit_combat), "combat": ("pre", _fit_combat),
    "scvi": ("pre", _fit_scvi), "scanvi": ("pre", _fit_scanvi), "scgen": ("pre", _fit_scgen),
    "seurat_cca": ("pre", lambda p: _fit_seurat(p, "cca")),
    "seurat_rpca": ("pre", lambda p: _fit_seurat(p, "rpca")),
}


class _MemoryPeak:
    """Pico de RSS do processo e dos filhos (o R) durante o ajuste, por amostragem."""

    def __init__(self, every: float = 0.25):
        import psutil

        self.proc = psutil.Process()
        self.every = every
        self.base = self._rss()
        self.peak = self.base
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _rss(self) -> int:
        tot = self.proc.memory_info().rss
        for c in self.proc.children(recursive=True):
            try:
                tot += c.memory_info().rss
            except Exception:
                pass
        return tot

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss())


def run_dir(key: str) -> Path:
    return CACHE / key


def is_integrated(key: str, signature: str) -> bool:
    m = run_dir(key) / "ajuste.json"
    return m.exists() and json.loads(m.read_text()).get("assinatura") == signature


def integrate(fit: str, inputs: dict) -> dict:
    """Roda um ajuste e grava ``saida.npz`` + ``ajuste.json`` de cada execução que ele produz."""
    src, func = _FIT_FUNCS[fit]
    pre = inputs[src]
    t0 = time.perf_counter()
    with _MemoryPeak() as mem:
        res = func(pre)
    elapsed = time.perf_counter() - t0
    gpu_gb = 0.0                       # processo novo por ajuste: o pico da GPU é o do ajuste
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            gpu_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    except Exception:
        pass
    out = {}
    for r in (r for r in RUNS if r.fit == fit):
        d = run_dir(r.key)
        io._assert_writable(d)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {}
        if r.output == "embedding":
            emb = res["emb"] if r.dims is None else res["emb"][:, : r.dims]
            arrays["emb"] = np.ascontiguousarray(emb, dtype=np.float32)
        elif r.output == "genes":
            arrays["X"] = res["X"]
        else:
            for k in ("conn", "dist"):
                m = res[k].tocsr()
                arrays.update({f"{k}_data": m.data, f"{k}_indices": m.indices, f"{k}_indptr": m.indptr})
        np.savez(d / "saida.npz", **arrays)
        meta = {"execucao": r.key, "ajuste": fit, "assinatura": inputs["assinatura"], "tempo_s": elapsed,
                "memoria_pico_gb": (mem.peak - mem.base) / 1e9, "memoria_base_gb": mem.base / 1e9,
                "gpu_pico_gb": gpu_gb, "info": str(res.get("info", "")), "celulas": int(pre.n_obs),
                "criado": time.strftime("%Y-%m-%d %H:%M:%S")}
        (d / "ajuste.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        if _REGISTER:
            io.register(stage=STAGE, name=f"integracao_{r.key}", kind="cache", path=d / "saida.npz",
                        signature=inputs["assinatura"], description=f"Saída de {r.label} ({r.output}, {r.scaling})")
        out[r.key] = meta
    return out


def read_output(key: str) -> dict:
    z = np.load(run_dir(key) / "saida.npz")
    r = RUN[key]
    if r.output == "grafo":
        n = len(z["conn_indptr"]) - 1
        return {k: sp.csr_matrix((z[f"{k}_data"], z[f"{k}_indices"], z[f"{k}_indptr"]), shape=(n, n))
                for k in ("conn", "dist")}
    return {k: z[k] for k in z.files}


# ─────────────────────────────────────────────────────────────────────────────
# kBET (porte do procedimento de scib.metrics.kBET, sem rpy2)
# ─────────────────────────────────────────────────────────────────────────────

class NeighborsError(RuntimeError):
    pass


def _scib():
    """``import scib`` sem o efeito colateral: o import aplica o contexto "talk"
    do seaborn e muda 20 rcParams (fonte de 10 para 18 pt) das figuras seguintes."""
    import matplotlib as mpl

    with mpl.rc_context():
        import scib
        import scib.metrics.utils  # noqa: F401
    return scib


def _row_normalize(G) -> sp.csr_matrix:
    G = sp.csr_matrix(G, dtype=np.float64)
    s = np.asarray(G.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / np.where(s > 0, s, 1.0)) @ G)


def diffusion_nn(G, k: int, max_iterations: int = 26, block: int = 1500) -> np.ndarray:
    """``scib.metrics.utils.diffusion_nn`` por blocos de linhas.

    M = T + T² + T³ (+ Tⁱ até cada linha ter ≥ k + 1 entradas), com T a matriz
    de conectividades normalizada por linha; os k maiores de cada linha, sem a
    própria célula, ordenados por peso. O ``scib`` monta M inteira em memória
    densa (n × n); aqui é o mesmo cálculo, um bloco de linhas por vez.
    """
    T = _row_normalize(G)
    n = T.shape[0]
    gpu = _diffusion_nn_gpu(T, k, max_iterations, block)
    if gpu is not None:
        return gpu
    out = np.empty((n, k), dtype=np.int64)
    for a in range(0, n, block):
        b = min(n, a + block)
        P = T[a:b]
        M = P.copy()
        for _ in range(2):
            P = P @ T
            M = M + P
        i = 4
        while ((M > 0).sum(axis=1).min() < k + 1) and i < max_iterations:
            P = P @ T
            M = M + P
            i += 1
        if (M > 0).sum(axis=1).min() < k + 1:
            raise NeighborsError(f"menos de {k} vizinhos em {max_iterations} passos de difusão")
        D = M.toarray()
        D[np.arange(b - a), np.arange(a, b)] = 0.0
        idx = np.argpartition(D, -k, axis=1)[:, -k:]
        order = np.argsort(-np.take_along_axis(D, idx, axis=1), axis=1, kind="stable")
        out[a:b] = np.take_along_axis(idx, order, axis=1)
    return out


def _diffusion_nn_gpu(T, k: int, max_iterations: int, block: int):
    """A mesma conta de `diffusion_nn`, na GPU (float64), quando houver uma.

    Nas saídas de grafo (BBKNN), depois de ``diffusion_conn`` cada linha tem
    milhares de vizinhos e T³ fica quase denso: na CPU (scipy, uma thread) são
    ~100 s por bloco de 500 linhas no rótulo CD4 naive. Devolve None sem GPU.
    """
    try:
        import torch
        if not torch.cuda.is_available() or T.shape[0] < 2000 or T.nnz / T.shape[0] < 500:
            return None                # grafo esparso (embeddings): a CPU é mais rápida
    except Exception:
        return None
    dev = torch.device("cuda")
    Tt = sp.csr_matrix(T.T)
    tt = torch.sparse_csr_tensor(torch.from_numpy(Tt.indptr.astype(np.int64)),
                                 torch.from_numpy(Tt.indices.astype(np.int64)),
                                 torch.from_numpy(Tt.data.astype(np.float64)), size=Tt.shape, device=dev)
    n = T.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for a in range(0, n, block):
        b = min(n, a + block)
        P = torch.from_numpy(T[a:b].toarray().T.copy()).to(dev)      # n × bloco (transposto)
        M = P.clone()
        for _ in range(2):
            P = tt @ P                                                # (T_blocoᵀ ← Tᵀ · ...)
            M += P
        i = 4
        while int((M > 0).sum(dim=0).min()) < k + 1 and i < max_iterations:
            P = tt @ P
            M += P
            i += 1
        if int((M > 0).sum(dim=0).min()) < k + 1:
            raise NeighborsError(f"menos de {k} vizinhos em {max_iterations} passos de difusão")
        M[torch.arange(a, b, device=dev), torch.arange(b - a, device=dev)] = 0.0
        out[a:b] = torch.topk(M.T, k, dim=1, sorted=True).indices.cpu().numpy()
        del P, M
    del tt
    torch.cuda.empty_cache()
    return out


def _kbet_rejection(nn: np.ndarray, rows: np.ndarray, codes: np.ndarray, freq: np.ndarray, k0: int,
                    alpha: float = 0.05) -> float:
    """Taxa de rejeição do teste qui-quadrado do kBET.

    Vizinhança = a própria célula + os k0 − 1 primeiros vizinhos (como no
    ``kBET`` do R); esperado = frequência do lote no rótulo × k0. Testa todas
    as células, em vez de 100 sorteios de 10% (mesmo valor esperado).
    """
    env = np.column_stack([nn[:, : k0 - 1], rows])
    B = len(freq)
    counts = np.zeros((len(rows), B))
    np.add.at(counts, (np.repeat(np.arange(len(rows)), env.shape[1]), codes[env.ravel()]), 1.0)
    exp = freq * env.shape[1]
    chi = ((counts - exp) ** 2 / exp).sum(axis=1)
    p = stats.chi2.sf(chi, B - 1)
    return float(np.mean(p < alpha))


def kbet(adata, batch_key: str, label_key: str, type_: str, embed: str | None = None,
         alpha: float = 0.05, return_df: bool = False):
    """kBET por rótulo, como ``scib.me.kBET`` (escalado: 1 = lotes bem misturados).

    Grafo de 50 vizinhos sobre o embedding, ou difusão do grafo para as saídas
    de grafo (``scib.metrics.utils.diffusion_conn``); por rótulo, k0 = ¼ do
    tamanho médio do lote, limitado a 10–70; componentes conexos pequenos
    (< 3·k0) ficam de fora se somarem ≤ 25% das células, senão a taxa de
    rejeição do rótulo é 1, como no ``scib``.
    """
    from scipy.sparse.csgraph import connected_components

    diffusion_conn = _scib().metrics.utils.diffusion_conn

    if type_ != "knn":
        tmp = sc.pp.neighbors(adata, n_neighbors=50, use_rep=embed, copy=True)
        G_all = tmp.obsp["connectivities"]
    else:
        tmp = diffusion_conn(adata, min_k=50, copy=True)
        G_all = tmp.uns["neighbors"]["diffusion_connectivities"]
    G_all = sp.csr_matrix(G_all)
    labels = adata.obs[label_key].astype(str).to_numpy()
    batches = adata.obs[batch_key].astype(str).to_numpy()
    rows = []
    for lab in pd.unique(labels):
        m = np.flatnonzero(labels == lab)
        b = pd.Categorical(batches[m])
        if len(m) < 10 or len(b.categories) < 2:
            rows.append({"rotulo": lab, "celulas": len(m), "k0": np.nan, "rejeicao": np.nan})
            continue
        bc = np.bincount(b.codes)
        k0 = int(min(70, max(10, int(np.floor(bc.mean() / 4)))))
        freq = bc / bc.sum()
        G = G_all[m][:, m]
        n_comp, comp = connected_components(G, connection="strong")
        keep = np.arange(len(m))
        if n_comp > 1:
            size = np.bincount(comp)
            keep = np.flatnonzero(size[comp] >= 3 * k0)
            if len(keep) / len(m) < 0.75:
                rows.append({"rotulo": lab, "celulas": len(m), "k0": k0, "rejeicao": 1.0})
                continue
        try:
            nn = diffusion_nn(G[keep][:, keep], k=k0)
            rate = _kbet_rejection(keep[nn], keep, b.codes, freq, k0, alpha)
        except NeighborsError:
            rate = 1.0
        rows.append({"rotulo": lab, "celulas": len(m), "k0": k0, "rejeicao": rate})
    df = pd.DataFrame(rows)
    score = 1.0 - float(np.nanmean(df["rejeicao"]))
    return (score, df) if return_df else score


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

#: Métricas do artigo, nos dois grupos, com os nomes do artigo.
BATCH_METRICS = ["PCR batch", "Batch ASW", "Graph iLISI", "Graph connectivity", "kBET"]
BIO_METRICS = ["NMI cluster/label", "ARI cluster/label", "Cell type ASW", "Isolated label F1",
               "Isolated label silhouette", "Graph cLISI", "Cell cycle conservation", "HVG conservation"]
#: Métricas do estudo (fora do score do artigo).
STUDY_METRICS = ["separacao_estimulo", "profundidade_vizinhanca"]
#: O que o artigo não calcula para saídas de grafo (sem embedding nem matriz).
NOT_FOR_GRAPH = {"PCR batch", "Batch ASW", "Cell type ASW", "Isolated label silhouette", "Cell cycle conservation"}


def _graph(adata_like, res: dict, run: Run):
    """AnnData com ``X_emb`` (embedding, ou PCA da matriz corrigida) e o grafo de 15 vizinhos."""
    a = ad.AnnData(obs=adata_like.obs.copy())
    if run.output == "genes":
        a = ad.AnnData(X=res["X"], obs=adata_like.obs.copy(), var=pd.DataFrame(index=adata_like.var_names))
        sc.tl.pca(a, n_comps=config.N_PCS, random_state=config.RANDOM_STATE)
        a.obsm["X_emb"] = a.obsm["X_pca"]
    elif run.output == "embedding":
        a.obsm["X_emb"] = res["emb"]
    if run.output == "grafo":
        a.obsp["connectivities"] = res["conn"]
        a.obsp["distances"] = res["dist"]
        a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                              "params": {"method": "bbknn"}}
    else:
        sc.pp.neighbors(a, n_neighbors=N_NEIGHBORS, use_rep="X_emb", random_state=SEED)
    return a


def depth_in_neighborhood(conn, total_counts, library) -> float:
    """ρ de Spearman entre a profundidade de cada célula e a média dos vizinhos.

    A profundidade é log10(UMIs) menos a mediana da biblioteca, para que a
    diferença entre bibliotecas não conte. Perto de 0 = vizinhança sem
    organização por profundidade.
    """
    x = np.log10(np.asarray(total_counts, float))
    lib = pd.Series(np.asarray(library).astype(str))
    x = x - lib.map(pd.Series(x).groupby(lib).median()).to_numpy()
    A = _row_normalize((sp.csr_matrix(conn) > 0).astype(np.float64))
    return float(stats.spearmanr(x, A @ x).statistic)


def _timed(times: dict, name: str, f, *args, **kw):
    t0 = time.perf_counter()
    try:
        v = float(f(*args, **kw))
    except Exception as e:                      # uma métrica que falha não derruba as outras
        times[name] = {"tempo_s": time.perf_counter() - t0, "erro": f"{type(e).__name__}: {e}"[:300]}
        return np.nan
    times[name] = {"tempo_s": time.perf_counter() - t0}
    return v


def _suffix(tag: str | None) -> str:
    return "" if tag is None else f"__{tag}"


def evaluate(key: str, inputs: dict, umap: bool = True, labels: pd.Series | None = None,
             tag: str | None = None) -> dict:
    """Métricas de uma execução, nas células com rótulo; grava ``metricas.json``.

    O grafo de 15 vizinhos de todas as células (e a UMAP sobre ele) vai para
    ``grafo15.npz`` / ``umap.npy``: é o que as figuras e o P08 usam.

    ``labels`` (indexado por ``cell_key``) troca o rótulo biológico do Azimuth
    por outro (P12); o resultado vai para ``metricas__<tag>.json``, e o grafo e
    a UMAP não são regravados.
    """
    scib = _scib()
    run = RUN[key]
    pre = inputs["pre_esc"] if run.scaled else inputs["pre"]
    if labels is not None:
        if tag is None:
            raise ValueError("rótulo próprio exige uma etiqueta (tag)")
        pre = pre.copy()
        pre.obs[LABEL] = pd.Categorical(labels.reindex(pre.obs["cell_key"].to_numpy()).to_numpy())
        umap = False
    res = read_output(key)
    times: dict = {}
    full_graph = _graph(pre, res, run)
    d = run_dir(key)
    conn = sp.csr_matrix(full_graph.obsp["connectivities"])
    dist = sp.csr_matrix(full_graph.obsp["distances"])
    if tag is None:
        sp.save_npz(d / "grafo15_conn.npz", conn)
        sp.save_npz(d / "grafo15_dist.npz", dist)
    if umap:
        t0 = time.perf_counter()
        sc.tl.umap(full_graph, random_state=SEED)
        np.save(d / "umap.npy", full_graph.obsm["X_umap"].astype(np.float32))
        times["UMAP"] = {"tempo_s": time.perf_counter() - t0}

    obs = pre.obs
    m: dict = {}
    sep = pl.within_patient_separation(conn, obs[BATCH].astype(str).to_numpy(), obs["patient"].to_numpy())
    m["separacao_estimulo"] = float(sep["separacao"].median())
    m["profundidade_vizinhanca"] = _timed(times, "profundidade", depth_in_neighborhood, conn,
                                          obs["total_counts"].to_numpy(), obs[BATCH].to_numpy())

    lab = obs[LABEL].notna().to_numpy()
    pre_s = pre[lab].copy()
    pre_s.obs[LABEL] = pre_s.obs[LABEL].cat.remove_unused_categories()
    if run.output == "grafo":
        a = ad.AnnData(obs=pre_s.obs.copy())
        idx = np.flatnonzero(lab)
        a.obsp["connectivities"] = conn[idx][:, idx]
        a.obsp["distances"] = dist[idx][:, idx]
        a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                              "params": {"method": "bbknn"}}
        type_, embed = "knn", None
    else:
        sub = {k: v[lab] for k, v in res.items()}
        a = _graph(pre_s, sub, run)
        if run.output == "genes":
            a.X = sub["X"]
        type_, embed = "embed", "X_emb"
    a.obs[LABEL] = pre_s.obs[LABEL]
    a.obs[BATCH] = pre_s.obs[BATCH].cat.remove_unused_categories()

    graph_ok = run.output == "grafo"
    if not graph_ok:
        m["PCR batch"] = _timed(times, "PCR batch", scib.me.pcr_comparison, pre_s, a, covariate=BATCH,
                                embed=embed, n_comps=config.N_PCS)
        m["Batch ASW"] = _timed(times, "Batch ASW", scib.me.silhouette_batch, a, batch_key=BATCH, label_key=LABEL,
                                embed=embed, verbose=False)
        m["Cell type ASW"] = _timed(times, "Cell type ASW", scib.me.silhouette, a, label_key=LABEL, embed=embed)
        m["Isolated label silhouette"] = _timed(times, "Isolated label silhouette", scib.me.isolated_labels_asw, a,
                                                label_key=LABEL, batch_key=BATCH, embed=embed, verbose=False)
        full_s = inputs["full"][lab].copy()
        m["Cell cycle conservation"] = _timed(times, "Cell cycle conservation", scib.me.cell_cycle, full_s, a,
                                              batch_key=BATCH, embed=embed, organism="human")
    m["Graph iLISI"] = _timed(times, "Graph iLISI", scib.me.ilisi_graph, a, batch_key=BATCH, type_=type_,
                              use_rep="X_emb")
    m["Graph cLISI"] = _timed(times, "Graph cLISI", scib.me.clisi_graph, a, label_key=LABEL, type_=type_,
                              use_rep="X_emb")
    m["Graph connectivity"] = _timed(times, "Graph connectivity", scib.me.graph_connectivity, a, label_key=LABEL)
    m["kBET"] = _timed(times, "kBET", kbet, a, batch_key=BATCH, label_key=LABEL, type_=type_, embed=embed)

    def _nmi_ari():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scib.me.cluster_optimal_resolution(a, label_key=LABEL, cluster_key="cluster", verbose=False)
        return 0.0

    _timed(times, "clusterização ótima (NMI)", _nmi_ari)
    if "cluster" in a.obs:
        m["NMI cluster/label"] = _timed(times, "NMI", scib.me.nmi, a, "cluster", LABEL)
        m["ARI cluster/label"] = _timed(times, "ARI", scib.me.ari, a, "cluster", LABEL)
        cluster_res = pd.Series(a.obs["cluster"].astype(str).to_numpy(), index=a.obs_names)
    else:
        cluster_res = None
    m["Isolated label F1"] = _timed(times, "Isolated label F1", scib.me.isolated_labels_f1, a, label_key=LABEL,
                                    batch_key=BATCH, embed=None, verbose=False)
    if run.output == "genes":
        m["HVG conservation"] = _timed(times, "HVG conservation", scib.me.hvg_overlap, pre_s, a,
                                       batch_key=BATCH, n_hvg=500)
    out = {"execucao": key, "assinatura": inputs["assinatura"], "metricas": m, "tempos": times,
           "separacao_por_biblioteca": sep["separacao"].to_dict(), "rotulo": tag or "azimuth",
           "celulas_avaliadas": int(lab.sum()), "criado": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / f"metricas{_suffix(tag)}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    if cluster_res is not None:
        cluster_res.to_frame("cluster").to_parquet(d / f"clusters_otimos{_suffix(tag)}.parquet")
    return out


def is_evaluated(key: str, signature: str, tag: str | None = None) -> bool:
    m = run_dir(key) / f"metricas{_suffix(tag)}.json"
    if not m.exists() or json.loads(m.read_text()).get("assinatura") != signature:
        return False
    a = run_dir(key) / "ajuste.json"
    return a.exists() and a.stat().st_mtime <= m.stat().st_mtime


# ─────────────────────────────────────────────────────────────────────────────
# Leitura e agregação
# ─────────────────────────────────────────────────────────────────────────────

def collect(tag: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas brutas (execução × métrica) e metadados dos ajustes, do cache.

    ``tag``: as métricas recalculadas com outro rótulo (P12); None = Azimuth (P07).
    """
    rows, meta = [], []
    for r in RUNS:
        mp, ap = run_dir(r.key) / f"metricas{_suffix(tag)}.json", run_dir(r.key) / "ajuste.json"
        if ap.exists():
            meta.append({"key": r.key, **json.loads(ap.read_text())})
        if mp.exists():
            j = json.loads(mp.read_text())
            erros = {k: v["erro"] for k, v in j["tempos"].items() if "erro" in v}
            rows.append({"key": r.key, **j["metricas"], "erros": json.dumps(erros, ensure_ascii=False),
                         "tempo_metricas_s": sum(v["tempo_s"] for v in j["tempos"].values())})
    raw = pd.DataFrame(rows).set_index("key") if rows else pd.DataFrame()
    return raw.reindex([r.key for r in RUNS if r.key in raw.index]), pd.DataFrame(meta).set_index("key")


def minmax(raw: pd.DataFrame, cols) -> pd.DataFrame:
    x = raw[cols].astype(float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo).replace(0, np.nan)


def aggregate(raw: pd.DataFrame, w_batch: float = 0.4, scaling: str = "minmax", drop=()) -> pd.DataFrame:
    """Scores do artigo: métricas escalonadas, médias por grupo e o geral.

    ``scaling``: ``"minmax"`` (o do artigo), ``"zscore"`` (a checagem do
    artigo) ou ``"none"`` (métricas brutas, todas já em 0–1); ``drop``:
    métricas deixadas de fora (análise de sensibilidade).
    """
    bmets = [m for m in BATCH_METRICS if m not in drop]
    omets = [m for m in BIO_METRICS if m not in drop]
    cols = bmets + omets
    x = raw[cols].astype(float)
    if scaling == "minmax":
        s = minmax(raw, cols)
    elif scaling == "zscore":
        s = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
    else:
        s = x
    out = pd.DataFrame(index=raw.index)
    out["Batch correction"] = s[bmets].mean(axis=1, skipna=True)
    out["Bio conservation"] = s[omets].mean(axis=1, skipna=True)
    out["Overall score"] = w_batch * out["Batch correction"] + (1 - w_batch) * out["Bio conservation"]
    return out


def stimulus_retention(raw: pd.DataFrame, reference: str = REFERENCE) -> pd.Series:
    """Fração da separação NS × ML (dentro do paciente) do não integrado que a execução mantém."""
    return raw["separacao_estimulo"] / raw.loc[reference, "separacao_estimulo"]


def choose(summary: pd.DataFrame, retention: pd.Series, current: str = CURRENT,
           tie: float = 0.02, veto: float = 0.10) -> dict:
    """Regra de escolha, fixada antes de ver os resultados.

    1. Elegíveis: métodos que não usam rótulo, fora o não integrado.
    2. Veto: preservar menos estímulo que a integração atual (Harmony do P05)
       por mais de ``veto`` (fração do não integrado).
    3. Entre os não vetados, os que ficam a menos de ``tie`` do maior score
       geral empatam; no empate vence quem preserva mais estímulo.
    """
    elig = [k for k in summary.index if not RUN[k].uses_labels and not RUN[k].method.startswith("Não integrado")]
    floor = retention[current] - veto
    vetoed = [k for k in elig if retention[k] < floor]
    ok = [k for k in elig if k not in vetoed]
    best = summary.loc[ok, "Overall score"].max()
    tied = [k for k in ok if summary.loc[k, "Overall score"] >= best - tie]
    pick = max(tied, key=lambda k: (retention[k], summary.loc[k, "Overall score"]))
    top = summary.loc[elig, "Overall score"].idxmax()
    return {"escolhido": pick, "melhor_geral_elegivel": top, "empatados": tied, "vetados": vetoed,
            "piso_estimulo": float(floor), "melhor_score": float(best), "tolerancia_empate": tie, "veto": veto}
