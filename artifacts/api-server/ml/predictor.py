"""Инференс для ML-сервиса: вероятность банкротства + объяснение (SHAP-вклады признаков и вклады слов).

Зависимости: numpy, pandas, pyarrow, scipy, lightgbm. Артефакты собирает export_artifacts.py.

    p = Predictor("artifacts", data_dir="C:/haha/Tcache")   # data_dir нужен для запросов по index
    p.by_index(239134)
    p.by_features({"feature_0": 0.5, ...}, word_count="кредит 3 займ 1", id=1)
    p.by_csv("one_row.csv")

SHAP-вклад (shap_value) — на сколько этот признак сдвинул прогноз ИМЕННО этого человека относительно
среднего прогноза (shap_base). > 0 — в сторону банкротства, < 0 — в сторону надёжности. Единицы — лог-шансы,
сумма всех вкладов + shap_base = итоговый «сырой» скор модели. Считается встроенным в LightGBM TreeSHAP
(тот же алгоритм, что в библиотеке shap, без лишней зависимости).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

BASELINE = "nonbankrupt_mean"   # что отдавать в baseline_value: среднее у не банкротов (так просил фронт)


# ---------- текст ----------
def parse_word_count(s: Any) -> dict[str, int]:
    """'кредит 3 займ 1' / {'кредит': 3} / None → {слово: число}."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return {}
    if isinstance(s, dict):
        return {str(w): int(c) for w, c in s.items() if int(c) > 0}
    t = str(s).split()
    d: dict[str, int] = {}
    for w, c in zip(t[0::2], t[1::2]):
        try:
            n = int(float(c))
        except ValueError:
            continue
        if n > 0:
            d[w] = d.get(w, 0) + n
    return d


def tfidf_csr(X: sp.csr_matrix, idf: np.ndarray, binary: bool = False) -> sp.csr_matrix:
    """То же, что TfidfTransformer(sublinear_tf=True, norm='l2') с готовым idf. Структура матрицы сохраняется."""
    X = X.tocsr().astype(np.float32, copy=True)
    X.eliminate_zeros()
    d = np.ones_like(X.data) if binary else X.data
    d = (np.log(d) + 1.0) * idf[X.indices]
    row_len = np.diff(X.indptr)
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int32), row_len)
    norm = np.sqrt(np.bincount(rows, weights=d.astype(np.float64) ** 2, minlength=X.shape[0]))
    norm[norm == 0] = 1.0
    X.data = (d / norm[rows]).astype(np.float32)
    return X


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def feature_label(name: str) -> str:
    if name == "dup_count":
        return "Сколько раз такой же профиль встречается в данных"
    if name == "dup_n_tr":
        return "Сколько совпадающих профилей с известным исходом"
    if name == "dup_target":
        return "Доля банкротов среди совпадающих профилей"
    if name.startswith("txt_"):
        return "Текстовый риск (содержание интернет-активности)"
    if name.startswith("svd_"):
        return f"Тема интернет-активности №{name.split('_')[1]}"
    if name == "txt_n_words":
        return "Сколько разных слов в интернет-активности"
    if name == "txt_total_log":
        return "Объём интернет-активности"
    return f"{name} (анонимизированный признак)"


def _num(v: Any) -> float | None:
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


class Predictor:
    def __init__(self, artifacts: str | Path = "artifacts", data_dir: str | Path | None = None):
        A = Path(artifacts)
        self.A = A
        self.cfg = json.load(open(A / "config.json", encoding="utf-8"))
        self.names: list[str] = self.cfg["features"]
        self.base: list[str] = self.cfg["base_features"]
        self.hash_feats: list[str] = self.cfg["hash_features"]
        self._hash_pos = {f: i for i, f in enumerate(self.hash_feats)}

        self.models: dict[int, list[lgb.Booster]] = {}
        for f in sorted((A / "models").glob("fold*_seed*.txt")):
            k = int(f.stem.split("_")[0][4:])
            self.models.setdefault(k, []).append(lgb.Booster(model_file=str(f)))
        self.all_models = [b for k in sorted(self.models) for b in self.models[k]]

        self.stats = pd.read_parquet(A / "feature_stats.parquet").set_index("feature")
        self.by_idx = pd.read_parquet(A / "by_index.parquet").set_index("index")
        self.ref_scores = np.load(A / "ref_scores.npy")

        self.dup_map: dict[int, tuple[float, float, float]] = {}
        if self.cfg["dup"]:
            t = pd.read_parquet(A / "dup_table.parquet")
            self.dup_map = dict(zip(t["h"].astype("uint64").tolist(),
                                    zip(t["count_all"].tolist(), t["n_tr"].tolist(), t["sum_tr"].tolist())))

        self.has_text = bool(self.cfg["text"]) or bool(self.cfg["svd"])
        if self.has_text:
            T = A / "text"
            self.vocab_full: list[str] = json.load(open(T / "vocab_full.json", encoding="utf-8"))
            self.word_pos = {w: i for i, w in enumerate(self.vocab_full)}
            self.word_stats = pd.read_parquet(T / "word_stats.parquet").set_index("word")
        if self.cfg["text"]:
            self.lr_cols = np.load(T / "lr_cols.npy")
            self.lr_idf = np.load(T / "lr_idf.npy")
            self.lr_coef = np.load(T / "lr_coef.npy")            # фолды × слова
            self.lr_icpt = np.load(T / "lr_intercept.npy")
            self.lr_coef_mean = self.lr_coef.mean(0)
            tw = pd.read_parquet(T / "top_words_by_index.parquet").sort_values("index", kind="stable")
            self.tw_index = tw["index"].to_numpy()
            self.tw_word = tw["word"].to_numpy()
            self.tw_count = tw["count"].to_numpy()
            self.tw_impact = tw["impact"].to_numpy()
        if self.cfg["svd"]:
            self.svd_comp = np.load(T / "svd_components.npy")
            self.svd_idf = np.load(T / "svd_idf.npy")

        self.rows: pd.DataFrame | None = None
        if data_dir is not None:
            D = Path(data_dir)
            cols = ["index"] + self.base
            self.rows = pd.concat([pd.read_parquet(D / "TRAIN.parquet", columns=cols),
                                   pd.read_parquet(D / "TEST.parquet", columns=cols)]).set_index("index")

    # ---------- публичное API ----------
    @property
    def info(self) -> dict:
        return {"model": self.cfg["model"], "version": self.cfg["version"], "oof_auc": self.cfg.get("oof_auc"),
                "n_features": len(self.names), "text": bool(self.cfg["text"]), "dup": self.cfg["dup"],
                "index_lookup": bool("pred" in self.by_idx.columns and self.by_idx["pred"].notna().all()),
                "index_explanations": bool(self.rows is not None)}

    def by_index(self, index: int, top_k: int = 5, n_words: int = 10) -> dict:
        index = int(index)
        if index not in self.by_idx.index:
            raise KeyError(f"index {index} не найден ни в TRAIN, ни в TEST")
        e = self.by_idx.loc[index]
        if self.rows is None:
            if "pred" not in e or pd.isna(e["pred"]):
                raise RuntimeError("Для этого index нет сохранённого прогноза")
            prob = float(e["pred"])
            result = {
                "id": index,
                "probability": round(prob, 6),
                "percentile": round(float(np.searchsorted(self.ref_scores, prob) / len(self.ref_scores) * 100), 1),
                "top_features": [],
                "top_words": self._index_words(index, n_words),
                "source": "train" if e["part"] == "train" else "test",
                "warnings": [
                    "Для index используется сохранённый прогноз. Исходные табличные признаки не приложены, поэтому SHAP-объяснение недоступно."
                ],
            }
            if e["part"] == "train" and pd.notna(e.get("target")):
                result["true_target"] = int(e["target"])
            return result
        vals = dict(zip(self.base, self.rows.loc[index, self.base].to_numpy(np.float32)))
        for c in self.names:
            if c not in vals:
                vals[c] = e[c]
        x = np.array([vals[c] for c in self.names], dtype=np.float64)
        is_train = e["part"] == "train"
        # train: модель того фолда, которая этого человека НЕ видела → честный прогноз (= OOF)
        models = self.models[int(e["fold"])] if is_train else self.all_models
        res = {"id": index, **self._explain(x, models, top_k)}
        res["source"] = "train" if is_train else "test"
        if is_train:
            res["true_target"] = int(e["target"])
        res["top_words"] = self._index_words(index, n_words)
        return res

    def by_features(self, features: dict[str, Any], word_count: Any = None, id: Any = None,
                    top_k: int = 5, n_words: int = 10) -> dict:
        warnings = []
        dropped = set(self.cfg.get("dropped_features", []))
        unknown = [k for k in features if k not in self._hash_pos and k not in dropped]
        if unknown:
            warnings.append(f"неизвестные признаки проигнорированы: {unknown[:10]}{'…' if len(unknown) > 10 else ''}")
        vec = np.full(len(self.hash_feats), np.nan, np.float32)
        for k, v in features.items():
            j = self._hash_pos.get(k)
            if j is not None and v is not None and v != "":
                vec[j] = float(v)
        n_missing = int(np.isnan(vec).sum())
        if n_missing:
            warnings.append(f"не передано признаков: {n_missing} (считаются отсутствующими)")
        vals: dict[str, float] = dict(zip(self.hash_feats, vec.astype(np.float64)))

        if self.cfg["dup"]:
            h = int(pd.util.hash_pandas_object(pd.DataFrame(vec.reshape(1, -1), columns=self.hash_feats),
                                               index=False).iloc[0])
            cnt, n_tr, s_tr = self.dup_map.get(h, (0.0, 0.0, 0.0))
            vals.update(dup_count=cnt + 1.0, dup_n_tr=n_tr, dup_target=(s_tr / n_tr) if n_tr else np.nan)

        words = parse_word_count(word_count)
        Xc = self._counts_row(words) if self.has_text else None
        top_words: list[dict] = []
        if self.cfg["text"]:
            name = f"txt_{self.cfg['text']}"
            if Xc is not None and Xc.nnz:
                Xl = tfidf_csr(Xc[:, self.lr_cols], self.lr_idf, self.cfg["text_binary"])
                z = Xl @ self.lr_coef.T + self.lr_icpt              # (1, фолды)
                vals[name] = float(sigmoid(np.asarray(z)).mean())
                raw = Xc[:, self.lr_cols]
                top_words = self._words_from_row(Xl.indices, Xl.data * self.lr_coef_mean[Xl.indices],
                                                 raw.data, n_words)
            else:
                vals[name] = np.nan
                if word_count is not None:
                    warnings.append("ни одного известного модели слова — текст не учтён")
        if self.cfg["svd"]:
            if Xc is not None and Xc.nnz:
                Z = np.asarray(tfidf_csr(Xc, self.svd_idf) @ self.svd_comp.T).ravel()
                vals.update({f"svd_{i}": float(v) for i, v in enumerate(Z)})
                vals["txt_n_words"] = float(Xc.nnz)
                vals["txt_total_log"] = float(np.log1p(Xc.sum()))
            else:
                vals.update({f"svd_{i}": np.nan for i in range(self.svd_comp.shape[0])})
                vals["txt_n_words"] = vals["txt_total_log"] = np.nan

        x = np.array([vals.get(c, np.nan) for c in self.names], dtype=np.float64)
        res = {"id": id, **self._explain(x, self.all_models, top_k), "source": "new", "top_words": top_words}
        if warnings:
            res["warnings"] = warnings
        return res

    def by_csv(self, path_or_buf: Any, top_k: int = 5, n_words: int = 10) -> list[dict]:
        df = pd.read_csv(path_or_buf, sep=None, engine="python")
        out = []
        for _, r in df.iterrows():
            feats = {c: r[c] for c in df.columns if c.startswith("feature_")}
            rid = int(r["index"]) if "index" in df.columns and pd.notna(r["index"]) else None
            out.append(self.by_features(feats, r.get("word_count"), rid, top_k, n_words))
        return out

    # ---------- внутреннее ----------
    def _explain(self, x: np.ndarray, models: list[lgb.Booster], top_k: int) -> dict:
        X = x.reshape(1, -1)
        prob = float(np.mean([b.predict(X)[0] for b in models]))
        contrib = np.mean([b.predict(X, pred_contrib=True)[0] for b in models], axis=0)
        shap, base = contrib[:-1], float(contrib[-1])
        top = []
        for j in np.argsort(-np.abs(shap))[:top_k]:
            n = self.names[j]
            st = self.stats.loc[n]
            top.append({
                "name": n,
                "label": feature_label(n),
                "raw_value": _num(x[j]),
                "baseline_value": _num(st[BASELINE]),          # среднее у не банкротов
                "baseline_median": _num(st["nonbankrupt_median"]),
                "bankrupt_value": _num(st["bankrupt_mean"]),
                "bankrupt_median": _num(st["bankrupt_median"]),
                "shap_value": round(float(shap[j]), 5),
            })
        pct = float(np.searchsorted(self.ref_scores, prob) / len(self.ref_scores) * 100)
        return {"probability": round(prob, 6), "percentile": round(pct, 1), "shap_base": round(base, 5),
                "top_features": top}

    def _counts_row(self, words: dict[str, int]) -> sp.csr_matrix:
        pairs = sorted((self.word_pos[w], c) for w, c in words.items() if w in self.word_pos)
        idx = np.array([p for p, _ in pairs], np.int32)
        val = np.array([c for _, c in pairs], np.float32)
        return sp.csr_matrix((val, idx, np.array([0, len(idx)])), shape=(1, len(self.vocab_full)))

    def _words_from_row(self, lr_idx: np.ndarray, impact: np.ndarray, counts: np.ndarray, n: int) -> list[dict]:
        if len(impact) == 0:
            return []
        sel = np.argsort(-np.abs(impact))[:n]
        return [self._word_item(self.vocab_full[self.lr_cols[lr_idx[i]]], counts[i], impact[i]) for i in sel]

    def _index_words(self, index: int, n: int) -> list[dict]:
        if not self.cfg["text"]:
            return []
        lo = int(np.searchsorted(self.tw_index, index, "left"))
        hi = min(int(np.searchsorted(self.tw_index, index, "right")), lo + n)
        return [self._word_item(self.tw_word[i], self.tw_count[i], self.tw_impact[i]) for i in range(lo, hi)]

    def _word_item(self, w: str, count: float, impact: float) -> dict:
        ws = self.word_stats.loc[w] if w in self.word_stats.index else None
        return {
            "word": w,
            "count": int(count),
            "impact": round(float(impact), 5),     # вклад в текстовый риск: > 0 — к банкротству
            "share_bankrupt": None if ws is None else round(float(ws["share_bankrupt"]), 4),
            "share_nonbankrupt": None if ws is None else round(float(ws["share_nonbankrupt"]), 4),
        }
