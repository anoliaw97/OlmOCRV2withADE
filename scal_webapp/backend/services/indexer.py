from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class LocalHybridIndex:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.store_dir / "tfidf_vectorizer.joblib"
        self.mat_path = self.store_dir / "tfidf_matrix.joblib"
        self.meta_path = self.store_dir / "chunk_metadata.joblib"
        self.text_path = self.store_dir / "chunk_texts.joblib"

        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.chunk_metadata: list[dict] = []
        self.chunk_texts: list[str] = []

    def build(self, chunks: list[dict]):
        self.chunk_texts = [c["chunk_text"] for c in chunks]
        self.chunk_metadata = [c["metadata"] for c in chunks]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.chunk_texts)

        joblib.dump(self.vectorizer, self.vec_path)
        joblib.dump(self.matrix, self.mat_path)
        joblib.dump(self.chunk_metadata, self.meta_path)
        joblib.dump(self.chunk_texts, self.text_path)

    def load(self):
        if not self.vec_path.exists():
            return False
        self.vectorizer = joblib.load(self.vec_path)
        self.matrix = joblib.load(self.mat_path)
        self.chunk_metadata = joblib.load(self.meta_path)
        self.chunk_texts = joblib.load(self.text_path)
        return True

    def search(self, query: str, top_k: int = 6, filters: dict | None = None):
        if self.vectorizer is None or self.matrix is None:
            if not self.load():
                return []

        qv = self.vectorizer.transform([query])
        sims = linear_kernel(qv, self.matrix).flatten()

        idxs = sims.argsort()[::-1]
        results = []
        for idx in idxs:
            score = float(sims[idx])
            if score <= 0:
                continue
            meta = self.chunk_metadata[idx]
            if filters:
                ok = True
                for k, v in filters.items():
                    if v in (None, ""):
                        continue
                    if str(meta.get(k, "")).lower() != str(v).lower():
                        ok = False
                        break
                if not ok:
                    continue
            results.append({"score": score, "text": self.chunk_texts[idx], "metadata": meta})
            if len(results) >= top_k:
                break
        return results
