"""Retrieval half of the chatbot: load notes, split into chunks, search them."""
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    source: str  # file the text came from
    text: str


def load_chunks(folder, size=800, overlap=150):
    """Read every .txt/.md file under `folder` and split it into overlapping chunks."""
    folder = Path(folder)
    chunks = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for start in range(0, len(text), size - overlap):
            piece = text[start : start + size].strip()
            if piece:
                chunks.append(Chunk(str(path.relative_to(folder)), piece))
    return chunks


class Index:
    """TF-IDF search over chunks. Swap this class for an embeddings index later."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query, k=4):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        best = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best if scores[i] > 0]
