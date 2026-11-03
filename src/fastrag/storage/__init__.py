"""Storage: one source of truth (the document store) plus derived, rebuildable indexes.

The document store owns documents, chunks and embeddings. The vector and keyword
indexes are projections of it: they can always be rebuilt from the store, so a delete
or re-ingest only has to be correct in one place.
"""

from fastrag.storage.base import DocumentStore, KeywordIndex, VectorIndex
from fastrag.storage.bm25 import BM25Index
from fastrag.storage.indexes import SearchIndexes
from fastrag.storage.sqlite import SQLiteDocumentStore
from fastrag.storage.vector import NumpyVectorIndex

__all__ = [
    "BM25Index",
    "DocumentStore",
    "KeywordIndex",
    "NumpyVectorIndex",
    "SQLiteDocumentStore",
    "SearchIndexes",
    "VectorIndex",
]
