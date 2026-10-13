import numpy as np
import unittest

from agent.indexer import collection_needs_rebuild


class FakeCollection:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def count(self):
        return len(self._embeddings)

    def get(self, **kwargs):
        return {"embeddings": self._embeddings[:1]}


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name):
        if self.collection is None:
            raise RuntimeError("missing collection")
        return self.collection


class CollectionDimensionTests(unittest.TestCase):
    def test_rebuilds_collection_when_embedding_dimension_changes(self):
        collection = FakeCollection([np.zeros(384)])

        self.assertTrue(collection_needs_rebuild(FakeClient(collection), "code_repo", 1024))

    def test_keeps_collection_when_embedding_dimension_matches(self):
        collection = FakeCollection(np.zeros((1, 1024)))

        self.assertFalse(collection_needs_rebuild(FakeClient(collection), "code_repo", 1024))

    def test_rebuilds_when_collection_is_missing(self):
        self.assertTrue(collection_needs_rebuild(FakeClient(None), "code_repo", 1024))

    def test_does_not_rebuild_empty_collection(self):
        collection = FakeCollection([])

        self.assertFalse(collection_needs_rebuild(FakeClient(collection), "code_repo", 1024))
