import unittest
import numpy as np

from agent.embeddings import embed_texts, embedding_dimension


class EmbeddingTests(unittest.TestCase):
    def test_empty_list_returns_empty_2d_array(self):
        res = embed_texts([])
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (0, embedding_dimension()))

    def test_single_text_dimension_matches_model_dimension(self):
        res = embed_texts(["hello test"])
        self.assertEqual(res.shape, (1, embedding_dimension()))
        # Check normalized (L2 norm should be approx 1)
        norm = np.linalg.norm(res[0])
        self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
