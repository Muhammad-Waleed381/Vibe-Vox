from __future__ import annotations

import unittest

from semantic_chunker import chunk_text, load_tokenizer


class TestSemanticChunker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_tokenizer()

    def test_chunks_respect_max_tokens(self) -> None:
        text = (
            "This is a calm sentence. "
            "Here is another one that adds more context to the narrative. "
            "Finally, a third sentence that should be grouped sensibly."
        )
        chunks = chunk_text(text, self.tokenizer, max_tokens=150)
        self.assertTrue(len(chunks) >= 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 150)

    def test_long_sentence_splits(self) -> None:
        text = (
            "This is a very long sentence, with multiple clauses, that keeps going, "
            "and it should be split, because the token budget is tiny."
        )
        chunks = chunk_text(text, self.tokenizer, max_tokens=20)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 20)

    def test_metadata_offsets(self) -> None:
        text = "Hello world. Another sentence."
        chunks = chunk_text(text, self.tokenizer, max_tokens=150)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[-1].end_char, len(text))


if __name__ == "__main__":
    unittest.main()
