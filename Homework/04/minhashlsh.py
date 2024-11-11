import pandas as pd
import numpy as np
from numpy.typing import NDArray

from minhash import MinHash


class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        super().__init__(num_permutations, threshold)
        self.num_buckets = num_buckets

    def get_buckets(self, minhash: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        """
        non_equal_division = bool(self.num_permutations % self.num_buckets)
        rows_per_bucket = self.num_permutations // self.num_buckets + non_equal_division

        buckets = []
        for i in range(self.num_buckets):
            start_row = i * rows_per_bucket
            end_row = start_row + rows_per_bucket
            bucket = minhash[start_row:end_row, :]
            buckets.append(bucket) if bucket.shape[0] > 0 else None

        if non_equal_division:
            buckets[-1] = np.vstack(
                (
                    buckets[-1],
                    np.zeros(
                        (rows_per_bucket - buckets[-1].shape[0], minhash.shape[1])
                    ),
                )
            )

        return np.array(buckets)

    def get_similar_candidates(self, buckets: NDArray[np.int_], minhash: NDArray[np.int_]) -> list[tuple[int, int]]:
        similar_candidates = set()
        num_docs = buckets.shape[2]

        for bucket in buckets[:-1]:
            hash_table: dict[tuple[int, int], list[int]] = {}
            for doc_id in range(num_docs):
                bucket_signature = tuple(bucket[:, doc_id])
                if bucket_signature in hash_table:
                    for other_doc in hash_table[bucket_signature]:
                        if (
                            self.get_minhash_similarity(
                                minhash[:, doc_id], minhash[:, other_doc]
                            )
                            > self.threshold
                        ):
                            similar_candidates.add(
                                (min(doc_id, other_doc), max(doc_id, other_doc))
                            )
                    hash_table[bucket_signature].append(doc_id)
                else:
                    hash_table[bucket_signature] = [doc_id]

        return list(similar_candidates)

    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple[int, int]]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        buckets = self.get_buckets(minhash)
        similar_candidates = self.get_similar_candidates(buckets, minhash)
        return set(similar_candidates)
