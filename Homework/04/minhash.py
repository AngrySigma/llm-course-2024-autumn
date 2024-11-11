from typing import Union

import numpy as np
import pandas as pd
import re
from itertools import combinations

from numpy.typing import NDArray


class MinHash:
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        return re.sub("( )+|(\n)+", " ", text).lower()

    def tokenize(self, text: str) -> set[str]:
        text = self.preprocess_text(text)
        return set(text.split(" "))

    def get_occurrence_matrix(self, corpus_of_texts: list[str]) -> pd.DataFrame:
        """
        Получение матрицы вхождения токенов. Строки - это токены, столбы это id документов.
        id документа - нумерация в списке начиная с нуля
        """
        # TODO:
        true_corpus = [self.tokenize(text) for text in corpus_of_texts]
        tokens= set()
        for text in true_corpus:
            tokens.update(text)
        tokens_sorted = sorted(list(tokens))
        df = pd.DataFrame(0, index=tokens_sorted, columns=range(len(true_corpus)))
        for i, text in enumerate(true_corpus):
            for token in text:
                df.loc[token, i] = 1
        df.sort_index(inplace=True)
        return df

    def is_prime(self, a: int) -> bool:
        if a % 2 == 0:
            return a == 2
        d = 3
        while d * d <= a and a % d != 0:
            d += 2
        return d * d > a

    def get_new_index(self, x: int, permutation_index: int, prime_num_rows: int) -> int:
        """
        Получение перемешанного индекса.
        values_dict - нужен для совпадения результатов теста, а в общем случае используется рандом
        prime_num_rows - здесь важно, чтобы число было >= rows_number и было ближайшим простым числом к rows_number

        """
        values_dict = {"a": [3, 4, 5, 7, 8], "b": [3, 4, 5, 7, 8]}
        a = values_dict["a"][permutation_index]
        b = values_dict["b"][permutation_index]
        return (a * (x + 1) + b) % prime_num_rows

    def get_minhash_similarity(self, array_a: NDArray[np.int_], array_b: NDArray[np.int_]) -> float:
        """
        Вовзращает сравнение minhash для НОВЫХ индексов. То есть: приходит 2 массива minhash:
            array_a = [1, 2, 1, 5, 3]
            array_b = [1, 3, 1, 4, 3]

            на выходе ожидаем количество совпадений/длину массива, для примера здесь:
            у нас 3 совпадения (1,1,3), ответ будет 3/5 = 0.6
        """
        # TODO:
        matches = [1 for i in range(len(array_a)) if array_a[i] == array_b[i]]

        return len(matches) / len(array_a)

    def get_similar_pairs(self, min_hash_matrix: NDArray[np.int_]) -> list[tuple[int, int]]:
        """
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        """
        similar_pairs = []
        num_docs = min_hash_matrix.shape[1]
        for i, j in combinations(range(num_docs), 2):
            similarity = self.get_minhash_similarity(
                min_hash_matrix[:, i], min_hash_matrix[:, j]
            )
            if similarity > self.threshold:
                similar_pairs.append((i, j))
        return similar_pairs

    def get_similar_matrix(self, min_hash_matrix: NDArray[np.int_]) -> list[tuple[float, ...]]:
        """
        Находит похожих кандидатов. Отдает матрицу расстояний
        """
        num_docs = min_hash_matrix.shape[1]
        similarity_matrix = np.zeros((num_docs, num_docs))
        for i, j in combinations(range(num_docs), 2):
            similarity = self.get_minhash_similarity(
                min_hash_matrix[:, i], min_hash_matrix[:, j]
            )
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
        similarity_list = similarity_matrix.tolist()
        assert isinstance(similarity_list, list)
        return similarity_list

    def get_minhash(self, occurrence_matrix: pd.DataFrame) -> NDArray[np.int_]:
        """
        Считает и возвращает матрицу мин хешей. MinHash содержит в себе новые индексы.

        new index = (2*(index +1) + 3) % 3

        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу
        [1, 0, 1]
        [1, 0, 1]
        [0, 1, 1]

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 0
        Doc2 : 2
        Doc3 : 0
        """
        num_rows, num_cols = occurrence_matrix.shape
        prime_num_rows = next(
            x for x in range(num_rows, 2 * num_rows) if self.is_prime(x)
        )
        minhash_matrix = np.full((self.num_permutations, num_cols), np.inf)
        for perm_index in range(self.num_permutations):
            shuffled_indices = [
                self.get_new_index(x, perm_index, prime_num_rows)
                for x in range(num_rows)
            ]
            sorted_indices = np.argsort(shuffled_indices)

            for col in range(num_cols):
                for idx in sorted_indices:
                    if occurrence_matrix.iloc[idx, col] == 1:
                        minhash_matrix[perm_index, col] = idx
                        break

        return minhash_matrix

    def run_minhash(self, corpus_of_texts: list[str]) -> list[tuple[int, int]]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs


class MinHashJaccard(MinHash):
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def get_jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """
        Вовзращает расстояние Жаккарда для двух сетов.
        """
        # TODO:
        return

    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        """
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        """
        # TODO:
        return

    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        """
        Находит похожих кандидатов. Отдает матрицу расстояний
        """
        # TODO:

        return

    def get_minhash_jaccard(self, occurrence_matrix: pd.DataFrame) -> np.array:
        """
        Считает и возвращает матрицу мин хешей. Но в качестве мин хеша выписываем минимальный исходный индекс, не новый.
        В такой ситуации можно будет пользоваться расстояние Жаккрада.

        new index = (2*(index +1) + 3) % 3

        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу
        [1, 0, 1] index: 2
        [1, 0, 1] index: 1
        [0, 1, 1] index: 0

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 2
        Doc2 : 0
        Doc3 : 2

        """
        # TODO:
        return

    def run_minhash(self, corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash_jaccard(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs
