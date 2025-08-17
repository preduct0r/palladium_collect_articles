import pandas as pd
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict
import time


class SimilarityFinder:
    def __init__(self, excel_file_path: str, cache_dir: str = "./cache"):
        """
        Инициализация поисковика похожих пар
        
        Args:
            excel_file_path: Путь к Excel файлу с данными
            cache_dir: Директория для кэширования эмбеддингов
        """
        self.excel_file_path = excel_file_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Инициализация embedder как в исходном файле
        self.embedder = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B", 
            base_url="http://localhost:8090/v1",
            api_key="EMPTY"
        )
        
        # Пути для кэширования
        self.embeddings_cache_path = os.path.join(cache_dir, "embeddings.pkl")
        self.questions_cache_path = os.path.join(cache_dir, "questions.pkl")
        
    def load_data(self) -> pd.DataFrame:
        """Загрузка данных из Excel файла"""
        print("Загружаю данные из Excel файла...")
        df = pd.read_excel(self.excel_file_path)
        print(f"Загружено {len(df)} вопросов")
        return df
    
    def get_or_compute_embeddings(self, questions: List[str], force_recompute: bool = False) -> np.ndarray:
        """
        Получение или вычисление эмбеддингов с кэшированием
        
        Args:
            questions: Список вопросов для векторизации
            force_recompute: Принудительное перевычисление эмбеддингов
            
        Returns:
            Матрица эмбеддингов размера (n_questions, embedding_dim)
        """
        # Проверяем кэш
        if not force_recompute and os.path.exists(self.embeddings_cache_path) and os.path.exists(self.questions_cache_path):
            print("Загружаю эмбеддинги из кэша...")
            with open(self.embeddings_cache_path, 'rb') as f:
                cached_embeddings = pickle.load(f)
            with open(self.questions_cache_path, 'rb') as f:
                cached_questions = pickle.load(f)
            
            # Проверяем, что кэшированные вопросы совпадают с текущими
            if cached_questions == questions:
                print("Эмбеддинги загружены из кэша")
                return cached_embeddings
            else:
                print("Вопросы изменились, перевычисляю эмбеддинги...")
        
        # Вычисляем эмбеддинги
        print("Вычисляю эмбеддинги для вопросов...")
        
        # Обрабатываем батчами для эффективности
        batch_size = 50  # Оптимальный размер батча для большинства моделей
        embeddings = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc="Обработка батчей"):
            batch = questions[i:i + batch_size]
            batch_embeddings = self.embedder.embed_documents(batch)
            embeddings.extend(batch_embeddings)
            
            # Небольшая пауза между батчами чтобы не перегружать API
            time.sleep(0.1)
        
        embeddings_matrix = np.array(embeddings)
        
        # Сохраняем в кэш
        print("Сохраняю эмбеддинги в кэш...")
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(embeddings_matrix, f)
        with open(self.questions_cache_path, 'wb') as f:
            pickle.dump(questions, f)
        
        return embeddings_matrix
    
    def find_most_similar_pairs(self, embeddings: np.ndarray, questions: List[str], 
                              top_k: int = 10, min_similarity: float = 0.0) -> List[Tuple[int, int, float, str, str]]:
        """
        Поиск наиболее похожих пар с оптимизацией
        
        Args:
            embeddings: Матрица эмбеддингов
            questions: Список вопросов
            top_k: Количество топ пар для возврата
            min_similarity: Минимальный порог сходства
            
        Returns:
            Список кортежей (индекс1, индекс2, сходство, вопрос1, вопрос2)
        """
        print("Вычисляю матрицу косинусного сходства...")
        
        # Нормализуем эмбеддинги для эффективного вычисления косинусного сходства
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Вычисляем матрицу сходства
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        print("Ищу наиболее похожие пары...")
        
        # Создаем маску для верхнего треугольника (исключаем диагональ и дубликаты)
        n = len(questions)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        
        # Получаем индексы и значения только для верхнего треугольника
        indices = np.where(mask)
        similarities = similarity_matrix[indices]
        
        # Фильтруем по минимальному порогу
        valid_indices = similarities >= min_similarity
        filtered_similarities = similarities[valid_indices]
        filtered_i = indices[0][valid_indices]
        filtered_j = indices[1][valid_indices]
        
        # Сортируем по убыванию сходства
        sorted_indices = np.argsort(filtered_similarities)[::-1]
        
        # Формируем результат
        results = []
        for idx in sorted_indices[:top_k]:
            i, j = filtered_i[idx], filtered_j[idx]
            similarity = filtered_similarities[idx]
            results.append((i, j, similarity, questions[i], questions[j]))
        
        return results
    
    def find_duplicates(self, embeddings: np.ndarray, questions: List[str], 
                       similarity_threshold: float = 0.95) -> List[Tuple[int, int, float, str, str]]:
        """
        Поиск потенциальных дубликатов (очень похожих вопросов)
        
        Args:
            embeddings: Матрица эмбеддингов
            questions: Список вопросов
            similarity_threshold: Порог сходства для считания дубликатами
            
        Returns:
            Список потенциальных дубликатов
        """
        print(f"Ищу потенциальные дубликаты (сходство >= {similarity_threshold})...")
        
        # Используем тот же алгоритм, но с высоким порогом
        similar_pairs = self.find_most_similar_pairs(
            embeddings, questions, 
            top_k=1000,  # Увеличиваем лимит для поиска всех дубликатов
            min_similarity=similarity_threshold
        )
        
        return similar_pairs
    
    def save_results(self, results: List[Tuple[int, int, float, str, str]], 
                    output_file: str, title: str = "Результаты поиска"):
        """
        Сохранение результатов в Excel файл
        
        Args:
            results: Список результатов
            output_file: Путь к выходному файлу
            title: Заголовок для данных
        """
        print(f"Сохраняю результаты в {output_file}...")
        
        df_results = pd.DataFrame(results, columns=[
            'Индекс_1', 'Индекс_2', 'Сходство', 'Вопрос_1', 'Вопрос_2'
        ])
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name=title, index=False)
        
        print(f"Результаты сохранены в {output_file}")
    
    def run_analysis(self, top_k: int = 20, similarity_threshold: float = 0.95, 
                    force_recompute: bool = False):
        """
        Запуск полного анализа сходства
        
        Args:
            top_k: Количество топ похожих пар
            similarity_threshold: Порог для поиска дубликатов
            force_recompute: Принудительное перевычисление эмбеддингов
        """
        # Загрузка данных
        df = self.load_data()
        questions = df['Вопрос'].astype(str).tolist()
        
        # Получение эмбеддингов
        embeddings = self.get_or_compute_embeddings(questions, force_recompute)
        
        print(f"Размер матрицы эмбеддингов: {embeddings.shape}")
        
        # Поиск топ похожих пар
        print(f"\n=== Поиск топ-{top_k} наиболее похожих пар ===")
        top_pairs = self.find_most_similar_pairs(embeddings, questions, top_k)
        
        print(f"\nТоп-{len(top_pairs)} наиболее похожих пар:")
        for i, (idx1, idx2, sim, q1, q2) in enumerate(top_pairs, 1):
            print(f"{i}. Сходство: {sim:.4f}")
            print(f"   Вопрос {idx1}: {q1[:100]}...")
            print(f"   Вопрос {idx2}: {q2[:100]}...")
            print()
        
        # Поиск дубликатов
        print(f"\n=== Поиск потенциальных дубликатов (сходство >= {similarity_threshold}) ===")
        duplicates = self.find_duplicates(embeddings, questions, similarity_threshold)
        
        if duplicates:
            print(f"\nНайдено {len(duplicates)} потенциальных дубликатов:")
            for i, (idx1, idx2, sim, q1, q2) in enumerate(duplicates[:10], 1):  # Показываем первые 10
                print(f"{i}. Сходство: {sim:.4f}")
                print(f"   Вопрос {idx1}: {q1[:100]}...")
                print(f"   Вопрос {idx2}: {q2[:100]}...")
                print()
        else:
            print("Потенциальных дубликатов не найдено")
        
        # Сохранение результатов
        self.save_results(top_pairs, "top_similar_pairs.xlsx", "Топ похожих пар")
        if duplicates:
            self.save_results(duplicates, "potential_duplicates.xlsx", "Потенциальные дубликаты")
        
        # Статистика
        print(f"\n=== Статистика ===")
        print(f"Всего вопросов: {len(questions)}")
        print(f"Размерность эмбеддингов: {embeddings.shape[1]}")
        print(f"Топ похожих пар: {len(top_pairs)}")
        print(f"Потенциальных дубликатов: {len(duplicates)}")
        
        return top_pairs, duplicates


def main():
    parser = argparse.ArgumentParser(description="Поиск похожих пар вопросов по косинусной близости")
    parser.add_argument("--excel_file", default="BENCH2.xlsx", help="Путь к Excel файлу с вопросами")
    parser.add_argument("--top_k", type=int, default=20, help="Количество топ похожих пар")
    parser.add_argument("--similarity_threshold", type=float, default=0.95, help="Порог сходства для дубликатов")
    parser.add_argument("--force_recompute", action="store_true", help="Принудительное перевычисление эмбеддингов")
    parser.add_argument("--cache_dir", default="./cache", help="Директория для кэширования")
    
    args = parser.parse_args()
    
    # Создаем и запускаем анализатор
    finder = SimilarityFinder(args.excel_file, args.cache_dir)
    top_pairs, duplicates = finder.run_analysis(
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        force_recompute=args.force_recompute
    )


if __name__ == "__main__":
    main() 