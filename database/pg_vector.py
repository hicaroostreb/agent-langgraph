import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Carregar variáveis de ambiente
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")


class SupabaseVectorDB:
    def __init__(self):
        self._connection = None
        self._connect()

    def _connect(self):
        """Estabelece conexão com o banco."""
        if not self._connection or self._connection.closed != 0:
            try:
                self._connection = psycopg2.connect(
                    host=DB_HOST,
                    database=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    port=DB_PORT,
                )
                self._connection.autocommit = True
            except psycopg2.Error as e:
                raise ConnectionError(f"Erro ao conectar ao banco: {e}")

    def close(self):
        """Fecha a conexão com o banco."""
        if self._connection and not self._connection.closed:
            self._connection.close()

    def search_similar_faqs(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        similarity_threshold: float = 0.4,
    ) -> List[dict]:
        """Busca as FAQs mais semelhantes usando pgvector."""
        self._connect()

        query = """
            SELECT
                id,
                pergunta,
                resposta,
                1 - (embedding <=> %s::vector) AS similaridade
            FROM faq_embeddings
            ORDER BY similaridade DESC
            LIMIT %s;
        """
        try:
            with self._connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (query_embedding, top_k * 2))
                results = cursor.fetchall()
        except Exception as e:
            print(f"Erro ao executar a consulta: {e}")
            return []

        return [r for r in results if r["similaridade"] >= similarity_threshold][:top_k]

    def __del__(self):
        self.close()
