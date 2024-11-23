import logging
from bge_reranker import BGEReranker  # Importing the BGE reranker class
from sentence_transformers import CrossEncoder

class HybridReranker:
    def __init__(self, bge_model_name="BAAI/bge-reranker-v2-m3", cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3):
        """
        Initialize the hybrid reranker with both BGE and CrossEncoder models.
        """
        self.bge_reranker = self._load_bge_model(bge_model_name)
        self.cross_encoder = self._load_cross_encoder(cross_encoder_model_name)
        self.top_n = top_n
        logging.info(f"HybridReranker initialized with BGE model: {bge_model_name} and CrossEncoder model: {cross_encoder_model_name}")

    def _load_bge_model(self, model_name):
        """
        Helper function to load the BGE model with error handling and logging.
        """
        try:
            bge_model = BGEReranker(model_name=model_name)
            logging.info(f"BGE model loaded successfully: {model_name}")
            return bge_model
        except Exception as e:
            logging.error(f"Error loading BGE model: {e}")
            raise

    def _load_cross_encoder(self, model_name):
        """
        Helper function to load the CrossEncoder model with error handling and logging.
        """
        try:
            cross_encoder_model = CrossEncoder(model_name)
            logging.info(f"CrossEncoder model loaded successfully: {model_name}")
            return cross_encoder_model
        except Exception as e:
            logging.error(f"Error loading CrossEncoder model: {e}")
            raise

    def rerank(self, query, candidates):
        """
        Hybrid reranking using BGE for initial filtering and CrossEncoder for final reranking.
        """
        try:
            filtered_candidates = self.bge_reranker.rerank(query, candidates)[:self.top_n]
            pairs = [[query, candidate] for candidate in filtered_candidates]
            scores = self.cross_encoder.predict(pairs)

            sorted_candidates = [
                candidate for _, candidate in sorted(zip(scores, filtered_candidates), key=lambda x: x[0], reverse=True)
            ]
            return sorted_candidates
        except Exception as e:
            logging.error(f"Error during reranking: {e}")
            raise
