from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        """
        Initialize the BGE reranker.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def rerank(self, query, candidates):
        """
        Rerank the candidates based on the similarity with the query using the BGE model.
        """
        inputs = self.tokenizer([query] * len(candidates), candidates, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        return [candidate for _, candidate in sorted(zip(logits.squeeze().tolist(), candidates), key=lambda x: x[0], reverse=True)]
