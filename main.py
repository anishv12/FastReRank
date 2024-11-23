import logging
from reranker import HybridReranker  # Importing the HybridReranker class

def load_config():
    """
    Example function for loading configuration. You could use a config file or environment variables here.
    """
    return {
        "bge_model_name": "BAAI/bge-reranker-v2-m3",
        "cross_encoder_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_n": 3
    }

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Load configuration (e.g., from a file or environment variables)
    config = load_config()

    # Initialize HybridReranker with the configuration
    reranker = HybridReranker(
        bge_model_name=config["bge_model_name"],
        cross_encoder_model_name=config["cross_encoder_model_name"],
        top_n=config["top_n"]
    )

    
    query = "What area did NVIDIA initially focus on before expanding to other computationally intensive fields?"
    candidates = [
                    {
                     'text' : "Since our original focus on PC graphics, we have expanded to several other large and important computationally intensive fields.",
                     'dist' : "some distance"
                    } ,
                    {
                     'text' : "NVIDIA is one of the highest valued company in the world , recently taking a lead over Intel",
                     'dist' : "some distance"
                    },
                    {
                     'text' : "NVIDIA recorded an acquisition termination cost of $1.35 billion in fiscal year 2023.",
                     'dist' : "some distance"
                    },
                    {
                    'text' : "Some of the most recent applications of GPU-powered deep learning include recommendation systems, which are AI algorithms trained to understand the preferences, previous decisions, and characteristics of people and products using data gathered about their interactions, large language models, which can recognize, summarize, translate, predict and generate text and other content based on knowledge gained from massive datasets, and generative AI, which uses algorithms that create new content, including audio, code, images, text, simulations, and videos, based on the data they have been trained on.",
                    'dist' : "some distance"
                    }   
                 ]
    candidates_list =[]
    for i in candidates:
        candidates_list.append(i['text']) 
    # Perform reranking
    reranked_candidates = reranker.rerank(query, candidates_list)

    # Output the reranked candidates
    logging.info("Hybrid Reranked Candidates:")
    for i, candidate in enumerate(reranked_candidates, start=1):
        logging.info(f"{i}. {candidate}")
