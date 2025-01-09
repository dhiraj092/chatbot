from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

def main():
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("design")
    print(f"Vector for 'design': {vector}")
    print(f"Vector length: {len(vector)}")

    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("ideate", "Crazy_Eights")
    result = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {result}")

if __name__ == "__main__":
    print("Starting main...")
    main()
    print("Script completed!")