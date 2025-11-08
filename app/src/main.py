import ollama
import langchain
import yaml
import argparse
from data_preprocessing import DataProcessor
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from prompt_templates import zero_shot_prompt, one_shot_prompt, few_shot_prompt, rag_prompt, parser
import mlflow
import requests
import json, os
from dotenv import load_dotenv
load_dotenv()
#--------------
# Temporary
#--------------



def pars_experiment():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="src/runs.yaml", help="Path to YAML config.")
    p.add_argument("--name", "-n", help="Run name.")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)["runs"] or {}

    run_name = args.name
    
    params = cfg[run_name]
    
    model = params["model"]
    datafile = params["datafile"]
    template = params["template"]
    return params, run_name

def set_template(template):
    match template:
        case "zero":
            return zero_shot_prompt()
        case "one":
            return one_shot_prompt()
        case "few":
            return few_shot_prompt()
        case "rag":
            return rag_prompt()


def compile_chain(llm, params):
    prompt = set_template(params['template'])
    if params['template'] == "rag":
        chain = {"context": DataProcessor().create_retriever(params['database']), "input": RunnablePassthrough()} | prompt | llm | parser
    else:
        chain = prompt | llm | parser
    return chain



def ensure_model_pulled(model_name: str):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    r = requests.get(f"{host}/api/tags")
    if model_name not in [m["name"] for m in r.json().get("models", [])]:
        print(f"Pulling model: {model_name}")
        r = requests.post(f"{host}/api/pull", json={"name": model_name})
        for line in r.iter_lines():
            print(line)  # Optional: stream progress
        print("Model pulled successfully")


def pull_model(model_name: str):
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    response = requests.get(f"{host}/api/tags")
    response.raise_for_status()

    available = [m.get("name") for m in response.json().get("models", [])]

    if model_name not in available:
        print(f"Model '{model_name}' not found. Pulling...")
        pull_response = requests.post(f"{host}/api/pull", json={"name": model_name})
        pull_response.raise_for_status()

        for line in pull_response.iter_lines():
            if line:
                print(line.decode("utf-8"))
        print(f"{model_name} is downloaded.")
    else:
        print(f"{model_name} already downloaded.")


def main():
    params, run_name = pars_experiment()
    mlflow.set_tracking_uri(f"{os.getenv('ENDPOINT_URL')}/mlflow/")
    with mlflow.start_run():
        try:
            mlflow.log_params(params)
            content = DataProcessor().fetch_data(params['datafile'])
            mlflow.set_tag("dataset", params['datafile'])
            
            pull_model(params['model'])
            llm = OllamaLLM(model=params['model'])

            chain = compile_chain(llm, params)
            output_file = f"../data/02_output/test_{run_name}"
            for text in content['text']:
                with open(output_file, "a") as f:
                    json.dump(chain.invoke({"input": text}), f); f.write("\n")

            mlflow.log_artifact(output_file)
        except Exception as e:
            mlflow.set_tag("error", e)
    mlflow.end_run()

if __name__ == "__main__":
    main()