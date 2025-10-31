import ollama
import langchain
import yaml
import argparse
from data_preprocessing import DataProcessor
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from prompt_templates import zero_shot_prompt, parser
#--------------
# Temporary
#--------------



def pars_experiment():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="/home/niksan/proj/hw/research-pipeline/app/data/01_input/runs.yaml", help="Path to YAML config.")
    p.add_argument("--name", "-n", help="Run name.")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)["runs"] or {}

    run_name = args.name
    
    params = cfg[run_name]
    
    model = params["model"]
    datafile = params["datafile"]
    template = params["template"]
    return params


def main():
    params = pars_experiment()
    print(params['model'])
    content = DataProcessor().fetch_data(params['datafile'])

    prompt = zero_shot_prompt()

    llm = OllamaLLM(model=params['model'])

    chain = prompt | llm | parser

    for text in content['text']:
        chain.invoke({"input": text})

if __name__ == "__main__":
    main()