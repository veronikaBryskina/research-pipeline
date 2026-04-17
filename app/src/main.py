import yaml
import argparse
from data_preprocessing import DataProcessor
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import OutputFixingParser
from prompt_templates import zero_shot_prompt, one_shot_prompt, few_shot_prompt, rag_prompt, parser
from data_preprocessing import pull_model
import mlflow
import json
from dotenv import load_dotenv
load_dotenv()

def pars_experiment():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="src/runs.yaml", help="Path to YAML config.")
    p.add_argument("--name", "-n", help="Run name.")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)["runs"] or {}

    run_name = args.name
    params = cfg[run_name]
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
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm) # in case parser is not enough
    prompt = set_template(params['template'])
    if params['template'] == "rag":
        retriever = DataProcessor().create_retriever(params['context'])
        chain = {"context": retriever, "input": RunnablePassthrough()} | prompt | llm | parser
    else:
        chain = prompt | llm | parser
    return chain

def run_llm(params, output_file):
    content = DataProcessor().fetch_data(params['datafile'])

    pull_model(params['model'])
    llm = OllamaLLM(
        model=params["model"],
        temperature=0,
        num_ctx=2048,
        num_predict=128,
        format="json",
        model_kwargs={"num_batch": 256,"num_gpu": 1,},
        )
    
    chain = compile_chain(llm, params)

    for text in content["text"]:
        result = chain.invoke({"input": text})
        with open(output_file, "a") as f:
            json.dump(result, f)
            f.write("\n")


def main():
    params, run_name = pars_experiment()
    output_file = f"data/02_output/test_{run_name}.jsonl"
    
    with mlflow.start_run():
        try:
            mlflow.log_params(params)
            mlflow.set_tag("dataset", params['datafile'])
            run_llm(params, output_file)
            mlflow.log_artifact(output_file)
        except Exception as e:
            mlflow.set_tag("error", e)
    mlflow.end_run()

if __name__ == "__main__":
    main()