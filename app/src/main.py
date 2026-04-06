import yaml
import argparse
from data_preprocessing import DataProcessor
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import OutputFixingParser
from prompt_templates import zero_shot_prompt, one_shot_prompt, few_shot_prompt, rag_prompt, parser
from data_preprocessing import pull_model
import mlflow
import requests
import json, os
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def compile_chain(llm, params, retriever=None):
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm) # in case parser is not enough
    prompt = set_template(params['template'])
    if params['template'] == "rag":
        chain = {"context": retriever, "input": RunnablePassthrough()} | prompt | llm | parser
    else:
        chain = prompt | llm | parser
    return chain

def unload_model(model_name):
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    requests.post(
        f"{host}/api/generate",
        json={"model": model_name, "prompt": "", "keep_alive": 0},
    )

def main():
    params, run_name = pars_experiment()
    with mlflow.start_run():
        #try:
        mlflow.log_params(params)
        content = DataProcessor().fetch_data(params['datafile'])
        mlflow.set_tag("dataset", params['datafile'])
        
        pull_model(params['model'])

        retriever = DataProcessor().create_retriever('rag_texts_1.csv') ############################

        llm = OllamaLLM(
            model=params["model"],
            temperature=0,
            num_ctx=2048,
            num_predict=128,
            keep_alive=-1,
            format="json",
            model_kwargs={
                "num_batch": 256,
                "num_gpu": 1,
            },
        )


        chain = compile_chain(llm, params, retriever) #####################################
        output_file = f"data/02_output/test_{run_name}.jsonl"
        context_file = f"data/02_output/test_{run_name}_retrieved.jsonl"


        texts = content["text"]
        def run_one(t):
            try:
                context = retriever.invoke({"input": t}) ##########
                with open(context_file, "a") as f: ###################
                    json.dump(str(context), f) ###############################
                    f.write("\n") ################"""
                return chain.invoke({"input": t})
            except Exception as e:
                return {"error": str(e), "raw_output": getattr(e, "llm_output", None)}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_one, t) for t in texts]

            for fut in as_completed(futures):
                result = fut.result()
                with open(output_file, "a") as f:
                    json.dump(result, f)
                    f.write("\n")

        mlflow.log_artifact(output_file)
        #except Exception as e:
            #mlflow.set_tag("error", e)
    mlflow.end_run()
    unload_model(params["model"])

if __name__ == "__main__":
    main()