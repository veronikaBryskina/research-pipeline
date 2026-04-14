# research-pipeline

### Description

This project was developed as part of a thesis focused on disinformation detection. Goal of the thesis was to To develop and evaluate an effective disinformation detection system by fine-tuning small Large Language Models on multilingual dataset, and to design a multi-LLM classification pipeline that combines multiple models to improve detection accuracy.

But due to insuficient ammount of VRAM needed to conduct experiments,  As a result, the entire experimental workflow had to be executed on rented GPU instances. At the time, Vast.ai provided the most cost-effective and reliable environment for this work.  

This setup, however, introduced a critical limitation: deleting an instance also removed all data stored on it. Given the large number of experiments conducted during the project, the risk of losing test results, logs, and associated metadata was too high. Solution a mechanism that sends all test results together with a bunch of metadata, from Vast.ai back to user PC. 

---

### Technologies

#### LLM orchestration
- LangChain
- Ollama
- MLflow

#### Storage and data management
- PostgreSQL
- MinIO
- fsspec

#### Infrastructure and networking
- Nginx
- ngrok

---

### Pipeline results

After additional data cleaning, the results of the pipeline’s classification of content as either trustworthy or disinformation are as follows:

__For ukraininan language data:__

| Trust | Zero | One | Few | RAG |
|---|---|---|---|---|
|LLama3.1|24.6%|76.1%|11.5%|88.5%|
|Qwen3|35.4%|89.6%|67.9%|89.4%|
|Gemma3|82.8%|88.3%|83.3%|74%|

| Disinfo | Zero | One | Few | RAG |
|---|---|---|---|---|
|LLama3.1|96.6%|61%|98.6%|39.3%|
|Qwen3|96.4%|47.8%|79.1%|52.2%|
|Gemma3|67.4%|63.1%|71.3|78.7%|

__For English language data:__

| Trust | Zero | One | Few | RAG |
|---|---|---|---|---|
|LLama3.1|93.7%|96.4%|48.2%|99%|
|Qwen3|99.8%|99.9%|99.3%|97.6%|
|Gemma3|99.6%|98.2%|97.7%|91%|

| Disinfo | Zero | One | Few | RAG |
|---|---|---|---|---|
|LLama3.1|91.5%|91.1%|99.7%|58%|
|Qwen3|24.2%|63.7%|60.3%|75.4%|
|Gemma3|51.2%|60%|80.6%|84.8%|