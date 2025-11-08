all: install serve instalation

install:
	curl -fsSL https://ollama.com/install.sh | sh

serve:
	nohup ollama serve > ollama.log 2>&1 &

pull:
	ollama pull llama3.1:8b
	ollama pull qwen3:8b
	ollama pull gemma3:12b
	ollama pull mxbai-embed-large

instalation:
	apt install python3.10-venv
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirenments.txt

runs: run1 run5 run9 run13 run17 run21 run25 run29 run33

run%:
	python3 src/main.py --name experiment_$*

