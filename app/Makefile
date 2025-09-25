all: install serve pull

install:
	curl -fsSL https://ollama.com/install.sh | sh

serve:
	nohup ollama serve > ollama.log 2>&1 &

pull:
	ollama pull <llm1>
	ollama pull <llm2>


