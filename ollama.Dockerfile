# 베이스 이미지를 공식 Ollama 이미지로 지정
FROM ollama/ollama

RUN /bin/sh -c "/bin/ollama serve & sleep 5 && ollama pull tinyllama:1.1b && pkill ollama"

CMD ["serve"]