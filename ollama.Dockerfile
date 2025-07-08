# 베이스 이미지를 공식 Ollama 이미지로 지정
FROM ollama/ollama

RUN ollama serve & \
    sleep 5 && \
    ollama pull bge-m3:567m && \
    pkill ollama
    
CMD ["ollama", "serve"]