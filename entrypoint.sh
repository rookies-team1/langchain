#!/bin/bash

# 에러 발생 시 스크립트 중단
set -e

# Ollama 서버를 백그라운드에서 실행
ollama serve &

# 백그라운드에서 실행된 프로세스의 PID 저장
pid=$!

echo "Ollama 서버 시작 중... (PID: $pid)"

# 서버가 응답할 때까지 잠시 대기 (필요 시 sleep 시간 조절)
sleep 5

echo "bge-m3:567m 모델 다운로드 시도..."

# 필요한 모델을 미리 다운로드
ollama pull bge-m3:567m

echo "모델 다운로드 완료."

# 백그라운드에서 실행 중인 Ollama 서버 프로세스가 종료되기를 기다림
# 이 명령 덕분에 컨테이너가 바로 종료되지 않고 계속 실행 상태를 유지함
wait $pid#!/bin/bash

# 에러 발생 시 스크립트 중단
set -e

# Ollama 서버를 백그라운드에서 실행
ollama serve &

# 백그라운드에서 실행된 프로세스의 PID 저장
pid=$!

echo "Ollama 서버 시작 중... (PID: $pid)"

# 서버가 응답할 때까지 잠시 대기 (필요 시 sleep 시간 조절)
sleep 5

echo "bge-m3:567m 모델 다운로드 시도..."

# 필요한 모델을 미리 다운로드
ollama pull bge-m3:567m

echo "모델 다운로드 완료."

# 백그라운드에서 실행 중인 Ollama 서버 프로세스가 종료되기를 기다림
# 이 명령 덕분에 컨테이너가 바로 종료되지 않고 계속 실행 상태를 유지함
wait $pid