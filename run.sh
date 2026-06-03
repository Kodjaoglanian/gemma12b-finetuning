#!/usr/bin/env bash
# =============================================================================
# run.sh - Pipeline completo: setup + fine-tuning + exportacao + benchmark
#          do Gemma-4-12B Juridico em NVIDIA H100 (tiro e queda)
#
# DOCUMENTACAO COMPLETA:
#
# 1. PRE-REQUISITO OBRIGATORIO:
#    Voce DEVE aceitar a licenca do Gemma 4 na HuggingFace:
#    https://huggingface.co/google/gemma-4-12b-it
#    Depois gere um token em https://huggingface.co/settings/tokens
#
# 2. CENARIO BASICO (somente treino, 1-2h):
#    export HF_TOKEN=hf_xxx
#    bash run.sh
#
# 3. TREINO + GGUF (exporta para llama.cpp/Ollama):
#    EXTRA_ARGS="--gguf" bash run.sh
#
# 4. PIPELINE COMPLETO (treino + benchmark + GGUF) — estilo startup de IA:
#    EXTRA_ARGS="--benchmark --benchmark_suite full --gguf" bash run.sh
#    
# 5. TREINO RAPIDO (pula instalacao, ideal para testes):
#    SKIP_INSTALL=1 bash run.sh
#
# 6. TREINO LIMITADO (2000 exemplos, 1 epoca, para testar rapido):
#    EXTRA_ARGS="--max_samples 2000 --epochs 1" bash run.sh
#
# 7. FULL FINE-TUNING (mais lento, requer H100 80GB):
#    EXTRA_ARGS="--full_finetune --batch_size 2 --gradient_accumulation_steps 4" bash run.sh
#
# SAIDA PADRAO (em ./outputs_gemma4_juridico/):
#   modelo_final/       -> adaptadores LoRA (menor, reutilizavel)
#   merged_16bit/       -> modelo mesclado 16-bit (MELHOR qualidade para deploy)
#   checkpoints/        -> checkpoints intermediarios (save_total_limit=2)
#   gguf/               -> modelos quantizados (se --gguf)
#   benchmark_results/  -> relatorio JSON + Markdown (se --benchmark)
#
# DICA: O benchmark eh LENTO (pode levar 1-3h extras). Rode sem benchmark
#       primeiro para garantir que o treino funciona, depois rode com
#       --benchmark para o pipeline completo.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 1. Checagem do token ----
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERRO: defina o token antes de rodar:  export HF_TOKEN=hf_xxx"
  echo "      (e aceite a licenca do Gemma 4 em https://huggingface.co/google/gemma-4-12b-it)"
  exit 1
fi

# ---- 2. Checagem de GPU ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "AVISO: nvidia-smi nao encontrado. Este script espera uma GPU NVIDIA (H100)."
else
  echo "== GPU detectada =="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

# ---- 3. Instalacao de dependencias (Gemma 4 exige transformers >= 5.5.0) ----
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  echo "== Instalando dependencias (pode levar alguns minutos) =="
  python -m pip install --upgrade pip

  # Unsloth + zoo (sem deps p/ nao rebaixar transformers)
  pip install --upgrade --no-cache-dir unsloth unsloth_zoo

  # Stack de treino na versao exigida pelo Gemma 4
  pip install --upgrade --no-cache-dir \
    "transformers>=5.5.0" \
    "trl>=0.15.0" \
    "datasets>=2.14.0" \
    "accelerate>=0.30.0" \
    "huggingface_hub>=0.34.0" \
    "bitsandbytes>=0.43.0" \
    sentencepiece protobuf packaging

  # Benchmarking (EleutherAI lm-eval)
  pip install --upgrade --no-cache-dir "lm-eval[api]"

  # Benchmarks em portugues + juridico (OAB, ENEM, BLUEX, ASSIN2, etc.)
  PT_HARNESS_DIR="$SCRIPT_DIR/lm-evaluation-harness-pt"
  if [[ ! -d "$PT_HARNESS_DIR/.git" ]]; then
    echo "== Clonando lm-evaluation-harness-PT (benchmarks em portugues) =="
    git clone https://github.com/eduagarcia/lm-evaluation-harness-pt "$PT_HARNESS_DIR" || true
  fi
  if [[ -d "$PT_HARNESS_DIR" && -f "$PT_HARNESS_DIR/setup.py" ]]; then
    (cd "$PT_HARNESS_DIR" && pip install -e . --no-cache-dir) || echo "AVISO: instalacao PT-harness falhou"
  fi

  echo "== Dependencias instaladas =="
else
  echo "== SKIP_INSTALL=1: pulando instalacao =="
fi

# ---- 4. Treino ----
echo "== Iniciando treinamento =="
python train_gemma4_juridico_h100.py \
  --hf_token "$HF_TOKEN" \
  --precision bf16 \
  --max_seq_length 4096 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --epochs 2 \
  --lora_r 32 \
  --lora_alpha 32 \
  ${EXTRA_ARGS:-}

echo "== Finalizado. Veja a pasta outputs_gemma4_juridico/ =="
