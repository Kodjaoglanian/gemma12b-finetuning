#!/usr/bin/env bash
# =============================================================================
# run.sh - Setup + treino "tiro e queda" do Gemma-4-12B Juridico em H100
#
# Uso:
#   export HF_TOKEN=hf_xxxxxxxx        # token com aceite da licenca Gemma
#   bash run.sh                        # instala deps (1a vez) e treina
#
# Variaveis opcionais:
#   SKIP_INSTALL=1 bash run.sh         # pula a instalacao (deps ja prontas)
#   EXTRA_ARGS="--gguf --epochs 3" bash run.sh
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
