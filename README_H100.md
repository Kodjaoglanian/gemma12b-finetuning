# Fine-Tuning Gemma-4-12B Juridico — Otimizado H100 (tiro e queda)

Script standalone para fine-tunar `unsloth/gemma-4-12b-it` com dados juridicos brasileiros em uma **NVIDIA H100 80GB**, projetado para **rodar de uma vez so, sem erros**, e entregar o modelo pronto na melhor qualidade — em **~1 a 3 horas**.

---

## Como rodar (1 comando)

```bash
export HF_TOKEN=hf_xxxxxxxx     # token com a licenca do Gemma 4 aceita
bash run.sh
```

O `run.sh` instala as dependencias corretas (incluindo `transformers>=5.5.0`, exigido pelo Gemma 4) e dispara o treino com os parametros recomendados.

Se as dependencias ja estiverem instaladas:

```bash
SKIP_INSTALL=1 bash run.sh
```

Para tambem exportar GGUF (uso local em llama.cpp/Ollama):

```bash
EXTRA_ARGS="--gguf" bash run.sh
```

---

## Filosofia: confiabilidade primeiro

| Decisao | Por que |
|--------|---------|
| **BF16 LoRA r=32** (padrao) | Caminho mais testado da Unsloth para Gemma 4. Cabe folgado nos 80GB, qualidade proxima de full-finetune, **sem risco de OOM**. |
| **Fallback automatico de carga** | Se a precisao escolhida falhar (ex.: FP8 indisponivel), cai para BF16 e depois 4-bit automaticamente. |
| **Colunas de dataset verificadas** | O notebook original usava campos inexistentes (`judgment`, `unanimity`). Aqui os campos reais (`judgment_label`, `decision_description`, `unanimity_text`) sao usados, com auto-deteccao de schema para os demais datasets. |
| **GGUF e merge sao nao-fatais** | Exportacoes opcionais nunca derrubam o run; o modelo treinado e sempre salvo antes. |
| **`max_seq_length=4096`** | Textos juridicos desses datasets sao curtos (medias de ~120 palavras). 4096 cobre tudo sem desperdicio, deixando o treino rapido. |
| **`group_by_length`** | Minimiza padding => maior throughput na H100. |

---

## O que mudou vs. o notebook original (T4)

| Configuracao | Notebook (T4) | Script H100 |
|---|---|---|
| Precisao | 4-bit QLoRA / fp16 | **BF16 nativo** (tensor cores Hopper) |
| LoRA rank | 16 | **32** |
| Seq length | 2048 | **4096** |
| Batch efetivo | 8 (2x4) | **8 (4x2)** com batches maiores e menos padding |
| Epocas | 1 | **2** |
| Datasets | campos com bug | **campos verificados + auto-deteccao** |
| Saida | GGUF Q4/F16 | **LoRA + merged 16-bit** (qualidade max) + GGUF opcional |

---

## Requisitos

- **GPU**: NVIDIA H100 80GB (ou H200 / A100 80GB). Funciona em GPUs menores com `--precision 4bit`.
- **CUDA**: 12.1+
- **Python**: 3.10+
- **RAM**: ~32-64 GB (para o merge 16-bit)
- **Disco**: ~60 GB livres
- **Licenca**: aceite o Gemma 4 em sua conta HuggingFace antes de rodar.

---

## Uso avancado (argumentos)

| Argumento | Padrao | Descricao |
|---|---|---|
| `--hf_token` | env `HF_TOKEN` | Token HuggingFace |
| `--model_name` | `unsloth/gemma-4-12b-it` | Modelo base |
| `--precision` | `bf16` | `bf16` \| `fp8` \| `4bit` |
| `--max_seq_length` | `4096` | Tamanho de sequencia |
| `--batch_size` | `4` | Batch por device |
| `--gradient_accumulation_steps` | `2` | Acumulo de gradiente |
| `--epochs` | `2` | Epocas |
| `--max_samples` | `0` (todos) | Limita exemplos (use p/ caber no tempo) |
| `--lora_r` / `--lora_alpha` | `32` / `32` | Capacidade LoRA |
| `--learning_rate` | `2e-4` | LR (auto -> 1e-5 se `--full_finetune`) |
| `--full_finetune` | off | Full FT (mais lento, mais VRAM) |
| `--eval_ratio` | `0.0` | Fracao p/ validacao (ex.: `0.02`) |
| `--gguf` | off | Exporta GGUF (nao-fatal) |
| `--gguf_quants` | `q4_k_m q8_0` | Quantizacoes GGUF |
| `--skip_merge` | off | Nao salvar merged 16-bit |
| `--benchmark` | off | Rodar benchmark automaticamente apos treino |
| `--benchmark_suite` | `full` | `startup` \| `pt` \| `full` |
| `--pt_harness_dir` | `./lm-evaluation-harness-pt` | Caminho do fork PT-BR |

### Exemplos

```bash
# Padrao recomendado (LoRA BF16)
python train_gemma4_juridico_h100.py

# Maxima qualidade com validacao + GGUF
python train_gemma4_juridico_h100.py --epochs 3 --eval_ratio 0.02 --gguf

# Caber em GPU menor (24-48GB)
python train_gemma4_juridico_h100.py --precision 4bit --batch_size 2 --max_seq_length 2048

# Full fine-tuning (requer H100 80GB)
python train_gemma4_juridico_h100.py --full_finetune --batch_size 2 --gradient_accumulation_steps 4

# Treino + benchmark completo (treino + avaliacao automatizada)
python train_gemma4_juridico_h100.py --benchmark --benchmark_suite full --gguf

# So benchmark em modelo ja treinado (sem treinar de novo)
python benchmark.py --model_path ./outputs_gemma4_juridico/merged_16bit --suite full
```

---

## Estrutura de saida

```
outputs_gemma4_juridico/
├── checkpoints/        # checkpoints intermediarios (save_total_limit=2)
├── modelo_final/       # adaptadores LoRA (ou pesos full)
├── merged_16bit/       # modelo mesclado 16-bit  <- MELHOR QUALIDADE p/ deploy
├── gguf/               # (se --gguf) arquivos .gguf p/ llama.cpp/Ollama
└── benchmark_results/  # (se --benchmark) resultados JSON + Markdown
    ├── benchmark_report.json
    └── benchmark_report.md
```

Carregar o modelo treinado:

```python
from unsloth import FastModel
model, tok = FastModel.from_pretrained("outputs_gemma4_juridico/modelo_final", load_in_4bit=False)
```

---

## Datasets

1. `joelniklaus/brazilian_court_decisions` — decisoes judiciais (campos verificados).
2. `celsowm/codigo_civil_brasileiro_lei_10406_2002` — Codigo Civil (auto-deteccao de schema).
3. `0rakul0/cpc_2015_brasil` — CPC/2015 (auto-deteccao de schema).

Formatados com o chat template `gemma-4` (sem modo *thinking*) e treinados **somente nas respostas** (`train_on_responses_only`).

---

## Benchmark completo estilo startup de IA

Apos o treino, rode uma **avaliacao padrao da industria** (inspirada nos relatorios de startups como Anthropic, OpenAI, Mistral, etc.):

### Benchmarks internacionais (padrao startups)

| Benchmark | O que mede |
|-----------|------------|
| **MMLU** | Conhecimento geral em 57 materias academicas |
| **MMLU-Pro** | Versao mais dificil do MMLU (conhecimento avancado) |
| **GPQA** | Raciocinio de nivel PhD (graduate-level Q&A) |
| **ARC Challenge** | Ciencia e raciocinio (ARC: AI2 Reasoning Challenge) |
| **HellaSwag** | Sentido comum e coerencia textual |
| **TruthfulQA** | Veracidade e tendencia a alucinacoes |
| **WinoGrande** | Resolucao de anaphora (pronomes) |
| **GSM8K** | Matematica e resolucao de problemas matematicos |

### Benchmarks juridicos e em portugues

| Benchmark | O que mede |
|-----------|------------|
| **OAB Exams** | Exame da Ordem dos Advogados do Brasil! (conhecimento juridico real) |
| **ENEM Challenge** | Exame Nacional do Ensino Medio (conhecimento geral BR) |
| **BLUEX** | Compreensao de leitura em portugues |
| **ASSIN2 RTE** | Reconhecimento de Entailment Textual em PT |
| **ASSIN2 STS** | Similaridade Semantica em portugues |
| **FaQuAD-NLI** | Natural Language Inference em portugues |
| **TweetSentBR** | Analise de sentimento em tweets PT-BR |
| **HateBR** | Deteccao de discurso ofensivo/hate speech em PT-BR |

### Como usar

**Automatico** (apos o treino):
```bash
python train_gemma4_juridico_h100.py --benchmark --benchmark_suite full
```

**Manual** (em modelo ja treinado):
```bash
# Todos os benchmarks
python benchmark.py --model_path ./outputs_gemma4_juridico/merged_16bit --suite full

# So benchmarks internacionais (rapido)
python benchmark.py --model_path ./outputs_gemma4_juridico/merged_16bit --suite startup

# So benchmarks juridicos/portugues
python benchmark.py --model_path ./outputs_gemma4_juridico/merged_16bit --suite pt

# Teste rapido (limitado a 100 exemplos por task)
python benchmark.py --model_path ./outputs_gemma4_juridico/merged_16bit --suite full --limit 100
```

**Saida**: `benchmark_results/benchmark_report.json` (dados) + `benchmark_report.md` (tabela Markdown para README/pitch).

---

## Solucao de problemas

- **`OutOfMemoryError`** → `--max_seq_length 2048`, `--batch_size 2`, ou `--precision 4bit`.
- **`transformers < 5.5.0`** → `pip install -U 'transformers>=5.5.0'` e reinicie.
- **Token / 401** → aceite a licenca do Gemma 4 e confira `HF_TOKEN`.
- **GGUF falhou** → normal em modelo multimodal; o `merged_16bit` e o LoRA ja estao salvos.
- **Treino muito longo** → reduza com `--max_samples 8000` ou `--epochs 1`.

---

## Referencias

- [Gemma 4 Fine-tuning Guide (Unsloth)](https://unsloth.ai/docs/models/gemma-4/train)
- [Gemma 4 Model Overview (Unsloth)](https://unsloth.ai/docs/models/gemma-4)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
