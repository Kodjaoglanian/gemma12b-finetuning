#!/usr/bin/env python3
"""
Fine-tuning de unsloth/gemma-4-12b-it (Juridico Brasileiro) - Otimizado NVIDIA H100.

OBJETIVO: rodar de uma unica vez ("tiro e queda"), sem erros, entregando o modelo
pronto na melhor qualidade possivel, treinando em ~1-3 horas em uma H100 80GB.

Estrategia de confiabilidade:
  - Precisao BF16 nativa (H100) por padrao => caminho mais testado e estavel.
  - LoRA de alta capacidade (r=32) => qualidade proxima de full-finetune, sem OOM.
  - Mapeamento de datasets ROBUSTO (colunas verificadas + auto-deteccao) => nao quebra
    se o schema for diferente do esperado.
  - Cada etapa "perigosa" (GGUF, merge) e nao-fatal: o treino e o modelo final sao
    sempre salvos mesmo que a exportacao opcional falhe.

Datasets juridicos brasileiros:
  - joelniklaus/brazilian_court_decisions  (decisoes judiciais)
  - celsowm/codigo_civil_brasileiro_lei_10406_2002  (Codigo Civil)
  - 0rakul0/cpc_2015_brasil  (Codigo de Processo Civil 2015)

Uso minimo:
  export HF_TOKEN=hf_xxx
  python train_gemma4_juridico_h100.py

Uso explicito:
  python train_gemma4_juridico_h100.py --hf_token hf_xxx --epochs 2 --gguf
"""

import os
import sys
import gc
import time
import argparse
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 0. VARIAVEIS DE AMBIENTE (antes de importar torch / unsloth)
# =============================================================================
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Reduz fragmentacao de memoria CUDA na H100
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# Acelera matmul em Hopper/Ampere (TF32) sem perda relevante de qualidade
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Evita recompilacoes excessivas do dynamo
try:
    torch._dynamo.config.recompile_limit = 128
except Exception:
    pass


# =============================================================================
# Utilidades
# =============================================================================
def log_banner(msg: str):
    print("\n" + "=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def get_gpu_info():
    if not torch.cuda.is_available():
        return None, 0.0, 0.0
    props = torch.cuda.get_device_properties(0)
    total = round(props.total_memory / 1024**3, 2)
    reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    return props.name, total, reserved


def print_gpu_stats(label: str = ""):
    name, total, reserved = get_gpu_info()
    if name:
        print(f"[VRAM] {label:<14} {name} | Total: {total} GB | Reservada(pico): {reserved} GB")
    else:
        print("[VRAM] GPU CUDA nao detectada!")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def first_present(ex, keys):
    """Retorna o primeiro campo nao-vazio dentre `keys` (string)."""
    for k in keys:
        if k in ex and ex[k] is not None:
            v = str(ex[k]).strip()
            if v and v.lower() not in ("none", "nan"):
                return v
    return ""


# =============================================================================
# 1. ARGUMENTOS CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tuning Gemma-4-12B Juridico - Otimizado H100 (tiro e queda)"
    )
    p.add_argument("--hf_token", type=str, default=None,
                   help="Token HuggingFace (read). Se omitido, usa a env HF_TOKEN.")
    p.add_argument("--model_name", type=str, default="unsloth/gemma-4-12b-it",
                   help="Modelo base (padrao: unsloth/gemma-4-12b-it)")
    p.add_argument("--output_dir", type=str, default="./outputs_gemma4_juridico",
                   help="Diretorio de saida (checkpoints + modelo final)")

    # Capacidade / performance
    p.add_argument("--precision", type=str, default="bf16",
                   choices=["bf16", "fp8", "4bit"],
                   help="bf16 (padrao, mais estavel/qualidade) | fp8 (experimental) | 4bit (menos VRAM)")
    p.add_argument("--max_seq_length", type=int, default=4096,
                   help="Comprimento maximo de sequencia (textos juridicos cabem em 2k-4k)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch por device (H100 BF16 LoRA: 4-8)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2,
                   help="Acumulo de gradiente (effective batch = batch * accum)")
    p.add_argument("--epochs", type=float, default=2.0,
                   help="Numero de epocas (2-3 recomendado)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Limita o nº de exemplos de treino (0 = todos). Use p/ caber no tempo.")

    # LoRA / Full FT
    p.add_argument("--full_finetune", action="store_true",
                   help="Full fine-tuning (mais lento, ~OOM-risk). Padrao: LoRA.")
    p.add_argument("--lora_r", type=int, default=32, help="Rank LoRA")
    p.add_argument("--lora_alpha", type=int, default=32, help="Alpha LoRA")
    p.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout LoRA")

    # Otimizacao
    p.add_argument("--learning_rate", type=float, default=2e-4,
                   help="LR (LoRA: 2e-4; Full FT: ~1e-5)")
    p.add_argument("--optim", type=str, default="adamw_8bit",
                   choices=["adamw_8bit", "paged_adamw_8bit", "adamw_torch", "adamw_torch_fused"],
                   help="Otimizador")
    p.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Clip de gradiente")
    p.add_argument("--lr_scheduler", type=str, default="cosine", help="Scheduler de LR")

    # Avaliacao / logging / checkpoint
    p.add_argument("--eval_ratio", type=float, default=0.0,
                   help="Fracao p/ validacao (0 = sem eval). Ex: 0.02")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint a cada N steps")
    p.add_argument("--logging_steps", type=int, default=5, help="Log a cada N steps")
    p.add_argument("--seed", type=int, default=3407, help="Seed")

    # Exportacao
    p.add_argument("--skip_merge", action="store_true",
                   help="Nao salvar o modelo mesclado 16-bit (so adaptadores LoRA)")
    p.add_argument("--gguf", action="store_true",
                   help="Tambem exportar GGUF (pode falhar em modelo multimodal; nao-fatal)")
    p.add_argument("--gguf_quants", type=str, nargs="+", default=["q4_k_m", "q8_0"],
                   help="Metodos de quantizacao GGUF")

    # Benchmark
    p.add_argument("--benchmark", action="store_true",
                   help="Rodar benchmark automaticamente apos o treino")
    p.add_argument("--benchmark_suite", type=str, default="full",
                   choices=["startup", "pt", "full"],
                   help="Suite de benchmark a rodar")
    p.add_argument("--pt_harness_dir", type=str, default="./lm-evaluation-harness-pt",
                   help="Caminho para o lm-evaluation-harness-pt (benchmarks PT-BR)")

    p.add_argument("--test_prompt", type=str,
                   default="O que e o principio da boa-fe objetiva no Codigo Civil brasileiro?",
                   help="Prompt de teste pos-treino")
    return p.parse_args()


# =============================================================================
# 2. VERIFICACAO DE DEPENDENCIAS
# =============================================================================
def check_dependencies():
    log_banner("VERIFICANDO DEPENDENCIAS")
    ok = True

    try:
        import transformers
        from packaging import version
        print(f"  transformers : {transformers.__version__}")
        if version.parse(transformers.__version__) < version.parse("5.5.0"):
            print("  ERRO: Gemma 4 exige transformers >= 5.5.0.")
            print("        pip install -U 'transformers>=5.5.0'")
            ok = False
    except ImportError as e:
        print(f"  ERRO: transformers ausente ({e})")
        ok = False

    for mod, pretty in [("unsloth", "unsloth"), ("trl", "trl"), ("datasets", "datasets")]:
        try:
            __import__(mod)
            print(f"  {pretty:<13}: OK")
        except ImportError:
            print(f"  ERRO: {pretty} nao instalado.")
            ok = False

    try:
        import bitsandbytes  # noqa: F401
        print("  bitsandbytes : OK")
    except ImportError:
        print("  AVISO: bitsandbytes ausente (necessario p/ otimizadores 8-bit).")

    if not torch.cuda.is_available():
        print("  ERRO: CUDA indisponivel. Este script requer GPU NVIDIA.")
        ok = False
    else:
        name, total, _ = get_gpu_info()
        print(f"  GPU          : {name} ({total} GB)")
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute cap. : {cap[0]}.{cap[1]}")

    if not ok:
        print("\n  Dependencias incompletas. Veja requirements.txt / run.sh.")
        sys.exit(1)
    print("  Dependencias OK.")


# =============================================================================
# 3. CARREGAMENTO DO MODELO (BF16 padrao, com fallbacks automaticos)
# =============================================================================
def load_model_and_tokenizer(args):
    log_banner("CARREGANDO MODELO")
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    print_gpu_stats("ANTES LOAD")
    print(f"  Modelo: {args.model_name} | precisao={args.precision} | seq={args.max_seq_length}")
    print(f"  full_finetuning={args.full_finetune}")

    common = dict(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        token=args.hf_token,
    )

    # Ordem de tentativas conforme a precisao escolhida (com fallback seguro)
    if args.precision == "4bit":
        attempts = [dict(load_in_4bit=True, dtype=None, full_finetuning=False)]
    elif args.precision == "fp8":
        attempts = [
            dict(load_in_fp8=True, dtype=None, full_finetuning=args.full_finetune),
            dict(dtype=torch.bfloat16, full_finetuning=args.full_finetune),
            dict(load_in_4bit=True, dtype=None, full_finetuning=False),
        ]
    else:  # bf16 (padrao)
        attempts = [
            dict(dtype=torch.bfloat16, full_finetuning=args.full_finetune),
            dict(load_in_4bit=True, dtype=None, full_finetuning=False),
        ]

    model = tokenizer = None
    last_err = None
    start = time.time()
    for i, extra in enumerate(attempts, 1):
        try:
            print(f"  Tentativa {i}/{len(attempts)} -> {extra}")
            model, tokenizer = FastModel.from_pretrained(**{**common, **extra})
            # Se caimos em 4bit, full finetuning nao e possivel
            if extra.get("load_in_4bit"):
                args.full_finetune = False
                args.precision = "4bit"
            break
        except Exception as e:
            last_err = e
            print(f"    Falhou: {type(e).__name__}: {e}")
            cleanup()

    if model is None:
        raise RuntimeError(f"Nao foi possivel carregar o modelo. Ultimo erro: {last_err}")

    print(f"  Modelo carregado em {time.time() - start:.1f}s")
    print_gpu_stats("DEPOIS LOAD")

    # Chat template Gemma-4 (sem thinking => respostas juridicas diretas)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    print(f"  Chat template 'gemma-4' aplicado. EOS={tokenizer.eos_token!r}")
    return model, tokenizer


# =============================================================================
# 4. CONFIGURACAO LoRA
# =============================================================================
def setup_peft(model, args):
    if args.full_finetune:
        log_banner("FULL FINE-TUNING ATIVADO")
        treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Parametros treinaveis: {treinaveis:,} / {total:,} ({100*treinaveis/max(total,1):.2f}%)")
        return model

    log_banner("CONFIGURANDO LoRA")
    from unsloth import FastModel

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,       # texto puro
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",  # economiza VRAM e trata Gemma-4 corretamente
    )
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"  Parametros treinaveis: {treinaveis:,} / {total:,} ({100*treinaveis/max(total,1):.4f}%)")
    print_gpu_stats("DEPOIS LoRA")
    return model


# =============================================================================
# 5. PIPELINE DE DADOS (robusto)
# =============================================================================
def prepare_dataset(tokenizer, args):
    log_banner("PREPARANDO DATASETS JURIDICOS")
    from datasets import load_dataset, concatenate_datasets

    def make_convo(system_txt, user_txt, asst_txt):
        messages = [
            {"role": "system", "content": system_txt},
            {"role": "user", "content": user_txt},
            {"role": "assistant", "content": asst_txt},
        ]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if templated.startswith("<bos>"):
            templated = templated[len("<bos>"):]
        return templated

    datasets_list = []

    # ---- Dataset 1: brazilian_court_decisions (colunas VERIFICADAS) ----
    print("\n[1/3] joelniklaus/brazilian_court_decisions ...")
    try:
        ds_raw = load_dataset("joelniklaus/brazilian_court_decisions",
                              split="train", token=args.hf_token)
        print("      Colunas:", ds_raw.column_names, "| Exemplos:", len(ds_raw))
        label_map = {
            "yes": "procedente (recurso/pedido favoravel)",
            "no": "improcedente (negado)",
            "partial": "parcialmente procedente",
        }

        def map_court(ex):
            texto = first_present(ex, ["decision_description", "ementa_text", "judgment_text"])
            label = first_present(ex, ["judgment_label"])
            unanim = first_present(ex, ["unanimity_text", "unanimity_label"])
            orgao = first_present(ex, ["orgao_julgador"])
            if len(texto) < 40:
                return {"text": ""}
            resultado = label_map.get(label.lower(), label) if label else "nao especificado"
            resposta = (f"Decisao proferida pelo orgao {orgao}. " if orgao else "")
            resposta += f"Resultado do julgamento: {resultado}."
            if unanim:
                resposta += f" Unanimidade: {unanim}."
            return {"text": make_convo(
                "Voce e um assistente juridico especialista em direito brasileiro.",
                f"Analise a decisao judicial a seguir e classifique o resultado do julgamento:\n\n{texto[:4000]}",
                resposta,
            )}

        ds = ds_raw.map(map_court, remove_columns=ds_raw.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 80)
        print("      Validos:", len(ds))
        if len(ds):
            datasets_list.append(ds)
    except Exception as e:
        print("      ERRO (ignorado):", e)

    # ---- Datasets 2 e 3: mapeamento GENERICO robusto (schema desconhecido) ----
    generic_sources = [
        ("celsowm/codigo_civil_brasileiro_lei_10406_2002",
         "Voce e um assistente juridico especialista no Codigo Civil brasileiro (Lei 10.406/2002)."),
        ("0rakul0/cpc_2015_brasil",
         "Voce e um assistente juridico especialista no Codigo de Processo Civil brasileiro (Lei 13.105/2015)."),
    ]
    instr_keys = ["instruction", "instrucao", "pergunta", "question", "prompt", "input", "entrada"]
    out_keys = ["output", "resposta", "answer", "completion", "saida", "response"]
    text_keys = ["text", "texto", "conteudo", "content", "artigo", "article", "body", "dispositivo"]

    for idx, (name, system_txt) in enumerate(generic_sources, start=2):
        print(f"\n[{idx}/3] {name} ...")
        try:
            ds_raw = load_dataset(name, split="train", token=args.hf_token)
            print("      Colunas:", ds_raw.column_names, "| Exemplos:", len(ds_raw))

            def map_generic(ex):
                instr = first_present(ex, instr_keys)
                out = first_present(ex, out_keys)
                if instr and out:
                    return {"text": make_convo(system_txt, instr, out)}
                body = first_present(ex, text_keys)
                if not body:
                    parts = [f"{k}: {str(v).strip()}" for k, v in ex.items()
                             if isinstance(v, str) and v.strip()]
                    body = "\n".join(parts)
                if len(body) < 40:
                    return {"text": ""}
                return {"text": make_convo(
                    system_txt,
                    "Reproduza e explique, com suas proprias palavras, o seguinte dispositivo legal:",
                    body[:4000],
                )}

            ds = ds_raw.map(map_generic, remove_columns=ds_raw.column_names)
            ds = ds.filter(lambda x: len(x["text"]) > 80)
            print("      Validos:", len(ds))
            if len(ds):
                datasets_list.append(ds)
        except Exception as e:
            print("      ERRO (ignorado):", e)

    if not datasets_list:
        raise RuntimeError("Nenhum dataset carregado. Verifique o HF_TOKEN e a conexao de rede.")

    dataset_final = concatenate_datasets(datasets_list).shuffle(seed=args.seed)

    if args.max_samples and len(dataset_final) > args.max_samples:
        dataset_final = dataset_final.select(range(args.max_samples))
        print(f"\n  Limitado a {args.max_samples} exemplos (--max_samples).")

    print(f"\n  >>> Dataset final: {len(dataset_final)} exemplos")
    print(f"  >>> Amostra (350 chars):\n{dataset_final[0]['text'][:350]}\n")

    eval_ds = None
    if args.eval_ratio and 0 < args.eval_ratio < 0.5 and len(dataset_final) > 200:
        split = dataset_final.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
        print(f"  >>> Split treino/eval: {len(train_ds)} / {len(eval_ds)}")
    else:
        train_ds = dataset_final

    return train_ds, eval_ds


# =============================================================================
# 6. TREINAMENTO
# =============================================================================
def train(model, tokenizer, train_ds, eval_ds, args):
    log_banner("CONFIGURANDO TREINAMENTO")
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    name, _, _ = get_gpu_info()
    bf16 = is_bfloat16_supported()
    fp16 = not bf16
    eff_batch = args.batch_size * args.gradient_accumulation_steps
    print(f"  GPU={name} | bf16={bf16} fp16={fp16} | optim={args.optim}")
    print(f"  batch={args.batch_size} x accum={args.gradient_accumulation_steps} => effective={eff_batch}")
    print(f"  epochs={args.epochs} | lr={args.learning_rate} | seq={args.max_seq_length}")

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_kwargs = dict(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        max_seq_length=args.max_seq_length,
        optim=args.optim,
        fp16=fp16,
        bf16=bf16,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        output_dir=str(ckpt_dir),
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=True,        # reduz padding => mais rapido
        remove_unused_columns=False,
    )
    if eval_ds is not None:
        cfg_kwargs.update(
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="steps",
            eval_steps=max(args.save_steps, 50),
        )

    config = SFTConfig(**cfg_kwargs)

    # Compatibilidade entre versoes do TRL: `tokenizer` (antigo) x `processing_class` (novo)
    base_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=config,
    )
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **base_kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **base_kwargs)

    # Treina somente nas respostas (mascara o prompt). Marcadores do template gemma-4.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    print_gpu_stats("ANTES TREINO")
    log_banner("INICIANDO TREINAMENTO")
    start = time.time()
    stats = trainer.train()
    mins = (time.time() - start) / 60
    print(f"\n  >>> Treino concluido em {mins:.1f} min | loss final={stats.training_loss:.4f}")
    print_gpu_stats("DEPOIS TREINO")

    final_dir = out_dir / "modelo_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    return trainer, final_dir


# =============================================================================
# 7. SALVAR / EXPORTAR (LoRA + merged 16-bit sempre; GGUF opcional e nao-fatal)
# =============================================================================
def save_model(model, tokenizer, args, final_dir):
    log_banner("SALVANDO MODELO FINAL")
    out_dir = Path(args.output_dir)

    # 1) Adaptadores LoRA (ou pesos full) - SEMPRE
    print(f"  [1] Salvando em {final_dir} ...")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("      OK.")

    # 2) Merge 16-bit (melhor qualidade p/ deploy) - so faz sentido com LoRA
    if not args.full_finetune and not args.skip_merge:
        merged_dir = out_dir / "merged_16bit"
        print(f"  [2] Mesclando LoRA->base (16-bit) em {merged_dir} ...")
        try:
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
            print("      Merge 16-bit OK (melhor qualidade).")
        except Exception as e:
            print(f"      AVISO (nao-fatal): merge 16-bit falhou: {e}")

    # 3) GGUF (opcional) - pode falhar em modelo multimodal; nunca derruba o run
    if args.gguf:
        gguf_dir = out_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        cleanup()
        for q in args.gguf_quants:
            print(f"  [3] Exportando GGUF '{q}' ...")
            try:
                model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=q)
                print(f"      GGUF {q} OK.")
            except Exception as e:
                print(f"      AVISO (nao-fatal): GGUF {q} falhou: {e}")
        gg = sorted(gguf_dir.glob("*.gguf"))
        if gg:
            print("      Arquivos GGUF:")
            for f in gg:
                print(f"        {f.name} -> {round(f.stat().st_size/1024**3, 2)} GB")


# =============================================================================
# 8. INFERENCIA DE TESTE
# =============================================================================
def test_inference(model, tokenizer, args):
    log_banner("TESTE DE INFERENCIA")
    from unsloth import FastModel

    try:
        FastModel.for_inference(model)
        messages = [{"role": "user", "content": args.test_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")

        print(f"  Prompt: {args.test_prompt}\n")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=512,
                use_cache=True,
                temperature=1.0,   # defaults oficiais Gemma 4
                top_p=0.95,
                top_k=64,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"  Resposta:\n{resp.strip()}")
        print("\n  --- Inferencia OK ---")
    except Exception as e:
        print(f"  AVISO (nao-fatal): teste de inferencia falhou: {e}")


# =============================================================================
# 9. MAIN
# =============================================================================
def main():
    args = parse_args()

    # Token: CLI > env HF_TOKEN
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not args.hf_token or len(args.hf_token) < 8:
        print("ERRO: forneca o token via --hf_token ou env HF_TOKEN (gemma-4 exige aceite de licenca).")
        sys.exit(1)
    os.environ["HF_TOKEN"] = args.hf_token

    # Ajuste de LR para full fine-tuning, se o usuario deixou o padrao de LoRA
    if args.full_finetune and abs(args.learning_rate - 2e-4) < 1e-12:
        args.learning_rate = 1e-5
        print(f"[info] full_finetune: ajustando learning_rate -> {args.learning_rate}")

    print("\n" + "=" * 72)
    print("  FINE-TUNING GEMMA-4-12B JURIDICO - OTIMIZADO H100")
    print("=" * 72)
    print(f"  precisao={args.precision} | full_finetune={args.full_finetune} | "
          f"LoRA r={args.lora_r}/alpha={args.lora_alpha}")
    print(f"  seq={args.max_seq_length} | batch={args.batch_size} x accum="
          f"{args.gradient_accumulation_steps} | epochs={args.epochs}")
    print(f"  output_dir={args.output_dir}")
    print("=" * 72)

    t0 = time.time()
    check_dependencies()
    model, tokenizer = load_model_and_tokenizer(args)
    model = setup_peft(model, args)
    train_ds, eval_ds = prepare_dataset(tokenizer, args)
    trainer, final_dir = train(model, tokenizer, train_ds, eval_ds, args)
    save_model(model, tokenizer, args, final_dir)
    test_inference(model, tokenizer, args)

    # Benchmark completo estilo startup de IA (opcional, nao-fatal)
    if args.benchmark:
        log_banner("BENCHMARK COMPLETO (iniciando)")
        merged_path = Path(args.output_dir) / "merged_16bit"
        # Usa merged 16-bit (melhor qualidade) se existir, senao o modelo final
        eval_model = str(merged_path) if merged_path.exists() else str(final_dir)
        bench_cmd = [
            sys.executable, "benchmark.py",
            "--model_path", eval_model,
            "--output_dir", str(Path(args.output_dir) / "benchmark_results"),
            "--suite", args.benchmark_suite,
            "--batch_size", "auto",
            "--device", "cuda:0",
        ]
        try:
            result = subprocess.run(bench_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  AVISO: benchmark retornou erro (nao-fatal):\n{result.stderr[:600]}")
            else:
                print("  Benchmark concluido. Veja os resultados em:",
                      Path(args.output_dir) / "benchmark_results")
        except Exception as e:
            print(f"  AVISO: benchmark falhou (nao-fatal): {e}")

    total_min = (time.time() - t0) / 60
    log_banner("CONCLUIDO")
    print(f"  Tempo total: {total_min:.1f} min")
    print(f"  Modelo final (adaptadores/pesos): {final_dir}")
    if not args.full_finetune and not args.skip_merge:
        print(f"  Modelo mesclado 16-bit (melhor qualidade): {Path(args.output_dir) / 'merged_16bit'}")
    if args.gguf:
        print(f"  GGUF: {Path(args.output_dir) / 'gguf'}")
    print(f"  Checkpoints: {Path(args.output_dir) / 'checkpoints'}")
    print("\n  Para carregar o modelo treinado:")
    print(f"    from unsloth import FastModel")
    print(f"    model, tok = FastModel.from_pretrained('{final_dir}', load_in_4bit=False)")
    print("=" * 72)


if __name__ == "__main__":
    main()
