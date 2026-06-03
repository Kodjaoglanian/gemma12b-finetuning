#!/usr/bin/env python3
"""
Benchmark completo estilo startup de IA para modelo juridico treinado.

Executa benchmarks padrao da industria (MMLU, GPQA, etc.) + benchmarks
em portugues/juridicos (OAB, ENEM, BLUEX, ASSIN2) usando:
  - EleutherAI/lm-evaluation-harness (benchmarks internacionais)
  - eduagarcia/lm-evaluation-harness-pt (benchmarks PT-BR + juridico)

Gera:
  - JSON consolidado com todas as metricas
  - Tabela Markdown para README/report

Uso:
  python benchmark.py --model_path ./outputs_gemma4_juridico/modelo_final
  python benchmark.py --model_path ./outputs_gemma4_juridico/merged_16bit --batch_size auto
  python benchmark.py --model_path ./outputs_gemma4_juridico/merged_16bit --suite startup
  python benchmark.py --model_path ./outputs_gemma4_juridico/merged_16bit --suite pt --pt_harness_dir ./lm-evaluation-harness-pt

Nota: instalacao automatica do lm-eval-harness PT via run.sh ou manual.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =============================================================================
# SUITES DE BENCHMARK
# =============================================================================
# Benchmarks internacionais (padrao startups de AI)
STARTUP_BENCHMARKS = {
    "core_knowledge": {
        "tasks": "mmlu",
        "desc": "MMLU - Conhecimento geral (57 materias)",
    },
    "advanced_knowledge": {
        "tasks": "mmlu_pro",
        "desc": "MMLU-Pro - Conhecimento dificil e mais desafiador",
    },
    "reasoning": {
        "tasks": "gpqa",
        "desc": "GPQA - Raciocinio de nivel PhD",
    },
    "science": {
        "tasks": "arc_challenge",
        "desc": "ARC Challenge - Ciencias e raciocinio",
    },
    "commonsense": {
        "tasks": "hellaswag",
        "desc": "HellaSwag - Sentido comum",
    },
    "truthfulness": {
        "tasks": "truthfulqa_mc2",
        "desc": "TruthfulQA - Veracidade e alucinacoes",
    },
    "anaphora": {
        "tasks": "winogrande",
        "desc": "WinoGrande - Resolucao de anaphora",
    },
    "math": {
        "tasks": "gsm8k",
        "desc": "GSM8K - Matemática e resolucao de problemas",
    },
}

# Benchmarks juridicos e em portugues
PT_BR_BENCHMARKS = {
    "legal_knowledge": {
        "tasks": "oab_exams",
        "desc": "OAB Exams - Exame da Ordem dos Advogados do Brasil",
    },
    "national_exam": {
        "tasks": "enem_challenge",
        "desc": "ENEM Challenge - Exame Nacional do Ensino Medio",
    },
    "reading_comprehension": {
        "tasks": "bluex",
        "desc": "BLUEX - Compreensao de leitura em PT",
    },
    "entailment": {
        "tasks": "assin2_rte",
        "desc": "ASSIN2 RTE - Reconhecimento de Entailment Textual em PT",
    },
    "semantic_similarity": {
        "tasks": "assin2_sts",
        "desc": "ASSIN2 STS - Similaridade Semantica em PT",
    },
    "qa_nli": {
        "tasks": "faquad_nli",
        "desc": "FaQuAD-NLI - Natural Language Inference em PT",
    },
    "sentiment": {
        "tasks": "tweetsentbr",
        "desc": "TweetSentBR - Analise de sentimento em tweets PT-BR",
    },
    "hate_speech": {
        "tasks": "hatebr_offensive",
        "desc": "HateBR - Deteccao de discurso ofensivo/hate speech PT-BR",
    },
    "hate_speech_2": {
        "tasks": "portuguese_hate_speech",
        "desc": "Portuguese Hate Speech - Deteccao em tweets PT",
    },
}

ALL_BENCHMARKS = {**STARTUP_BENCHMARKS, **PT_BR_BENCHMARKS}


# =============================================================================
# CLI ARGS
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark completo Gemma-4-12B Juridico")
    p.add_argument("--model_path", type=str, required=True,
                   help="Caminho para o modelo (LoRA ou merged 16-bit)")
    p.add_argument("--output_dir", type=str, default="./benchmark_results",
                   help="Onde salvar os resultados")
    p.add_argument("--suite", type=str, default="full",
                   choices=["startup", "pt", "full"],
                   help="Suite de benchmarks: startup (internacionais), pt (portugues), full (todos)")
    p.add_argument("--batch_size", type=str, default="auto",
                   help="Batch size para eval (auto ou inteiro)")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Dispositivo CUDA")
    p.add_argument("--trust_remote_code", action="store_true", default=True,
                   help="Trust remote code")
    p.add_argument("--pt_harness_dir", type=str, default="./lm-evaluation-harness-pt",
                   help="Caminho para o lm-evaluation-harness-pt (clone do fork)")
    p.add_argument("--few_shot", type=str, default="",
                   help="Numero de few-shots (ex.: 'mmlu:5,hellaswag:10' ou deixe vazio para padrao)")
    p.add_argument("--limit", type=int, default=0,
                   help="Limita exemplos por task (0 = todos). Use para teste rapido.")
    p.add_argument("--skip_startup", action="store_true",
                   help="Pula benchmarks internacionais")
    p.add_argument("--skip_pt", action="store_true",
                   help="Pula benchmarks PT-BR")
    return p.parse_args()


# =============================================================================
# UTILS
# =============================================================================
def log_banner(msg: str):
    print("\n" + "=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def check_lm_eval():
    try:
        subprocess.run(["lm-eval", "--version"], capture_output=True, check=True)
        return True, None
    except FileNotFoundError:
        return False, "lm-eval nao encontrado. Instale: pip install lm-eval[api]"
    except subprocess.CalledProcessError:
        return True, None  # lm-eval existe mas --version pode nao existir


def check_pt_harness(pt_dir: str):
    p = Path(pt_dir) / "lm_eval" / "__init__.py"
    if p.exists():
        return True, None
    return False, (
        f"lm-evaluation-harness-PT nao encontrado em {pt_dir}.\n"
        "Clone: git clone https://github.com/eduagarcia/lm-evaluation-harness-pt "
        f"{pt_dir}\n"
        f"cd {pt_dir} && pip install -e ."
    )


def run_lm_eval(model_path, tasks, output_path, batch_size="auto", device="cuda:0",
                trust_remote_code=True, limit=0, pt_mode=False, pt_harness_dir=None):
    """Roda lm-eval para uma task ou grupo de tasks."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code={trust_remote_code}",
        "--tasks", tasks,
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", str(out),
        "--log_samples",
    ]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]

    # Se estamos usando o fork PT, precisa rodar do diretorio correto
    cwd = pt_harness_dir if pt_mode and pt_harness_dir else None
    env = os.environ.copy()

    print(f"  Rodando: {' '.join(cmd)}")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
        elapsed = time.time() - start
        print(f"  Concluido em {elapsed:.1f}s")
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[:500]}")
            return None, f"returncode={result.returncode}"
        return result.stdout, None
    except Exception as e:
        return None, str(e)


def parse_lm_eval_results(output_dir: Path, task_name: str):
    """Extrai metricas do JSON de resultados do lm-eval."""
    results_file = output_dir / "results.json"
    if not results_file.exists():
        # Tentar nome alternativo
        alt = output_dir / f"{task_name}_results.json"
        if alt.exists():
            results_file = alt
        else:
            return None

    try:
        with open(results_file) as f:
            data = json.load(f)
    except Exception:
        return None

    # lm-eval salva como {"results": {"task_name": {"metric": value}}}
    if "results" in data:
        for key, val in data["results"].items():
            if isinstance(val, dict):
                # Procurar metrica padrao (acc, acc_norm, exact_match, f1)
                for metric in ["acc,none", "acc_norm,none", "exact_match,none", "f1,none",
                               "acc", "acc_norm", "exact_match", "f1"]:
                    if metric in val:
                        score = val[metric]
                        stderr = val.get(f"{metric}_stderr", 0.0)
                        return {"score": score, "stderr": stderr, "metric": metric}
    return None


def format_score(score):
    if score is None:
        return "N/A"
    return f"{score * 100:.2f}%"


# =============================================================================
# MAIN BENCHMARK LOGIC
# =============================================================================
def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("  BENCHMARK COMPLETO - GEMMA-4-12B JURIDICO")
    print("=" * 72)
    print(f"  Modelo : {args.model_path}")
    print(f"  Suite  : {args.suite}")
    print(f"  Batch  : {args.batch_size} | Device: {args.device}")
    print("=" * 72)

    # ---- Verificacoes de dependencias ----
    ok, msg = check_lm_eval()
    if not ok:
        print(f"\n  ERRO: {msg}")
        print("  Instale: pip install lm-eval[api]")
        sys.exit(1)

    # Se suite inclui PT, verificar o fork PT-BR
    if args.suite in ("pt", "full") and not args.skip_pt:
        pt_ok, pt_msg = check_pt_harness(args.pt_harness_dir)
        if not pt_ok:
            print(f"\n  AVISO: {pt_msg}")
            print("  Serao rodados apenas os benchmarks internacionais.")
            args.skip_pt = True

    # ---- Seleciona benchmarks ----
    suite = {}
    if not args.skip_startup and args.suite in ("startup", "full"):
        suite.update(STARTUP_BENCHMARKS)
    if not args.skip_pt and args.suite in ("pt", "full"):
        suite.update(PT_BR_BENCHMARKS)

    if not suite:
        print("  Nenhum benchmark selecionado.")
        sys.exit(0)

    # ---- Roda benchmarks ----
    all_results = {}
    total_start = time.time()

    for key, info in suite.items():
        log_banner(f"BENCHMARK: {info['desc']}")
        print(f"  Task: {info['tasks']}")

        # Determina se precisa do fork PT-BR
        is_pt = key in PT_BR_BENCHMARKS
        pt_dir = args.pt_harness_dir if is_pt else None

        task_out = out_dir / key
        stdout, err = run_lm_eval(
            model_path=args.model_path,
            tasks=info["tasks"],
            output_path=task_out,
            batch_size=args.batch_size,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            limit=args.limit,
            pt_mode=is_pt,
            pt_harness_dir=pt_dir,
        )

        if err:
            print(f"  [FALHA] {err}")
            all_results[key] = {"desc": info["desc"], "score": None, "status": "failed", "error": err}
            continue

        metrics = parse_lm_eval_results(task_out, info["tasks"])
        if metrics:
            score_pct = metrics["score"] * 100
            stderr_pct = metrics["stderr"] * 100
            print(f"  [OK] Score: {score_pct:.2f}% (+/- {stderr_pct:.2f}%)")
            all_results[key] = {
                "desc": info["desc"],
                "score": metrics["score"],
                "score_pct": score_pct,
                "stderr": metrics["stderr"],
                "stderr_pct": stderr_pct,
                "metric": metrics["metric"],
                "status": "ok",
            }
        else:
            print("  [WARN] Resultado nao encontrado. Veja logs em:", task_out)
            all_results[key] = {"desc": info["desc"], "score": None, "status": "no_result"}

    total_elapsed = (time.time() - total_start) / 60

    # ---- Relatorio JSON ----
    report = {
        "model": args.model_path,
        "suite": args.suite,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_minutes": round(total_elapsed, 2),
        "results": all_results,
    }
    json_path = out_dir / "benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Relatorio JSON: {json_path}")

    # ---- Relatorio Markdown ----
    md_path = out_dir / "benchmark_report.md"
    with open(md_path, "w") as f:
        f.write("# Benchmark Report - Gemma-4-12B Juridico\n\n")
        f.write(f"**Modelo:** `{args.model_path}`\n\n")
        f.write(f"**Suite:** `{args.suite}` | **Tempo total:** {total_elapsed:.1f} min\n\n")
        f.write(f"**Timestamp:** {report['timestamp']}\n\n")
        f.write("---\n\n")

        # Startup
        f.write("## Benchmarks Internacionais (Startup/Industria)\n\n")
        f.write("| Benchmark | Score | +/- | Status |\n")
        f.write("|-----------|-------|-----|--------|\n")
        for k, info in STARTUP_BENCHMARKS.items():
            r = all_results.get(k, {})
            score = format_score(r.get("score"))
            stderr = f"{r['stderr_pct']:.2f}%" if r.get("stderr") is not None else "N/A"
            status = r.get("status", "not_run")
            f.write(f"| {info['desc']} | {score} | {stderr} | {status} |\n")

        f.write("\n## Benchmarks Juridicos e Portugues\n\n")
        f.write("| Benchmark | Score | +/- | Status |\n")
        f.write("|-----------|-------|-----|--------|\n")
        for k, info in PT_BR_BENCHMARKS.items():
            r = all_results.get(k, {})
            score = format_score(r.get("score"))
            stderr = f"{r['stderr_pct']:.2f}%" if r.get("stderr") is not None else "N/A"
            status = r.get("status", "not_run")
            f.write(f"| {info['desc']} | {score} | {stderr} | {status} |\n")

    print(f"  Relatorio Markdown: {md_path}")

    # ---- Resumo no terminal ----
    log_banner("RESUMO")
    ok_count = sum(1 for v in all_results.values() if v.get("status") == "ok")
    fail_count = sum(1 for v in all_results.values() if v.get("status") == "failed")
    total = len(all_results)
    print(f"  Total benchmarks : {total}")
    print(f"  OK               : {ok_count}")
    print(f"  Falhas           : {fail_count}")
    print(f"  Tempo total      : {total_elapsed:.1f} min")
    print(f"\n  Resultados salvos em: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
