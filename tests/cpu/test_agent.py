"""CPU-only tests of the agent: code extraction, the PyTorch-delegation guard,
prompts, the optimize loop (scripted LLM + fake evaluator), the subprocess
sandbox, the OpenAI-compatible client and the evaluation report."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from kernelpilot.agent.llm import OpenAICompatLLM, ScriptedLLM
from kernelpilot.agent.loop import Agent, forbidden_call, save_best
from kernelpilot.agent.prompts import AttemptView, extract_code, system_prompt, user_prompt
from kernelpilot.agent.result import EvalResult
from kernelpilot.agent.sandbox import run_candidate
from kernelpilot.cli import build_parser
from kernelpilot.evaluation import report
from kernelpilot.specs import SPECS, get_spec

TRITON_DIR = Path(__file__).resolve().parents[2] / "kernelpilot" / "triton_kernels"


def ok_result(speedup: float) -> EvalResult:
    return EvalResult(stage="ok", checks_passed=8, bench=[
        {"shape": [4096, 4096], "ms": 1.0 / speedup, "torch_ms": 1.0, "speedup": speedup,
         "gbs": 100.0, "tflops": 0.1, "pct_peak_bw": 30.0, "pct_peak_flops": 1.0}])


def reply(code: str, lang: str = "python") -> str:
    return f"Plan: try something.\n```{lang}\n{code}\n```\n"


# ----------------------------------------------------------------- extraction

def test_extract_code_tagged():
    assert extract_code(reply("def kernel_fn(x): pass"), "triton") == "def kernel_fn(x): pass"


def test_extract_code_prefers_language_and_longest():
    text = "```bash\nls\n```\n```python\nshort\n```\n```python\nmuch longer code\n```"
    assert extract_code(text, "triton") == "much longer code"


def test_extract_code_hip_and_untagged_fallback():
    assert extract_code(reply("__global__ void k(){}", "cpp"), "hip") == "__global__ void k(){}"
    assert extract_code("```\nkernel\n```", "triton") == "kernel"


def test_extract_code_none():
    assert extract_code("I think you should use shared memory.", "triton") is None


# ----------------------------------------------------------------- guard

@pytest.mark.parametrize("name", list(SPECS))
def test_own_triton_kernels_not_flagged(name):
    path = TRITON_DIR / f"{name}.py"
    assert forbidden_call(name, path.read_text(), "triton") is None


@pytest.mark.parametrize("kernel,code", [
    ("softmax", "def kernel_fn(x):\n    return torch.softmax(x, -1)"),
    ("softmax", "def kernel_fn(x):\n    return x.softmax(dim=-1)"),
    ("matmul", "def kernel_fn(a, b):\n    return a @ b"),
    ("matmul", "def kernel_fn(a, b):\n    return torch.mm(a, b)"),
    ("layernorm", "def kernel_fn(x, w, b):\n    return F.layer_norm(x, x.shape[-1:], w, b)"),
    ("reduce_sum", "def kernel_fn(x):\n    return x.sum(dtype=torch.float32)"),
    ("attention", "def kernel_fn(q, k, v):\n    return F.scaled_dot_product_attention(q, k, v)"),
])
def test_cheats_are_flagged(kernel, code):
    assert forbidden_call(kernel, code, "triton")


def test_decorators_and_comments_not_flagged():
    code = '@triton.autotune(configs=[])\n@triton.jit\ndef k(x):\n    # torch.matmul(a, b)\n' \
           '    pass\n'
    assert forbidden_call("matmul", code, "triton") is None


# ----------------------------------------------------------------- prompts

def test_prompts_contain_spec_feedback_and_best():
    spec = get_spec("softmax")
    sys_p = system_prompt(spec, "triton", "float32")
    assert "kernel_fn" in sys_p and spec.signature in sys_p
    bad = EvalResult(stage="correctness", error="softmax[candidate] MISMATCH max_abs_err=3.1e-01",
                     checks_passed=3)
    hist = [AttemptView(1, "BEST_CODE", ok_result(1.3)), AttemptView(2, "BROKEN_CODE", bad)]
    p = user_prompt(spec, "triton", "float32", "Fake GPU", hist, hist[0], spec.bench_shapes)
    assert "def softmax_reference" in p
    assert "BEST_CODE" in p and "1.300x" in p
    assert "max_abs_err=3.1e-01" in p and "BROKEN_CODE" in p
    assert "passed 3 smaller shapes" in p


def test_hip_system_prompt_lists_arguments():
    p = system_prompt(get_spec("layernorm"), "hip", "float16")
    assert "torch::Tensor kernel_fn(torch::Tensor x0, torch::Tensor x1, torch::Tensor x2)" in p


# ----------------------------------------------------------------- loop

def test_loop_keeps_fastest_correct_and_never_accepts_wrong(tmp_path):
    codes = {
        "WRONG_BUT_FAST": EvalResult(stage="correctness", error="MISMATCH",
                                     bench=[{"speedup": 100.0}]),
        "COMPILE_ERR": EvalResult(stage="compile", error="SyntaxError: invalid syntax"),
        "OK_1_2": ok_result(1.2),
        "OK_0_9": ok_result(0.9),
        "OK_1_5": ok_result(1.5),
    }
    llm = ScriptedLLM([
        "no code here",
        reply("WRONG_BUT_FAST"),
        reply("COMPILE_ERR"),
        reply("OK_1_2"),
        reply("OK_0_9"),
        reply("def kernel_fn(x):\n    return torch.softmax(x, -1)"),
        reply("OK_1_5"),
    ])
    evaluated = []

    def fake_eval(code):
        evaluated.append(code)
        return codes[code]

    log = tmp_path / "run.jsonl"
    agent = Agent(llm, "softmax", log_path=log, evaluator=fake_eval, gpu_desc="Fake GPU",
                  verbose=False)
    run = agent.optimize(7)

    assert [a.result.stage for a in run.attempts] == [
        "no_code", "correctness", "compile", "ok", "ok", "rejected", "ok"]
    assert run.best.iteration == 7 and run.best_score == pytest.approx(1.5)
    assert run.attempts[4].is_regression and not run.attempts[4].is_best
    assert not run.attempts[1].is_best  # wrong kernels never win, however fast
    assert "torch.softmax" not in " ".join(evaluated)  # cheat never ran

    # Feedback flows into the next prompt.
    assert "SyntaxError: invalid syntax" in llm.calls[3][1]["content"]
    assert "No code block" in llm.calls[1][1]["content"]

    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 7
    assert {"code", "error", "bench", "stage", "score", "is_best", "model"} <= set(records[0])
    assert records[-1]["is_best"] and records[-1]["best_score_so_far"] == pytest.approx(1.5)

    out = save_best(run, tmp_path / "best")
    assert out.read_text().splitlines()[1] == "OK_1_5"


def test_loop_survives_llm_failure(tmp_path):
    class FlakyLLM:
        model = "flaky"

        def complete(self, messages):
            raise RuntimeError("HTTP 503")

    agent = Agent(FlakyLLM(), "vector_add", log_path=tmp_path / "x.jsonl",
                  evaluator=lambda c: ok_result(1.0), gpu_desc="Fake", verbose=False)
    run = agent.optimize(2)
    assert [a.result.stage for a in run.attempts] == ["crash", "crash"] and run.best is None


# ----------------------------------------------------------------- sandbox

def test_sandbox_reports_missing_gpu_without_crashing(tmp_path):
    res = run_candidate("def kernel_fn(a, b):\n    return a + b\n", "vector_add", "triton",
                        "float32", timeout_s=120, workdir=tmp_path)
    assert res.stage == "crash" and "no GPU" in res.error


def test_sandbox_timeout(tmp_path):
    res = run_candidate("x = 1\n", "vector_add", "triton", "float32", timeout_s=0.01,
                        workdir=tmp_path)
    assert res.stage == "timeout"


# ----------------------------------------------------------------- LLM client

def test_openai_compatible_client_roundtrip():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"model": "m", "choices": [{"message": {"content": "hi"}}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        llm = OpenAICompatLLM(model="m", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                              api_key="test-key")
        c = llm.complete([{"role": "user", "content": "hello"}])
    finally:
        server.shutdown()
    assert c.text == "hi" and c.prompt_tokens == 5
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer test-key"
    assert seen["body"]["messages"][0]["content"] == "hello"


def test_from_env_requires_model(monkeypatch):
    monkeypatch.delenv("KP_MODEL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatLLM.from_env()


# ----------------------------------------------------------------- evaluation + CLI

def test_evaluation_report(tmp_path):
    from kernelpilot.benchmark import BenchResult, save_results
    from kernelpilot.device import GPUInfo

    gpu = GPUInfo("Fake GPU", "nvidia", 300.0, {"float32": 8.0}, "test")
    rows = [BenchResult("softmax", "torch", "float32", (4096, 4096), 1.0, speedup_vs_torch=1.0),
            BenchResult("softmax", "hip:online_fused", "float32", (4096, 4096), 0.8,
                        speedup_vs_torch=1.25),
            BenchResult("softmax", "hip:naive", "float32", (4096, 4096), None, correct=False)]
    save_results(rows, gpu, tmp_path)
    runs = tmp_path / "runs"
    agent = Agent(ScriptedLLM([reply("A"), reply("B")]), "softmax", log_path=runs / "r.jsonl",
                  evaluator=lambda c: ok_result(1.1) if c == "B" else
                  EvalResult(stage="correctness", error="x"), gpu_desc="Fake", verbose=False)
    agent.optimize(2)
    text = report(tmp_path / "bench.json", runs)
    assert "1.25x (hip:online_fused)" in text and "1.10x" in text
    assert "| softmax | float32 | triton | scripted | 2 | 1 | 0 | 1 |" in text


def test_cli_optimize_args():
    a = build_parser().parse_args(["optimize", "--kernel", "softmax", "--iterations", "10"])
    assert a.kernel == "softmax" and a.iterations == 10 and a.lang == "triton"
