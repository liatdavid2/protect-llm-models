import os
import time
from typing import Any, Dict, List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv(override=False)

from harmful_content_input_guard.config import EMBEDDING_MODEL_NAME as HARMFUL_EMBEDDING_MODEL_NAME
from harmful_content_input_guard.features import EmbeddingEncoder as HarmfulEmbeddingEncoder
from harmful_content_input_guard.latest_run import get_latest_run_dir as get_latest_harmful_run_dir
from harmful_content_input_guard.model import load_model as load_harmful_model

from pii_output_guard.config import EMBEDDING_MODEL_NAME as PII_EMBEDDING_MODEL_NAME
from pii_output_guard.features import EmbeddingEncoder as PiiEmbeddingEncoder
from pii_output_guard.latest_run import get_latest_run_dir as get_latest_pii_run_dir
from pii_output_guard.model import load_model as load_pii_model

from prompt_injection_input_guard.config import EMBEDDING_MODEL_NAME as PROMPT_EMBEDDING_MODEL_NAME
from prompt_injection_input_guard.features import EmbeddingEncoder as PromptEmbeddingEncoder
from prompt_injection_input_guard.latest_run import get_latest_run_dir as get_latest_prompt_run_dir
from prompt_injection_input_guard.model import load_model as load_prompt_model

from system_prompt_leakage_output_guard.config import (
    EMBEDDING_MODEL_NAME as SYSTEM_PROMPT_LEAKAGE_EMBEDDING_MODEL_NAME,
)
from system_prompt_leakage_output_guard.features import (
    EmbeddingEncoder as SystemPromptLeakageEmbeddingEncoder,
)
from system_prompt_leakage_output_guard.latest_run import (
    get_latest_run_dir as get_latest_system_prompt_leakage_run_dir,
)
from system_prompt_leakage_output_guard.model import (
    load_model as load_system_prompt_leakage_model,
)

APP_TITLE = "Secure LLM Gateway"
APP_VERSION = "1.0.0"

DEFAULT_PROMPT_THRESHOLD = float(os.getenv("PROMPT_INJECTION_THRESHOLD", "0.70"))
DEFAULT_HARMFUL_THRESHOLD = float(os.getenv("HARMFUL_CONTENT_THRESHOLD", "0.70"))
DEFAULT_PII_THRESHOLD = float(os.getenv("PII_OUTPUT_THRESHOLD", "0.70"))
DEFAULT_SYSTEM_PROMPT_LEAKAGE_THRESHOLD = float(
    os.getenv("SYSTEM_PROMPT_LEAKAGE_OUTPUT_THRESHOLD", "0.70")
)

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_SMALL_MODEL = os.getenv("DEFAULT_SMALL_MODEL", "qwen2.5:0.5b")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

STEP_PROMPT_GUARD = "prompt_injection_guard"
STEP_HARMFUL_GUARD = "harmful_content_guard"
STEP_SMALL_MODEL = "small_model"
STEP_PII_OUTPUT_GUARD = "pii_output_guard"
STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD = "system_prompt_leakage_output_guard"

ALLOWED_STEPS = {
    STEP_PROMPT_GUARD,
    STEP_HARMFUL_GUARD,
    STEP_SMALL_MODEL,
    STEP_PII_OUTPUT_GUARD,
    STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD,
}

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model_name: str = Field(default=DEFAULT_SMALL_MODEL)

    prompt_injection_threshold: float = Field(
        default=DEFAULT_PROMPT_THRESHOLD, ge=0.0, le=1.0
    )
    harmful_content_threshold: float = Field(
        default=DEFAULT_HARMFUL_THRESHOLD, ge=0.0, le=1.0
    )
    pii_output_threshold: float = Field(
        default=DEFAULT_PII_THRESHOLD, ge=0.0, le=1.0
    )
    system_prompt_leakage_output_threshold: float = Field(
        default=DEFAULT_SYSTEM_PROMPT_LEAKAGE_THRESHOLD, ge=0.0, le=1.0
    )

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    disabled_steps: List[str] = Field(default_factory=list)


class GuardResult(BaseModel):
    predicted_label: int
    malicious_probability: float
    threshold: float
    blocked: bool
    latency_ms: float


class StepMetric(BaseModel):
    enabled: bool
    skipped: bool
    blocked: bool = False
    latency_ms: float
    detail: Optional[str] = None


class ChatResponse(BaseModel):
    allowed: bool
    prompt_guard: Optional[GuardResult] = None
    harmful_guard: Optional[GuardResult] = None
    pii_output_guard: Optional[GuardResult] = None
    system_prompt_leakage_output_guard: Optional[GuardResult] = None

    model_name: Optional[str] = None
    response: Optional[str] = None
    block_reason: Optional[str] = None

    disabled_steps: List[str]
    step_metrics: Dict[str, StepMetric]

    total_latency_ms: float
    model_latency_ms: float


class InferenceState(TypedDict, total=False):
    prompt: str
    model_name: str

    prompt_injection_threshold: float
    harmful_content_threshold: float
    pii_output_threshold: float
    system_prompt_leakage_output_threshold: float

    temperature: float
    max_tokens: Optional[int]
    disabled_steps: List[str]

    prompt_guard: Dict[str, Any]
    harmful_guard: Dict[str, Any]
    pii_output_guard: Dict[str, Any]
    system_prompt_leakage_output_guard: Dict[str, Any]

    allowed: bool
    block_reason: Optional[str]
    response: Optional[str]
    model_latency_ms: float
    step_metrics: Dict[str, Dict[str, Any]]


class SecureGateway:
    def __init__(self) -> None:
        prompt_run_dir = get_latest_prompt_run_dir()
        harmful_run_dir = get_latest_harmful_run_dir()
        pii_run_dir = get_latest_pii_run_dir()
        system_prompt_leakage_run_dir = get_latest_system_prompt_leakage_run_dir()

        self.prompt_guard_model_path = prompt_run_dir / "xgb_prompt_injection.joblib"
        self.harmful_guard_model_path = harmful_run_dir / "xgb_harmful_content.joblib"
        self.pii_output_guard_model_path = pii_run_dir / "xgb_pii_output_guard.joblib"
        self.system_prompt_leakage_output_guard_model_path = (
            system_prompt_leakage_run_dir / "xgb_system_prompt_leakage_output_guard.joblib"
        )

        self.prompt_encoder = PromptEmbeddingEncoder(PROMPT_EMBEDDING_MODEL_NAME)
        self.harmful_encoder = HarmfulEmbeddingEncoder(HARMFUL_EMBEDDING_MODEL_NAME)
        self.pii_encoder = PiiEmbeddingEncoder(PII_EMBEDDING_MODEL_NAME)
        self.system_prompt_leakage_encoder = SystemPromptLeakageEmbeddingEncoder(
            SYSTEM_PROMPT_LEAKAGE_EMBEDDING_MODEL_NAME
        )

        self.prompt_guard_model = load_prompt_model(self.prompt_guard_model_path)
        self.harmful_guard_model = load_harmful_model(self.harmful_guard_model_path)
        self.pii_output_guard_model = load_pii_model(self.pii_output_guard_model_path)
        self.system_prompt_leakage_output_guard_model = load_system_prompt_leakage_model(
            self.system_prompt_leakage_output_guard_model_path
        )

    def run_prompt_guard(self, prompt: str, threshold: float) -> Dict[str, Any]:
        start = time.perf_counter()
        X = self.prompt_encoder.encode([prompt])
        pred = int(self.prompt_guard_model.predict(X)[0])
        proba = float(self.prompt_guard_model.predict_proba(X)[0][1])
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "predicted_label": pred,
            "malicious_probability": round(proba, 6),
            "threshold": threshold,
            "blocked": proba >= threshold,
            "latency_ms": round(latency_ms, 2),
        }

    def run_harmful_guard(self, prompt: str, threshold: float) -> Dict[str, Any]:
        start = time.perf_counter()
        X = self.harmful_encoder.encode([prompt])
        pred = int(self.harmful_guard_model.predict(X)[0])
        proba = float(self.harmful_guard_model.predict_proba(X)[0][1])
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "predicted_label": pred,
            "malicious_probability": round(proba, 6),
            "threshold": threshold,
            "blocked": proba >= threshold,
            "latency_ms": round(latency_ms, 2),
        }

    def run_pii_output_guard(self, text: str, threshold: float) -> Dict[str, Any]:
        start = time.perf_counter()
        X = self.pii_encoder.encode([text])
        pred = int(self.pii_output_guard_model.predict(X)[0])
        proba = float(self.pii_output_guard_model.predict_proba(X)[0][1])
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "predicted_label": pred,
            "malicious_probability": round(proba, 6),
            "threshold": threshold,
            "blocked": proba >= threshold,
            "latency_ms": round(latency_ms, 2),
        }

    def run_system_prompt_leakage_output_guard(
        self, text: str, threshold: float
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        X = self.system_prompt_leakage_encoder.encode([text])
        pred = int(self.system_prompt_leakage_output_guard_model.predict(X)[0])
        proba = float(
            self.system_prompt_leakage_output_guard_model.predict_proba(X)[0][1]
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "predicted_label": pred,
            "malicious_probability": round(proba, 6),
            "threshold": threshold,
            "blocked": proba >= threshold,
            "latency_ms": round(latency_ms, 2),
        }

    def run_small_model(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            resp = requests.post(
                f"{DEFAULT_OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=DEFAULT_OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to call protected model via Ollama: {exc}",
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "response": data.get("response", ""),
            "latency_ms": round(latency_ms, 2),
        }


gateway = SecureGateway()


def init_step_metrics(disabled_steps: List[str]) -> Dict[str, Dict[str, Any]]:
    return {
        STEP_PROMPT_GUARD: {
            "enabled": STEP_PROMPT_GUARD not in disabled_steps,
            "skipped": STEP_PROMPT_GUARD in disabled_steps,
            "blocked": False,
            "latency_ms": 0.0,
            "detail": "disabled by request" if STEP_PROMPT_GUARD in disabled_steps else None,
        },
        STEP_HARMFUL_GUARD: {
            "enabled": STEP_HARMFUL_GUARD not in disabled_steps,
            "skipped": STEP_HARMFUL_GUARD in disabled_steps,
            "blocked": False,
            "latency_ms": 0.0,
            "detail": "disabled by request" if STEP_HARMFUL_GUARD in disabled_steps else None,
        },
        STEP_SMALL_MODEL: {
            "enabled": STEP_SMALL_MODEL not in disabled_steps,
            "skipped": STEP_SMALL_MODEL in disabled_steps,
            "blocked": False,
            "latency_ms": 0.0,
            "detail": "disabled by request" if STEP_SMALL_MODEL in disabled_steps else None,
        },
        STEP_PII_OUTPUT_GUARD: {
            "enabled": STEP_PII_OUTPUT_GUARD not in disabled_steps,
            "skipped": STEP_PII_OUTPUT_GUARD in disabled_steps,
            "blocked": False,
            "latency_ms": 0.0,
            "detail": "disabled by request" if STEP_PII_OUTPUT_GUARD in disabled_steps else None,
        },
        STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD: {
            "enabled": STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD not in disabled_steps,
            "skipped": STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD in disabled_steps,
            "blocked": False,
            "latency_ms": 0.0,
            "detail": "disabled by request"
            if STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD in disabled_steps
            else None,
        },
    }


def validate_disabled_steps(disabled_steps: List[str]) -> None:
    unknown_steps = sorted(set(disabled_steps) - ALLOWED_STEPS)
    if unknown_steps:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown steps in disabled_steps",
                "unknown_steps": unknown_steps,
                "allowed_steps": sorted(ALLOWED_STEPS),
            },
        )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    validate_disabled_steps(req.disabled_steps)

    total_start = time.perf_counter()

    state: InferenceState = {
        "prompt": req.prompt,
        "model_name": req.model_name,
        "prompt_injection_threshold": req.prompt_injection_threshold,
        "harmful_content_threshold": req.harmful_content_threshold,
        "pii_output_threshold": req.pii_output_threshold,
        "system_prompt_leakage_output_threshold": req.system_prompt_leakage_output_threshold,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "disabled_steps": req.disabled_steps,
        "response": None,
        "model_latency_ms": 0.0,
        "block_reason": None,
        "allowed": True,
        "step_metrics": init_step_metrics(req.disabled_steps),
    }

    if STEP_PROMPT_GUARD not in req.disabled_steps:
        prompt_guard = gateway.run_prompt_guard(
            req.prompt, req.prompt_injection_threshold
        )
        state["prompt_guard"] = prompt_guard
        state["step_metrics"][STEP_PROMPT_GUARD]["latency_ms"] = prompt_guard["latency_ms"]

        if prompt_guard["blocked"]:
            state["allowed"] = False
            state["block_reason"] = "Prompt blocked by prompt injection guard."
            state["step_metrics"][STEP_PROMPT_GUARD]["blocked"] = True

    if state["allowed"] and STEP_HARMFUL_GUARD not in req.disabled_steps:
        harmful_guard = gateway.run_harmful_guard(
            req.prompt, req.harmful_content_threshold
        )
        state["harmful_guard"] = harmful_guard
        state["step_metrics"][STEP_HARMFUL_GUARD]["latency_ms"] = harmful_guard["latency_ms"]

        if harmful_guard["blocked"]:
            state["allowed"] = False
            state["block_reason"] = "Prompt blocked by harmful content guard."
            state["step_metrics"][STEP_HARMFUL_GUARD]["blocked"] = True

    if state["allowed"] and STEP_SMALL_MODEL not in req.disabled_steps:
        model_result = gateway.run_small_model(
            prompt=req.prompt,
            model_name=req.model_name,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        state["response"] = model_result["response"]
        state["model_latency_ms"] = model_result["latency_ms"]
        state["step_metrics"][STEP_SMALL_MODEL]["latency_ms"] = model_result["latency_ms"]

    if state["allowed"] and STEP_SMALL_MODEL in req.disabled_steps:
        state["allowed"] = False
        state["block_reason"] = "No response generated because small_model step is disabled."
        state["step_metrics"][STEP_SMALL_MODEL]["detail"] = "response generation disabled"

    if (
        state["allowed"]
        and STEP_PII_OUTPUT_GUARD not in req.disabled_steps
        and state.get("response")
    ):
        pii_output_guard = gateway.run_pii_output_guard(
            state["response"], req.pii_output_threshold
        )
        state["pii_output_guard"] = pii_output_guard
        state["step_metrics"][STEP_PII_OUTPUT_GUARD]["latency_ms"] = pii_output_guard["latency_ms"]

        if pii_output_guard["blocked"]:
            state["allowed"] = False
            state["response"] = None
            state["block_reason"] = "Model output blocked by PII output guard."
            state["step_metrics"][STEP_PII_OUTPUT_GUARD]["blocked"] = True

    if (
        state["allowed"]
        and STEP_PII_OUTPUT_GUARD not in req.disabled_steps
        and not state.get("response")
    ):
        state["step_metrics"][STEP_PII_OUTPUT_GUARD]["skipped"] = True
        state["step_metrics"][STEP_PII_OUTPUT_GUARD]["enabled"] = False
        state["step_metrics"][STEP_PII_OUTPUT_GUARD]["detail"] = "no model output to inspect"

    if (
        state["allowed"]
        and STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD not in req.disabled_steps
        and state.get("response")
    ):
        system_prompt_leakage_output_guard = gateway.run_system_prompt_leakage_output_guard(
            state["response"], req.system_prompt_leakage_output_threshold
        )
        state["system_prompt_leakage_output_guard"] = system_prompt_leakage_output_guard
        state["step_metrics"][STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD]["latency_ms"] = (
            system_prompt_leakage_output_guard["latency_ms"]
        )

        if system_prompt_leakage_output_guard["blocked"]:
            state["allowed"] = False
            state["response"] = None
            state["block_reason"] = "Model output blocked by system prompt leakage output guard."
            state["step_metrics"][STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD]["blocked"] = True

    if (
        state["allowed"]
        and STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD not in req.disabled_steps
        and not state.get("response")
    ):
        state["step_metrics"][STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD]["skipped"] = True
        state["step_metrics"][STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD]["enabled"] = False
        state["step_metrics"][STEP_SYSTEM_PROMPT_LEAKAGE_OUTPUT_GUARD]["detail"] = (
            "no model output to inspect"
        )

    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    return ChatResponse(
        allowed=state["allowed"],
        prompt_guard=GuardResult(**state["prompt_guard"]) if state.get("prompt_guard") else None,
        harmful_guard=GuardResult(**state["harmful_guard"]) if state.get("harmful_guard") else None,
        pii_output_guard=GuardResult(**state["pii_output_guard"]) if state.get("pii_output_guard") else None,
        system_prompt_leakage_output_guard=(
            GuardResult(**state["system_prompt_leakage_output_guard"])
            if state.get("system_prompt_leakage_output_guard")
            else None
        ),
        model_name=req.model_name if state["allowed"] and state.get("response") else None,
        response=state.get("response"),
        block_reason=state.get("block_reason"),
        disabled_steps=req.disabled_steps,
        step_metrics={k: StepMetric(**v) for k, v in state["step_metrics"].items()},
        total_latency_ms=round(total_latency_ms, 2),
        model_latency_ms=state.get("model_latency_ms", 0.0),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)