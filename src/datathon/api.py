"""API de recomendação de canal (FastAPI)."""
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from datathon.features import segment_of
from datathon.model import PosteriorModel

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "policy.json"

# Mesmo critério de incerteza do Golden Set (caso 5): 0,05 < P(celular melhor) < 0,95.
LIMITES_INCERTEZA = (0.05, 0.95)

# Métricas expostas em /metrics para o Prometheus.
REQUISICOES = Counter("api_requisicoes", "Requisições HTTP recebidas pela API.",
                      ["metodo", "rota", "status"])
LATENCIA = Histogram("api_latencia_segundos", "Tempo de resposta das requisições HTTP.", ["rota"])
RECOMENDACOES = Counter("recomendacoes", "Recomendações feitas, por canal e segmento.",
                        ["canal", "segmento"])
RECOMENDACOES_INCERTAS = Counter(
    "recomendacoes_incertas",
    "Recomendações de baixa confiança (candidatas a revisão humana), por segmento.",
    ["segmento"])


@lru_cache(maxsize=1)
def get_model() -> PosteriorModel:
    return PosteriorModel.load(Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))


def _model_or_503() -> PosteriorModel:
    try:
        return get_model()
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=f"Modelo indisponível: {exc}") from exc


class Cliente(BaseModel):
    model_config = ConfigDict(extra="ignore")

    age: int = Field(ge=18, le=100, description="Idade do cliente")
    poutcome: Literal["success", "failure", "nonexistent"] = Field(
        description="Resultado da campanha anterior")
    mode: Literal["explotacao", "thompson"] = "explotacao"
    seed: int | None = None


class Recomendacao(BaseModel):
    segment: str
    recommended_channel: str
    estimated_conversion_rate: dict[str, float]
    prob_cellular_better: float
    mode: str


app = FastAPI(title="Datathon MLET — Recomendação de canal de oferta",
              description="Thompson Sampling contextual com esquecimento (celular × telefone).")


@app.middleware("http")
async def medir_requisicoes(request: Request, call_next):
    inicio = perf_counter()
    response = await call_next(request)
    # Usa o molde da rota (ex.: /recommend), não a URL crua, para não criar uma série por URL;
    # caminhos inexistentes (404) ficam agrupados em "desconhecida".
    rota = request.scope.get("route")
    caminho = rota.path if rota else "desconhecida"
    if caminho != "/metrics":
        LATENCIA.labels(caminho).observe(perf_counter() - inicio)
        REQUISICOES.labels(request.method, caminho, str(response.status_code)).inc()
    return response


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict:
    _model_or_503()
    return {"status": "ok"}


@app.post("/recommend", response_model=Recomendacao)
def recommend(cliente: Cliente) -> Recomendacao:
    model = _model_or_503()
    segment = segment_of(cliente.age, cliente.poutcome)
    channel = model.recommend(segment, cliente.mode, cliente.seed)
    prob = model.prob_cellular_better(segment)
    RECOMENDACOES.labels(channel, segment).inc()
    if LIMITES_INCERTEZA[0] < prob < LIMITES_INCERTEZA[1]:
        RECOMENDACOES_INCERTAS.labels(segment).inc()
    return Recomendacao(
        segment=segment,
        recommended_channel=channel,
        estimated_conversion_rate=model.estimated_rates(segment),
        prob_cellular_better=prob,
        mode=cliente.mode,
    )
