"""Gera tráfego contínuo no /recommend para a demonstração do monitoramento.

Os alertas do Prometheus olham para o que está acontecendo agora (taxa por
segundo), então sem requisições chegando não há o que medir. Este script manda
uma requisição a cada intervalo, ciclando por uma lista fixa de clientes.

Perfis:
  normal   quase só clientes dos segmentos em que o modelo é confiante
           (10% incertos), o alerta de baixa confiança não dispara;
  incerto  o público muda para segmentos em que o modelo tem pouca certeza
           (80% incertos), o alerta MuitasRecomendacoesIncertas dispara.

Uso (com a API rodando, por exemplo via docker compose up):
  python scripts/gerar_trafego.py --perfil normal
  python scripts/gerar_trafego.py --perfil incerto --segundos 90

Só usa a biblioteca padrão do Python, então roda sem instalar nada.
"""
import argparse
import itertools
import json
import time
import urllib.error
import urllib.request

# Um cliente por segmento. "confiante" e "incerto" seguem o critério da API
# (0,05 < P(celular melhor) < 0,95) com o modelo atual em models/policy.json.
JOVEM_SEM = {"age": 25, "poutcome": "nonexistent"}   # confiante (P = 99,97%)
ADULTO_SEM = {"age": 45, "poutcome": "nonexistent"}  # confiante (P = 97,1%)
JOVEM_COM = {"age": 25, "poutcome": "success"}       # incerto (P = 87,4%)
ADULTO_COM = {"age": 40, "poutcome": "success"}      # incerto (P = 80,6%)
SENIOR_SEM = {"age": 70, "poutcome": "failure"}      # incerto (P = 65,5%)
SENIOR_COM = {"age": 65, "poutcome": "success"}      # incerto (P = 37,1%)

PERFIS = {
    "normal": [JOVEM_SEM] * 5 + [ADULTO_SEM] * 4 + [SENIOR_COM],
    "incerto": [JOVEM_SEM, ADULTO_SEM, JOVEM_COM, ADULTO_COM, SENIOR_SEM,
                SENIOR_COM, SENIOR_COM, SENIOR_SEM, ADULTO_COM, SENIOR_COM],
}


def recomendar(url: str, cliente: dict) -> str:
    req = urllib.request.Request(f"{url}/recommend", data=json.dumps(cliente).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    # O esquema da URL é validado em main() (só http/https).
    with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
        return json.load(resp)["recommended_channel"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--perfil", choices=sorted(PERFIS), default="normal")
    parser.add_argument("--segundos", type=float, default=120, help="duração total")
    parser.add_argument("--intervalo", type=float, default=0.5, help="pausa entre requisições")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    if not args.url.startswith(("http://", "https://")):
        parser.error("--url precisa começar com http:// ou https://")

    print(f"Perfil {args.perfil!r} por {args.segundos:.0f}s em {args.url} (Ctrl+C para parar)")
    fim = time.monotonic() + args.segundos
    enviados = erros = 0
    for cliente in itertools.cycle(PERFIS[args.perfil]):
        if time.monotonic() >= fim:
            break
        try:
            recomendar(args.url, cliente)
            enviados += 1
        except (urllib.error.URLError, OSError) as exc:
            erros += 1
            print(f"  falha ao chamar a API: {exc}")
        if (enviados + erros) % 20 == 0:
            print(f"  {enviados} recomendações enviadas, {erros} falhas")
        time.sleep(args.intervalo)
    print(f"Fim: {enviados} recomendações enviadas, {erros} falhas")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido.")
