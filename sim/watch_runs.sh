#!/bin/bash
# Estado das runs do sim: processo, etapa, progresso da simulacao e ETA.
#
# Uso:   ./sim/watch_runs.sh [padrao]        # uma leitura
#        watch -n 30 -t ./sim/watch_runs.sh  # atualizando sozinho
#
# O padrao default cobre a varredura percent-parallelism; passe outro glob
# para acompanhar um conjunto diferente de runs.
cd "$(dirname "$0")/.." || exit 1
PADRAO="${1:-sim/runs/hls_time_driven_percent_*/2026*}"

printf '%-7s %-11s %-8s %-9s %-22s %-10s %-8s %-9s %s\n' \
  RUN RUN_ID PID TEMPO ETAPA PROGRESSO OCIOSO RITMO ETA
printf '%.0s-' {1..106}; printf '\n'

ativas=0
for run in $PADRAO; do
  run="${run%/}"
  [ -d "$run" ] || continue
  nome=$(basename "$(dirname "$run")"); nome="${nome##*_}"
  # Um projeto pode ter varias runs. O sufixo do diretorio e' o hash do
  # projeto, igual entre elas; o que distingue e' o timestamp do inicio.
  run_id=$(basename "$run"); run_id="${run_id%%-*}"
  run_id="${run_id:4:2}/${run_id:6:2} ${run_id:9:2}:${run_id:11:2}"
  log="$run/logs/post-synth-sim-xsim.log"

  pid=$(pgrep -f -- "bin/python -m sim resume --run $run" | head -1)
  tempo="--"
  if [ -n "$pid" ]; then
    # Tempo de parede, nao de CPU: o processo Python so espera o xsim filho,
    # entao o tempo de CPU dele fica perto de zero mesmo com a run ativa.
    tempo=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    ativas=$((ativas + 1))
  fi

  read -r etapa feitas total decorrido < <(./.venv/bin/python - "$run" <<'PY'
import json, sys, datetime
from pathlib import Path
run = Path(sys.argv[1])
ordem = ("prepare","vitis-project","csim","hls-synth","cosim-setup",
         "export","vivado-synth","post-synth-sim","power")
try:
    estado = json.loads((run / "status.json").read_text())
except Exception:
    print("-- 0 0 0"); raise SystemExit
etapas = estado.get("stages") or {}
# A primeira etapa nao concluida diz onde a run realmente esta.
for nome in ordem:
    e = etapas.get(nome) or {}
    if e.get("state") not in ("success", "skipped"):
        atual = f'{nome}:{e.get("state","pendente")}'
        break
else:
    atual = f'fim:{estado.get("state","?")}'

feitas = total = 0
log = run / "logs/post-synth-sim-xsim.log"
if log.is_file():
    for linha in log.read_text(errors="replace").splitlines():
        if "RTL Simulation : " in linha and linha.split("RTL Simulation : ")[1][:1].isdigit():
            n, _, t = linha.split("RTL Simulation : ")[1].split('[')[0].partition('/')
            feitas, total = int(n.strip()), int(t.strip())

p = (etapas.get("post-synth-sim") or {}).get("started_at")
decorrido = 0
if p:
    inicio = datetime.datetime.fromisoformat(p.replace("Z", "+00:00"))
    decorrido = int((datetime.datetime.now(datetime.timezone.utc) - inicio).total_seconds())
print(atual, feitas, total, decorrido)
PY
)

  prog="--"; [ "$total" -gt 0 ] && prog="$feitas/$total"
  ocioso="--"
  [ -f "$log" ] && ocioso="$(( ($(date +%s) - $(stat -c %Y "$log")) / 60 ))min"

  ritmo="--"; eta="--"
  if [ -n "$pid" ] && [ "$feitas" -gt 0 ] && [ "$decorrido" -gt 0 ]; then
    s=$(( decorrido / feitas ))
    ritmo="${s}s/tx"
    eta="$(( (total - feitas) * s / 3600 ))h"
  fi

  printf '%-7s %-11s %-8s %-9s %-22s %-10s %-8s %-9s %s\n' \
    "$nome" "$run_id" "${pid:---}" "$tempo" "$etapa" "$prog" "$ocioso" "$ritmo" "$eta"
done

printf '\n%s   |   %d run(s) ativa(s)\n' "$(date '+%H:%M:%S')" "$ativas"
