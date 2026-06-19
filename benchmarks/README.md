# SEF Benchmarks

Questa cartella contiene benchmark piccoli e ripetibili pensati per README e
tesi. L'obiettivo non e dimostrare che SEF e "piu veloce" di ogni alternativa,
ma misurare gli effetti di alcune scelte architetturali: batch, streaming,
materializzazione e piano di esecuzione.

## Benchmark 1: batch vs streaming vs mixed

File:

```bash
benchmarks/batch_stream_mix_benchmark.py
```

Il benchmark usa frame sintetici generati in memoria. Non richiede webcam,
video esterni, OpenCV o modelli YOLO. Le tre pipeline hanno la stessa struttura
logica minima:

```text
frame extractor -> frame processor -> signal extractor -> analyzer
```

Scenari:

| Scenario | Descrizione | Cosa dimostra |
|---|---|---|
| `batch` | Tutti gli stage espongono contratti batch. | Baseline offline/materializzata. |
| `streaming` | Tutti gli stage principali espongono contratti streaming. | Pipeline end-to-end streamable. |
| `mixed` | Frame source e processor sono streaming, signal extractor e analyzer sono batch. | Pipeline ibrida con materialization boundary. |

## Esecuzione rapida

Dal root del repository:

```bash
python benchmarks/batch_stream_mix_benchmark.py \
  --frame-count 900 \
  --width 320 \
  --height 180 \
  --repetitions 5 \
  --warmup 1
```

Output predefinito:

```text
benchmarks/results/batch_stream_mix/
```

File generati:

| File | Uso |
|---|---|
| `runs.csv` | Misure grezze di ogni ripetizione. |
| `summary.csv` | Valori mediani pronti per tabelle di tesi. |
| `summary.json` | Dati completi in formato strutturato. |
| `execution_plans.json` | Piano batch/streaming per ogni scenario. |
| `elapsed_seconds_median.png` | Grafico tempo mediano. |
| `fps_median.png` | Grafico throughput mediano. |
| `process_peak_rss_mib_median.png` | Grafico memoria RSS di picco del processo. |
| `materialization_count.png` | Grafico numero materializzazioni. |

Se Matplotlib non e installato, lo script produce grafici SVG equivalenti
senza dipendenze aggiuntive.
Per saltare i grafici:

```bash
python benchmarks/batch_stream_mix_benchmark.py --no-plots
```

## Parametri consigliati per la tesi

Usa prima il comando rapido. Se vuoi un risultato piu stabile:

```bash
python benchmarks/batch_stream_mix_benchmark.py \
  --frame-count 1500 \
  --width 320 \
  --height 180 \
  --repetitions 7 \
  --warmup 1
```

Se il computer e lento, riduci solo `--frame-count`; non cambiare troppe cose
insieme.

## Come leggere i risultati

Nel capitolo della tesi conviene commentare questi aspetti:

1. `batch` mostra il comportamento offline classico.
2. `streaming` mostra cosa succede quando tutti gli stage possono lavorare in
   modo progressivo.
3. `mixed` mostra il caso realistico in cui uno stage non e streamable e il
   runtime deve materializzare l'input.
4. `materialization_count` e `materialization_stage_ids` collegano la misura al
   planner, quindi non sono solo numeri ma decisioni architetturali osservabili.
5. `estimated_full_frame_sequence_mib` e `estimated_frame_queue_mib` spiegano
   perche lo streaming puo ridurre il working set anche quando il tempo totale
   non migliora molto.

Frase breve riutilizzabile:

```text
Il benchmark non misura la qualita degli algoritmi di computer vision, ma il
comportamento del runtime SEF al variare dei contratti esposti dagli stage. Gli
stessi passaggi logici vengono eseguiti in modalita batch, streaming e ibrida,
cosi da evidenziare l'effetto delle capability e dei materialization boundaries.
```

## Benchmark 2: latency policy sotto backpressure realtime

File:

```bash
benchmarks/latency_policy_benchmark.py
```

Il benchmark costruisce una pipeline streaming sintetica in cui la sorgente
produce frame piu velocemente di quanto il processor riesca a consumarli. In
questo modo la coda frame va in pressione e le policy di latenza diventano
osservabili.

Pipeline:

```text
realtime synthetic frame extractor
  -> slow streaming frame processor
  -> streaming frame-index signal extractor
  -> freshness analyzer
```

Scenari:

| Scenario | Policy SEF | Cosa dimostra |
|---|---|---|
| `blocking` | `blocking` | Conserva tutti i frame bloccando la sorgente quando la coda e piena. |
| `drop_newest` | `drop_newest` | Scarta i frame in arrivo quando la coda e piena, preservando quelli gia in attesa. |
| `drop_oldest` | `drop_oldest` | Rimuove frame vecchi dalla coda e privilegia frame recenti. |
| `adaptive_sampling` | `adaptive_sampling` | Aumenta o riduce il campionamento in base al riempimento della coda. |

Esecuzione consigliata:

```bash
python benchmarks/latency_policy_benchmark.py \
  --frame-count 240 \
  --width 160 \
  --height 90 \
  --frame-buffer-size 8 \
  --source-interval-seconds 0.0005 \
  --processor-delay-seconds 0.003 \
  --repetitions 5 \
  --warmup 1
```

Output predefinito:

```text
benchmarks/results/latency_policy/
```

File generati:

| File | Uso |
|---|---|
| `runs.csv` | Misure grezze di ogni ripetizione. |
| `summary.csv` | Valori mediani pronti per tabelle di tesi. |
| `summary.json` | Dati completi in formato strutturato. |
| `execution_plans.json` | Piano di esecuzione per verificare che la pipeline sia end-to-end streaming. |
| `elapsed_seconds_median.png` o `.svg` | Tempo mediano per policy. |
| `processed_ratio_median.png` o `.svg` | Quota di frame prodotti che arriva all'analyzer. |
| `policy_drop_ratio_median.png` o `.svg` | Quota di frame scartata dalla policy. |
| `mean_latency_ms_median.png` o `.svg` | Latenza media end-to-end dei frame processati. |
| `last_frame_staleness_frames_median.png` o `.svg` | Quanto e vecchio l'ultimo frame osservato rispetto alla sorgente. |

Metriche principali:

| Metrica | Interpretazione |
|---|---|
| `accepted_frames` | Frame accettati dalla policy nel buffer frame. |
| `dropped_frames` | Frame scartati dalla policy o rimossi dalla coda. |
| `processed_frames` | Frame che arrivano davvero all'analyzer. |
| `processed_ratio` | Rapporto tra frame processati e frame prodotti. |
| `policy_drop_ratio` | Rapporto tra frame scartati dalla policy e frame prodotti. |
| `mean_latency_ms` | Latenza media dei frame processati, dalla produzione al consumo nell'analyzer. |
| `mean_staleness_frames` | Distanza media, in frame, tra frame visto e ultimo frame prodotto. |
| `last_frame_staleness_frames` | Distanza tra ultimo frame processato e ultimo frame prodotto. |
| `freshness_score` | Indice 0-1: valori vicini a 1 indicano frame finali piu recenti. |

Nota importante: `accepted_frames` e `processed_frames` non sono sempre uguali.
Alcune policy possono accettare un frame nel buffer e poi rimuoverlo dalla coda
per fare spazio a un frame piu recente. Per la tesi conviene distinguere
metriche della policy e metriche osservate a valle dall'analyzer.

Frase breve riutilizzabile:

```text
Il benchmark forza una condizione realtime in cui la sorgente produce dati piu
velocemente dello stage di elaborazione. A parita di pipeline, cambiare solo la
latency policy permette di osservare il compromesso tra copertura dei frame,
tempo di esecuzione e freschezza del dato elaborato.
```

## Benchmark 3: overhead del framework

File:

```bash
benchmarks/framework_overhead_benchmark.py
```

Il benchmark confronta lo stesso workload sintetico eseguito in tre modi:

```text
synthetic frame generation
  -> deterministic pixel transform
  -> signal extraction
  -> mean analyzer
```

Scenari:

| Scenario | Descrizione | Cosa dimostra |
|---|---|---|
| `direct_loop` | Esecuzione procedurale diretta, senza `Pipeline`, planner o runtime SEF. | Baseline minima dello stesso workload. |
| `sef_batch` | Pipeline SEF con contratti batch. | Overhead di componenti, planner, buffer e output contract in modalita offline. |
| `sef_streaming` | Pipeline SEF end-to-end streaming. | Overhead/beneficio del runtime streaming con buffer bounded e task concorrenti. |

Il baseline `direct_loop` usa comunque gli artifact di dominio SEF (`Frame`,
`Signal`, `BoxSignalSample`). Questo rende il confronto piu corretto: la misura
isola soprattutto il costo dell'orchestrazione del framework, non il costo di
sostituire completamente il modello dati.

Esecuzione consigliata:

```bash
python benchmarks/framework_overhead_benchmark.py \
  --frame-count 1200 \
  --width 160 \
  --height 90 \
  --repetitions 7 \
  --warmup 1
```

Output predefinito:

```text
benchmarks/results/framework_overhead/
```

File generati:

| File | Uso |
|---|---|
| `runs.csv` | Misure grezze di ogni ripetizione. |
| `summary.csv` | Valori mediani pronti per tabelle di tesi. |
| `summary.json` | Dati completi in formato strutturato. |
| `execution_plans.json` | Piano SEF per gli scenari `sef_batch` e `sef_streaming`. |
| `elapsed_seconds_median.png` o `.svg` | Tempo mediano per scenario. |
| `fps_median.png` o `.svg` | Throughput mediano. |
| `overhead_vs_direct_ratio.png` o `.svg` | Rapporto tra tempo SEF e baseline diretta. |
| `process_peak_rss_mib_median.png` o `.svg` | RSS di picco mediana. |

Metriche principali:

| Metrica | Interpretazione |
|---|---|
| `elapsed_seconds_median` | Tempo mediano di esecuzione del workload. |
| `fps_median` | Frame elaborati al secondo. |
| `overhead_vs_direct_ratio` | Quante volte lo scenario e piu lento/veloce rispetto al baseline diretto. |
| `overhead_vs_direct_seconds` | Differenza assoluta di tempo rispetto al baseline diretto. |
| `process_peak_rss_mib_median` | Memoria RSS di picco mediana del processo. |
| `streaming_stage_count` | Numero di stage pianificati in streaming. |
| `batch_stage_count` | Numero di stage pianificati in batch. |

Frase breve riutilizzabile:

```text
Il benchmark misura il costo dell'astrazione introdotta da SEF rispetto a una
implementazione procedurale dello stesso workload. Il confronto non serve a
dimostrare che il framework sia sempre piu veloce, ma a quantificare il prezzo
pagato per modularita, contratti espliciti, planner, output standardizzati e
supporto streaming.
```

## Tabelle utili in tesi

Tabella principale:

| Scenario | Tempo mediano | FPS mediani | RSS picco | Stage streaming | Stage batch | Materializzazioni |
|---|---:|---:|---:|---:|---:|---:|
| batch | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` |
| streaming | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` |
| mixed | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` |

Tabella planner:

| Scenario | Streamable end-to-end | Materialization stage | Interpretazione |
|---|---|---|---|
| batch | false | - | Esecuzione offline completamente batch. |
| streaming | true | - | Tutti gli stage preservano lo stream. |
| mixed | false | `signal_extraction` | Il segnale batch forza la materializzazione dei frame. |

Tabella latency policy:

| Policy | Tempo mediano | Frame processati | Frame scartati | Latenza media | Staleness finale | Interpretazione |
|---|---:|---:|---:|---:|---:|---|
| blocking | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | Massima copertura, sorgente rallentata dalla backpressure. |
| drop_newest | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | Evita backlog aggiuntivo ma puo mantenere frame meno recenti. |
| drop_oldest | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | Privilegia freschezza e preview realtime. |
| adaptive_sampling | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | Compromesso dinamico basato sul riempimento della coda. |

Tabella overhead framework:

| Scenario | Tempo mediano | FPS mediani | Overhead vs direct | RSS picco | Stage streaming | Interpretazione |
|---|---:|---:|---:|---:|---:|---|
| direct_loop | da `summary.csv` | da `summary.csv` | 1.00x | da `summary.csv` | 0 | Baseline procedurale senza runtime SEF. |
| sef_batch | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | 0 | Costo della pipeline batch e dei contratti. |
| sef_streaming | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | da `summary.csv` | Costo/beneficio del runtime streaming. |

## Note metodologiche

- Ogni scenario viene eseguito in un processo Python separato, cosi la memoria
  RSS di picco e piu confrontabile tra scenari.
- Il tempo misurato esclude l'avvio del processo figlio e misura solo
  `pipeline.run()`.
- I frame sono sintetici per evitare variabilita dovuta a codec, disco,
  webcam o modelli esterni.
- Nel benchmark sulle policy di latenza il processor e volutamente piu lento
  della sorgente. Questo non rappresenta un algoritmo specifico, ma una
  situazione controllata di backpressure.
- Nel benchmark sull'overhead, il baseline diretto non sostituisce gli artifact
  di dominio SEF: evita solo il runtime di pipeline, cosi il confronto rimane
  focalizzato sull'astrazione architetturale.
- Riporta sempre parametri macchina e comando usato: CPU, RAM, sistema
  operativo, Python version, numero frame, dimensione frame e ripetizioni.
