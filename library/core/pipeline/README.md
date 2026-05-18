# Core Pipeline Architecture

Questa cartella contiene il motore applicativo della pipeline SEF. Il codice
qui presente non implementa algoritmi OpenCV, visualizzazioni Matplotlib o UI:
coordina componenti esterni tramite interfacce, applica le regole di esecuzione
e restituisce output riproducibili.

## Obiettivi Architetturali

- Mantenere il core indipendente dai componenti concreti.
- Separare costruzione, pianificazione, esecuzione, monitoraggio e output.
- Supportare pipeline batch, streaming e ibride senza duplicare logica.
- Rendere gli errori diagnosticabili tramite stage espliciti.
- Preservare la riproducibilita di ogni run tramite config e code export.
- Favorire estensioni tramite interfacce e piccoli collaboratori sostituibili.

## Responsabilita dei Componenti

### Facciata Pubblica

- `Pipeline`: entry point per eseguire un singolo `PipelineContext`. Coordina
  injector, executor e output assembler, ma non contiene logica di stage.
- `PipelineOrchestrator`: facciata applicativa per run sincrone, submit
  asincroni e trigger event-driven.
- `ThreadedPipelineRunner`: esegue pipeline tramite `ThreadPoolExecutor`,
  gestendo lifecycle events, retry, deduplicazione degli ID e monitor.

### Costruzione e Configurazione

- `PipelineContext`: contenitore immutabile delle dipendenze di una pipeline.
  Valida invarianti strutturali prima dell'esecuzione.
- `FluentPipelineBuilder`: builder programmatico per codice Python.
- `ConfigPipelineBuilder`: builder dichiarativo basato su configurazioni
  esterne e `PluginRegistry`.
- `DefaultPipelineFactory`: factory predefinita che costruisce una `Pipeline`
  da un context e dai metadati runtime.
- `StreamRuntimeConfig` e `LatencyPolicy`: configurano buffer, backpressure e
  politiche di latenza per lo streaming.

### Pianificazione

- `PipelineExecutionPlanner`: genera un piano leggibile prima della run.
- `PipelineExecutionPlan`: rappresenta stages, materialization boundary,
  modalita batch/streaming e stime di memoria.
- `PipelineComponentCapabilities`: fonte unica delle regole per determinare se
  un componente puo essere eseguito in streaming.

### Esecuzione

- `AdaptivePipelineExecutor`: sceglie il percorso batch o streaming in base
  alle capability dei componenti.
- `FramePipelineExecutor`: esegue frame extraction e frame processors,
  materializzando quando incontra componenti batch-only.
- `FrameExporterExecutor`: esegue frame exporters batch o streaming.
- `SignalPipelineExecutor`: esegue la coda batch: signal extraction, cleaning e
  analyzers.
- `StreamingSignalTailExecutor`: costruisce ed esegue il grafo concorrente
  della coda streaming.
- `VisualizationExecutor`: risolve binding dei visualizer e crea i rispettivi
  `VisualizationContext`.
- `PipelineStageExecutor`: esegue un singolo stage e normalizza gli errori in
  `PipelineExecutionError`.
- `PipelineBuffers`: utility per materializzazione e abort dei buffer.

### Output, Export e Osservabilita

- `PipelineOutputAssembler`: converte il risultato grezzo in `PipelineOutputs`,
  allegando metadata, piano di esecuzione e artifact di riproducibilita.
- `PipelineConfigExporter`: esporta una run in configurazione dichiarativa.
- `PipelineCodeExporter`: genera codice Python equivalente alla configurazione.
- `PipelineExportUtils`: funzioni di supporto per serializzazione JSON/YAML.
- `InMemoryPipelineMonitor`: monitor in memoria degli stati di run.
- `InMemoryPipelineOutputStore`: store opzionale in memoria per gli output.
- `PipelineRunSnapshot`: snapshot immutabile dello stato osservabile.

### Eventi e Branching

- `PipelineEventInjector`: inietta event bus e metadata nei componenti che
  implementano `IEventEmitter`.
- `BranchingCoordinator`: ascolta eventi di dominio, valuta regole di branching
  e dispatcha trigger per pipeline secondarie.
- `VisualizerBinding`: collega visualizer a specifici risultati degli analyzer.
- `IntermediateFrameCapture`: cattura snapshot intermedi per debug e artifact.
- `FrameProcessingStage`: adapter per processori frame context-aware.
- `SingleFrameProcessorAdapter`: adatta un `ISingleFrameProcessor` al contratto
  `IFrameBufferProcessor` e allo streaming.

## Flusso di Esecuzione

```text
Pipeline.run()
  -> PipelineEventInjector.inject(...)
  -> AdaptivePipelineExecutor.run()
     -> FramePipelineExecutor.build(...)
        (la parte frame puo essere batch, streaming o ibrida)
     -> se la coda downstream e interamente streamable:
          StreamingSignalTailExecutor.run(...)
        altrimenti:
          (eventuali stream frame gia avviati vengono materializzati)
          FramePipelineExecutor.materialize(...)
          FrameExporterExecutor.run_batch(...)
          SignalPipelineExecutor.run_batch(...)
          VisualizationExecutor.run_final_visualizers(...)
  -> PipelineOutputAssembler.build(...)
```

## Batch, Streaming, Ibrido e Materializzazione

La pipeline non sceglie solo tra "tutto batch" e "tutto streaming". Il modello
reale supporta tre casi:

- batch end-to-end: il frame extractor produce gia un buffer completo e gli
  stage successivi lavorano su sequenze materializzate;
- streaming end-to-end: tutti gli stage necessari supportano buffer bounded e
  non richiedono l'intera sequenza;
- ibrido: una parte iniziale puo scorrere in streaming, poi uno stage batch-only
  forza una materializzazione prima di proseguire.

Una pipeline puo quindi attraversare una sequenza mista di componenti:

- uno stage streaming riceve e pubblica dati tramite buffer bounded;
- uno stage batch-only richiede l'intera sequenza;
- quando uno stage batch-only segue uno stage streaming, il core crea una
  materialization boundary;
- il piano di esecuzione espone queste boundary prima della run.

La materializzazione serve solo nel caso ibrido. Non e necessaria quando:

- la pipeline e batch end-to-end, perche i dati sono gia materializzati;
- la pipeline e streaming end-to-end, perche nessuno stage richiede l'intera
  sequenza.

La coda dopo il frame pipeline viene eseguita in streaming solo quando tutti i
componenti downstream supportano streaming:

- signal extractor;
- signal cleaners;
- analyzers;
- visualizers;
- frame exporters.

Se uno di questi componenti non e streamable, la pipeline non scarta il lavoro
streaming gia possibile a monte: materializza il frame stream corrente, poi usa
il percorso batch per preservare correttezza e semplicita operativa.

## Gestione Errori

Ogni stage viene eseguito tramite `PipelineStageExecutor`. Un errore sollevato
da un componente esterno viene incapsulato in `PipelineExecutionError`, che
espone:

- `stage`: nome dello stage fallito, per esempio `signal_extraction`;
- `cause`: eccezione originale.

Il runner usa questo errore per emettere lifecycle events, applicare retry policy
e aggiornare il monitor.

## Regole di Estensione

Per aggiungere un nuovo comportamento:

1. Definire o riusare una interfaccia in `library/core/interfaces`.
2. Implementare il componente concreto fuori da `library/core/pipeline`.
3. Registrare il componente nel `PluginRegistry`.
4. Lasciare che builder, planner ed executor lo trattino tramite il contratto.

Non introdurre import di componenti concreti dentro questa cartella, ad eccezione
dei factory/exporter gia previsti. Le dipendenze devono continuare a puntare
verso interfacce e modelli core.

## Esempio Minimo

```python
from library.core.pipeline.Pipeline import Pipeline

pipeline = Pipeline(context)
plan = pipeline.execution_plan()
outputs = pipeline.run()

print(plan.as_text())
print(outputs.metadata.pipeline_id)
```

## Checklist di Manutenzione

- `Pipeline` deve rimanere una facciata, non un orchestratore operativo pesante.
- Ogni executor deve avere una sola ragione di cambiamento.
- Le regole di capability devono restare centralizzate in
  `PipelineComponentCapabilities`.
- La serializzazione degli output deve restare in `PipelineOutputAssembler` e
  negli exporter dedicati.
- Le modifiche allo streaming devono includere test su ordine, chiusura e abort
  dei buffer.
