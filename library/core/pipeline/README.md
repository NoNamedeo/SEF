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

- `SegmentedPipelineExecutor`: attraversa l'intera pipeline e sceglie la
  modalita migliore stage per stage delegando ai segment executor.
- `FrameSegmentExecutor`: gestisce frame extraction, frame processors e frame
  exporters.
- `SignalSegmentExecutor`: gestisce signal extraction e signal cleaners.
- `AnalysisSegmentExecutor`: gestisce analyzer batch/streaming, fan-out e
  visualizer finali.
- `PipelineBoundaryMaterializer`: materializza frame o signal solo alle
  boundary esplicite tra segmenti streaming e stage batch.
- `PipelineExecutionResources`: raccoglie buffer e artifact condivisi per una
  singola run.
- `PipelineExecutionPolicy`: contratto strategy per decidere batch/streaming.
  `DefaultPipelineExecutionPolicy` e l'implementazione predefinita, ma puo
  essere sostituita con policy latency-first, memory-first o domain-specific.
- `PipelineExecutionLookahead`: risponde alle domande sugli stage streamable a
  valle, condividendo la stessa logica tra planner e runtime.
- `PipelineRuntimeState`: rappresenta lo stato runtime corrente di frame e
  signal, distinguendo dati materializzati da stream con task pendenti.
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
  Il binding valida e risolve gli indici target in un unico punto.
- `IntermediateFrameCapture`: cattura snapshot intermedi per debug e artifact.
- `FrameProcessingStage`: adapter per processori frame context-aware.
- `SingleFrameProcessorAdapter`: adatta un `ISingleFrameProcessor` al contratto
  `IFrameBufferProcessor` e allo streaming.

## Flusso di Esecuzione

```text
Pipeline.run()
  -> PipelineEventInjector.inject(...)
  -> SegmentedPipelineExecutor.run()
     -> frame_extraction
     -> frame_processing[*]
     -> frame_exporters[*]
     -> signal_extraction
     -> signal_cleaners[*]
     -> analyzers[*] + visualizers[*]
     -> intermediate_frame_visualizers[*]
  -> PipelineOutputAssembler.build(...)
```

`SegmentedPipelineExecutor` non valuta la pipeline come un blocco unico. Ogni
stage viene attraversato nell'ordine reale di esecuzione e viene scelto il modo
migliore per quel punto specifico della catena:

- se lo stage puo streammare e l'input e gia streaming, lo stage resta nello
  stream segment corrente;
- se lo stage puo streammare, l'input e batch, e un successore streaming puo
  trarne beneficio, il runtime apre un nuovo stream segment;
- se lo stage richiede la sequenza completa, il runtime chiude eventuali task
  pendenti, materializza solo il segmento necessario e prosegue in batch;
- dopo una boundary batch, la pipeline puo ripartire in streaming quando un
  successore rende utile riaprire un buffer bounded.

Il piano prodotto da `PipelineExecutionPlanner` usa le stesse regole di
`SegmentedPipelineExecutor`, quindi `Pipeline.execution_plan()` descrive le
decisioni che verranno poi applicate durante `run()`.

## Politica Batch, Streaming e Ibrida

La pipeline non sceglie solo tra "tutto batch" e "tutto streaming". Il modello
reale supporta tre casi:

- batch end-to-end: il frame extractor produce gia un buffer completo e gli
  stage successivi lavorano su sequenze materializzate;
- streaming end-to-end: tutti gli stage necessari supportano buffer bounded e
  non richiedono l'intera sequenza;
- ibrido: una parte iniziale puo scorrere in streaming, poi uno stage batch-only
  forza una materializzazione prima di proseguire.

La decisione e centralizzata nel contratto `PipelineExecutionPolicy`. Il core
fornisce `DefaultPipelineExecutionPolicy`, una strategia conservativa e
cost-aware che considera capability, stato dello stream corrente, domanda
downstream e stime di memoria.

La policy riceve un `PipelineStagePolicyContext` per ogni decisione. Il context
espone:

- capability dello stage;
- presenza di input gia streaming;
- presenza di successori streaming;
- presenza di consumer progressivi, come visualizer streaming;
- stime opzionali di coda bounded e materializzazione.

La policy default evita switch isolati del tipo batch -> streaming -> batch:
in quel caso i thread e i buffer bounded introdurrebbero overhead senza ridurre
la pressione di memoria. Quando invece uno switch permette a uno o piu stage
successivi di lavorare progressivamente, oppure quando la coda bounded e
stimata piu economica della materializzazione, lo streaming viene riaperto anche
dopo una boundary batch.

Una policy custom puo essere iniettata direttamente:

```python
from library.core.pipeline.Pipeline import Pipeline

pipeline = Pipeline(context, execution_policy=MyExecutionPolicy())
```

## Materializzazione

La materializzazione e una boundary esplicita tra un segmento streaming e uno
stage che richiede la sequenza completa. Serve a preservare i contratti dei
componenti batch-only senza trasformare tutta la pipeline in batch.

Esempi:

- `StreamingFrameExtractor -> StreamingProcessor -> BatchProcessor`: il runtime
  streamma il prefisso, poi materializza prima del `BatchProcessor`;
- `BatchProcessor -> StreamingFrameExporter`: dopo il batch processor, il
  runtime puo riaprire uno stream verso l'exporter se questo riduce memoria o
  latenza;
- `StreamingSignalExtractor -> StreamingAnalyzer + BatchAnalyzer`: il signal
  viene distribuito a piu consumer; l'analyzer streaming riceve campioni
  progressivi, mentre il batch analyzer riceve una materializzazione dedicata;
- `StreamingAnalyzer -> StreamingVisualizer`: il visualizer riceve i punti
  progressivi e il render finale evita duplicazioni sugli stessi target.

La materializzazione non e necessaria quando:

- la pipeline e batch end-to-end, perche i dati sono gia prodotti come sequenze
  complete;
- la pipeline e streaming end-to-end, perche nessuno stage richiede l'intera
  sequenza;
- lo streaming sarebbe un passaggio isolato senza benefici downstream.

## Contratti Runtime

Il core non importa componenti concreti. La scelta runtime si basa su due
contratti stabili:

- capability dichiarate tramite `StageCapabilities`;
- interfacce streaming in `library/core/interfaces/StreamingContracts.py`.

`PipelineComponentCapabilities` e la fonte unica per verificare se un componente
puo davvero streammare. Un componente e streamable solo se:

- implementa l'interfaccia streaming corretta;
- dichiara `supports_streaming=True`;
- non dichiara `requires_complete_sequence=True`.

I visualizer fanno eccezione perche la loro capacita progressiva dipende dal
contratto `IStreamingVisualizer`.

## Segmenti Downstream

Anche la parte dopo i frame e segmentata. Non serve che tutti i componenti
downstream siano streamable per mantenere benefici locali di streaming. Ogni
stage viene valutato nel proprio contesto:

- signal extractor;
- signal cleaners;
- analyzers;
- visualizers;
- frame exporters.

Se uno di questi componenti non e streamable, la pipeline materializza solo la
parte necessaria, esegue lo stage batch e puo ripartire in streaming se gli
stage successivi lo rendono conveniente.

Gli analyzer supportano un caso ulteriore: fan-out misto. Se alcuni analyzer
streammano e altri richiedono un `ISignal` completo, il runtime usa subscription
separate sullo stesso signal buffer. Gli analyzer streaming e i visualizer
progressivi possono produrre output mentre un consumer dedicato materializza il
signal per gli analyzer batch.

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
- Ogni collaboratore runtime deve avere una sola ragione di cambiamento.
- Le regole di capability devono restare centralizzate in
  `PipelineComponentCapabilities`.
- Le regole di scelta batch/streaming devono restare in strategie che
  implementano `PipelineExecutionPolicy`.
- Le domande di lookahead sugli stage downstream devono passare da
  `PipelineExecutionLookahead`.
- La serializzazione degli output deve restare in `PipelineOutputAssembler` e
  negli exporter dedicati.
- Le modifiche allo streaming devono includere test su ordine, chiusura e abort
  dei buffer.
