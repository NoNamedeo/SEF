# SEF

SEF is currently under active development.
Public APIs may evolve before the first stable release.

SEF è un framework Python per l'estrazione e l'analisi di segnali da video e sequenze di immagini tramite componenti componibili di computer vision. Il progetto è costruito intorno a un `core` stabile, estendibile e relativamente agnostico rispetto agli algoritmi concreti: il motore esegue pipeline, mentre i moduli OpenCV, gli analyzer, i visualizer e la UI si innestano sopra tale nucleo.

## Core Authors

- Matteo Vittori

- Alejandro Innocenzi

## Public Documentation

The public English documentation starts at [docs/index.md](docs/index.md). It includes overview, quickstart, API contracts, config versioning, registry guarantees, plugin authoring, streaming runtime, error handling, versioning policy, reference pages, and runnable examples.

Nel repository convivono due facce complementari del sistema:

- una libreria modulare per costruire ed eseguire pipeline video;
- un'applicazione Streamlit, `SEF Studio`, che usa il core per comporre pipeline, configurarle, lanciarle e osservare risultati ed eventi.

## Obiettivi del progetto

- separare la definizione della pipeline dalla sua esecuzione;
- rendere sostituibili gli algoritmi tramite interfacce e registry;
- supportare sia uso programmatico sia uso dichiarativo via configurazione;
- permettere esecuzione sincrona, asincrona ed event-driven;
- produrre non solo dati analitici, ma anche artefatti visuali riutilizzabili dalla UI;
- offrire una base per casi d'uso diversi: tracking singolo, tracking multi-oggetto, optical flow, conteggi su barriere, playback annotato.

## Architettura in breve

### Struttura principale

```text
SEF/
├── library/
│   ├── core/
│   │   ├── artifacts/         # modelli dati di base: Frame, Signal, samples, graph data
│   │   ├── events/            # Event, EventBus, eventi di pipeline
│   │   ├── interfaces/        # contratti astratti dei componenti
│   │   ├── pipeline/          # builder, orchestrator, runner, monitor, contesto
│   │   ├── plugins/           # PluginRegistry e factory dei plugin built-in
│   │   ├── utils/             # utility OpenCV per selezioni geometriche
│   │   └── visualization/     # PipelineOutputs, metadata, VisualArtifact
│   ├── frame_extractors/      # acquisizione frame da video
│   ├── frame_processors/        # preprocessing dei frame
│   ├── signal_extractors/     # estrazione/tracking dei segnali
│   ├── signal_cleaners/       # smoothing, widening, outlier rejection
│   ├── analyzers/             # trasformazione del segnale in dati analitici
│   ├── visualizers/           # trasformazione dei dati in artefatti visuali
│   ├── retry_policies/        # politiche di retry per il runner
│   └── branching_rules/       # regole di branching event-driven
├── ui/
│   ├── components/            # componenti Streamlit
│   ├── services/              # orchestration applicativa lato UI
│   ├── models/                # modelli di supporto alla UI
│   └── state/                 # stato di sessione/canvas
├── tests/                     # test di core, registry, branching, builder
├── pipeline.yaml              # esempio di pipeline dichiarativa
├── requirements.txt           # dipendenze complete per libreria + UI
└── pyproject.toml             # packaging del pacchetto Python
```

### Livelli architetturali

Il progetto segue, in forma pragmatica, una struttura vicina alla Clean Architecture:

- `library/core/interfaces`: contratti astratti dei componenti;
- `library/core/artifacts` e `library/core/visualization`: modelli dati e output del dominio tecnico;
- `library/core/pipeline`: orchestrazione, esecuzione, monitoraggio, retry, contesto;
- `library/*` fuori da `core`: implementazioni concrete OpenCV/Matplotlib;
- `ui/*`: presentazione e composizione visuale.

La dipendenza va dal concreto verso l'astratto: la pipeline conosce le interfacce, non le implementazioni specifiche.

## Come funziona il sistema

### Flusso di esecuzione

Il cuore del sistema è la classe `Pipeline`, che esegue i passi nell'ordine seguente:

```mermaid
flowchart LR
    A[Frame Extractor] --> B[Frame Processors]
    B --> C[Signal Extractor]
    C --> D[Signal Cleaners]
    D --> E[Analyzers]
    E --> F[Visualizers]
    F --> G[PipelineOutputs]
```

In termini di responsabilità:

- il `FrameExtractor` produce un `FrameBuffer`;
- gli `ISingleFrameProcessor` trasformano un singolo `Frame`;
- gli `IFrameBufferProcessor` trasformano l'intero `FrameBuffer`;
- il `SignalExtractor` converte i frame in un `Signal`;
- i `SignalCleaner` raffinano il segnale;
- gli `Analyzer` producono dati strutturati (`IData`);
- i `Visualizer` producono `VisualArtifact` indipendenti dalla UI.

### Il ruolo di `PipelineContext`

`PipelineContext` è il contenitore immutabile dei collaboratori necessari all'esecuzione. Impone alcune invarianti:

- `frame_extractor` obbligatorio;
- `signal_extractor` obbligatorio;
- almeno un `analyzer` obbligatorio;
- processor, cleaner di segnale, visualizer e binding visuali opzionali;
- nessun campo può contenere `None`.

Questa scelta è importante perché la pipeline non decide nulla: riceve un contesto valido e lo esegue.

### Il ruolo di `Pipeline`

`library/core/pipeline/Pipeline.py` è il motore di esecuzione puro. Non conosce:

- come i componenti siano stati costruiti;
- da quale configurazione provengano;
- se siano stati caricati da codice, YAML o UI;
- quale significato di business abbiano i dati prodotti.

La classe si occupa solo di:

- eseguire gli step nell'ordine corretto;
- iniettare l'`EventBus` nei componenti che implementano `IEventEmitter`;
- creare l'oggetto finale `PipelineOutputs`;
- avvolgere gli errori di stage in `PipelineExecutionError`.

### Il ruolo di `PipelineOutputs`

L'output finale di una run è un `PipelineOutputs`, che contiene:

- `results`: i risultati analitici degli analyzer;
- `artifacts`: gli artefatti visuali generati dai visualizer;
- `metadata`: informazioni di esecuzione, inclusi `pipeline_id`, timestamp e metadati runtime.

Questo rende il core adatto sia a script Python sia a UI o servizi che vogliano persistere o visualizzare i risultati.

## Esecuzione: sync, async ed eventi

### Facciata pubblica: `PipelineOrchestrator`

L'interfaccia applicativa consigliata è `PipelineOrchestrator`. Espone:

- `run(context, ...)` per esecuzione sincrona;
- `submit(context, ...)` per esecuzione asincrona;
- `terminate(pipeline_id)` per cancellazione best-effort;
- `active_ids()` per ispezionare le pipeline attive;
- `shutdown()` per chiudere il runner sottostante.

### Esecuzione asincrona: `ThreadedPipelineRunner`

L'implementazione di default del runner è `ThreadedPipelineRunner`, basata su `ThreadPoolExecutor`. Gestisce:

- deduplicazione degli `id` attivi;
- snapshot di stato tramite `IPipelineMonitor`;
- retry policy configurabili;
- lifecycle events della pipeline;
- output store opzionale.

Gli stati osservabili dal monitor sono, di fatto:

- `QUEUED`;
- `RUNNING`;
- `SUCCEEDED`;
- `FAILED`;
- `CANCELLED`.

I lifecycle event emessi sono:

- `pipeline.before_run`;
- `pipeline.after_run`;
- `pipeline.error`;
- `pipeline.retry`;
- `pipeline.cancelled`;
- `pipeline.rejected`;
- `pipeline.submit_failed`.

### Event bus e branching

Il sistema supporta componenti che emettono eventi di dominio tramite `IEventEmitter`. L'implementazione base è `EventBus`, thread-safe e sincrona nel dispatch.

Su questa base si innesta `BranchingCoordinator`, che:

- ascolta gli eventi di dominio;
- valuta una o più `IBranchingRule`;
- costruisce nuove `PipelineContext` secondarie;
- dispatcha un `PipelineEvent` che l'orchestrator usa per sottomettere nuove pipeline.

Un esempio concreto già presente è `NewTrackBranchingRule`, che reagisce all'evento `track_created` generato dal tracker multi-oggetto e avvia una pipeline secondaria focalizzata sul nuovo seed track.

## Il sistema di plugin

### `PluginRegistry`

`PluginRegistry` è il catalogo centrale delle implementazioni disponibili. Ogni plugin è registrato tramite:

- categoria;
- nome;
- factory;
- descrizione.

Le categorie canoniche sono:

- `frame_extractor`
- `single_frame_processor`
- `signal_extractor`
- `signal_cleaner`
- `analyzer`
- `visualizer`
- `branching_rule`

### Due livelli di registry

Nel repository esistono due punti di ingresso principali:

- `create_builtin_registry()` in `library/core/plugins/PluginRegistry.py`
  registra un set minimo e stabile di componenti built-in;
- `ui/services/registry_bootstrap.py`
  costruisce il registry esteso usato da `SEF Studio`, includendo più processor, cleaner di segnale, analyzer, visualizer e signal extractor.

### Perché è importante

Questo disaccoppiamento consente di:

- definire pipeline via nome e parametri;
- collegare facilmente una UI a un catalogo di componenti;
- aggiungere nuovi moduli senza toccare il motore di esecuzione;
- testare builder e registry separatamente.

## Builder disponibili

### `FluentPipelineBuilder`

È il builder programmatico. Serve quando la pipeline viene definita direttamente in Python e permette di comporre il contesto passo dopo passo.

### `ConfigPipelineBuilder`

È il builder dichiarativo. Serve quando la pipeline nasce da una configurazione esterna, ad esempio JSON, YAML o da un editor UI. Il builder:

- legge la configurazione;
- valida struttura minima e tipi attesi;
- usa il `PluginRegistry` per istanziare i componenti;
- restituisce un `PipelineContext`.

Un esempio di configurazione reale è presente in [pipeline.yaml](/Users/matteo/Documents/UNICAM/3°anno/STAGE/SEF/pipeline.yaml).

## Componenti concreti presenti nel progetto

Di seguito i gruppi principali già implementati nel repository.

### Frame extractors

- `OpenCVBufferedFrameExtractor`
  legge un video con OpenCV e produce un `FrameBuffer`, con supporto a `resize`, `stride` e `max_frames`.

### Frame processors

- `OpenCVGrayFrameProcessor`
- `SmoothingFrameProcessor`
- `OpenCVResizeFrameProcessor`
- `OpenCVBackgroundSubtractionFrameProcessor`
- `OpenCVHistogramEqualizationFrameProcessor`

Questi moduli servono a normalizzare o evidenziare il segnale utile prima dell'estrazione.

### Signal extractors

- `OpenCVBufferedSignalExtractor`
  tracking singolo oggetto a partire da una ROI iniziale;
- `OpenCVMultiObjectSignalExtractor`
  tracking multi-oggetto con seed ROI, template expansion e eventi `track_created` / `track_lost`;
- `OpenCVDenseOpticalFlowSignalExtractor`
  estrazione di optical flow denso.

### Signal cleaners

- `MovingAverageCleaner`
- `OutlierRejectionCleaner`
- `SignalWidenerCleaner`
- `OpticalFlowOutlierCleaner`

### Analyzers

- posizione verticale e orizzontale;
- velocità verticale e orizzontale;
- frequenza verticale e orizzontale;
- conteggio attraversamenti barriere (`MultiObjectBarrierCountingAnalyzer`);
- trasformazione del tracking in playback video-ready (`TrackingPlaybackAnalyzer`);
- conversione del dense optical flow in campo vettoriale (`DenseOpticalFlowVectorFieldAnalyzer`).

### Visualizers

- grafici Matplotlib per funzioni, istogrammi, traiettorie, heatmap, vector field;
- `TrackingVideoVisualizer` per produrre un video annotato con bounding box e centroidi.

### Retry policies

- `NoRetryPolicy`
- `FixedRetryPolicy`
- `ExponentialBackoffRetryPolicy`

## Artefatti dati e visualizzazione

Il progetto distingue bene tra dati analitici e presentazione:

- `library/core/artifacts/*` contiene i modelli dati di passaggio tra gli step;
- `library/core/visualization/*` contiene output e artefatti pronti alla UI.

Tra gli oggetti più importanti:

- `Frame`, `FrameBuffer`
- `Signal`, `BoxSignalSample`, `MultiObjectSignalSample`, `DenseOpticalFlowSignalSample`
- `TwoDimGraphData`, `VectorFieldGraphData`, `TrajectoryData`, `CategoryData`, `TrackingPlaybackData`
- `ImageArtifact`, `VideoArtifact`, `TableArtifact`, `JsonArtifact`, `TextArtifact`

Questa separazione evita che la UI dipenda dalla struttura interna degli analyzer.

## SEF Studio

`SEF Studio` è la superficie applicativa del progetto, avviabile con Streamlit. La UI:

- carica il registry condiviso;
- offre preset per tracking singolo, multi-oggetto e dense optical flow;
- permette di selezionare ROI, geometrie e barriere;
- compone la pipeline via canvas interattivo;
- mostra editor JSON della configurazione;
- avvia run sincrone o asincrone;
- visualizza snapshot, eventi, output e artefatti.

In altre parole, `ui/` non replica la logica del core: la orchestra e la rende ispezionabile.

## Installazione

### Requisiti

- Python `>= 3.11`
- OpenCV contrib
- NumPy
- Matplotlib
- Streamlit

### Installazione minima della libreria

```bash
pip install -e .
```

### Installazione completa per libreria + UI

```bash
pip install -r requirements.txt
```

## Avvio

### Avviare la UI

```bash
streamlit run ui/app.py
```

### Eseguire i test

```bash
python -m unittest discover -s tests -v
```

## Esempi d'uso

### 1. Pipeline programmatica con builder fluente

```python
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.frame_processors.OpenCVGrayFrameProcessor import OpenCVGrayFrameProcessor
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor

context = (
    FluentPipelineBuilder()
    .with_frame_extractor(
        OpenCVBufferedFrameExtractor(
            path="videos/Baloons.mp4",
            config={"stride": 1, "max_frames": 120},
        )
    )
    .add_frame_processor(SingleFrameProcessorAdapter(OpenCVGrayFrameProcessor()))
    .with_signal_extractor(
        OpenCVBufferedSignalExtractor(
            tracker_type="CSRT",
            start_box=(100, 200, 50, 80),
            config={"show": False},
        )
    )
    .add_signal_cleaner(MovingAverageCleaner(window_size=5))
    .add_analyzer(VerticalPositionAnalyzer())
    .build_context()
)

outputs = PipelineOrchestrator().run(context)
series = outputs.results[0]

print(series.title)
print(series.x[:5])
print(series.y[:5])
```

Quando usare questo approccio:

- script Python;
- test;
- notebook;
- servizi backend che costruiscono la pipeline via codice.

### 2. Pipeline dichiarativa da configurazione

```python
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.plugins.PluginRegistry import create_builtin_registry

registry = create_builtin_registry()

config = {
    "pipeline": {
        "frame_extractor": {
            "name": "opencv_buffered",
            "params": {"path": "videos/Baloons.mp4"},
        },
        "frame_processors": [
            {"name": "opencv_gray"},
        ],
        "signal_extractor": {
            "name": "opencv_tracker",
            "params": {"start_box": [100, 200, 50, 80]},
        },
        "signal_cleaners": [
            {"name": "moving_average", "params": {"window_size": 5}},
        ],
        "analyzers": [
            {"name": "vertical_position"},
        ],
        "visualizers": [
            {"name": "matplotlib"},
        ],
    }
}

context = ConfigPipelineBuilder(registry).build_context(config)
outputs = PipelineOrchestrator().run(context)
```

Quando usare questo approccio:

- editor visuale;
- JSON/YAML configurabili;
- integrazione con strumenti no-code o configuratori.

### 3. Esecuzione asincrona con monitoraggio

```python
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner

monitor = InMemoryPipelineMonitor()
runner = ThreadedPipelineRunner(monitor=monitor)
orchestrator = PipelineOrchestrator(runner=runner)

pipeline_id = orchestrator.submit(context)

print("Active ids:", orchestrator.active_ids())
print("Snapshot:", runner.snapshot(pipeline_id))
```

Quando usare questo approccio:

- UI interattive;
- code di esecuzione;
- flussi in background;
- orchestrazione concorrente.

### 4. Branching event-driven

Schema concettuale del flusso:

1. una pipeline primaria esegue un `OpenCVMultiObjectSignalExtractor`;
2. l'estrattore emette `track_created` quando nasce un seed track;
3. `BranchingCoordinator` intercetta l'evento;
4. una `IBranchingRule` costruisce un nuovo `PipelineContext`;
5. `PipelineOrchestrator` sottomette la pipeline secondaria.

Questo approccio è utile quando l'analisi deve ramificarsi dinamicamente a partire da eventi osservati durante l'esecuzione.

## Casi d'uso supportati oggi

- tracking di un singolo oggetto a partire da ROI iniziale;
- tracking multi-oggetto con seed manuale e rilevazione di oggetti simili;
- analisi di traiettorie verticali e orizzontali;
- estrazione di velocità e frequenze del movimento;
- conteggio di attraversamenti rispetto a barriere geometriche;
- generazione di playback video annotato;
- ispezione visuale di dense optical flow;
- prototipazione rapida di pipeline CV tramite interfaccia Streamlit.

## Cosa rende il progetto interessante

Dal punto di vista ingegneristico, il valore del progetto non sta solo nei singoli algoritmi, ma nella combinazione di alcune scelte solide:

- il core è disaccoppiato dagli algoritmi concreti;
- il contesto di pipeline è immutabile e validato;
- builder, orchestrator e runner hanno responsabilità nettamente separate;
- il sistema di eventi rende possibile il branching senza accoppiare i componenti;
- la UI riusa il core invece di duplicarne la logica;
- i test coprono i punti architetturalmente più critici: pipeline, branching, registry, builder, event bus.

## Possibilità di sviluppo future

Le direzioni più coerenti con l'architettura attuale sono:

- introdurre uno schema di configurazione tipizzato e versionato;
- unificare o rendere autodiscoverable il sistema di plugin;
- aggiungere output store persistenti per risultati e artefatti;
- esporre meglio i contratti pubblici come API documentate;
- introdurre metriche, logging strutturato e telemetria di pipeline;
- parallelizzare selettivamente analyzer o visualizer indipendenti;
- supportare bus/event sink esterni oltre all'`EventBus` in-memory;
- costruire una CLI ufficiale per esecuzione batch da config;
- rafforzare i casi d'uso multi-pipeline e i workflow di branching.

## A chi serve SEF

SEF è adatto quando serve un'infrastruttura componibile per computer vision sperimentale o applicativa, in particolare se si vuole:

- cambiare facilmente estrattori, analyzer e visualizer;
- mantenere separati motore, configurazione e presentazione;
- passare dallo scripting alla UI senza riscrivere la pipeline;
- evolvere il sistema verso scenari event-driven o più orchestrati.

## Riferimenti rapidi

- entry point UI: [ui/app.py](/Users/matteo/Documents/UNICAM/3°anno/STAGE/SEF/ui/app.py)
- motore di esecuzione: [library/core/pipeline/Pipeline.py](/Users/matteo/Documents/UNICAM/3°anno/STAGE/SEF/library/core/pipeline/Pipeline.py)
- facciata applicativa: [library/core/pipeline/PipelineOrchestrator.py](/Users/matteo/Documents/UNICAM/3°anno/STAGE/SEF/library/core/pipeline/PipelineOrchestrator.py)
- builder dichiarativo: [library/core/pipeline/ConfigPipelineBuilder.py](/Users/matteo/Documents/UNICAM/3°anno/STAGE/SEF/library/core/pipeline/ConfigPipelineBuilder.py)
- builder fluente: [library/core/pipeline/FluentPipelineBuilder.py](/Users/matteo/Documents/UNICAM/3°anno/STAGE/SEF/library/core/pipeline/FluentPipelineBuilder.py)
- registry: [library/core/plugins/PluginRegistry.py](/Users/matteo/Documents/UNICAM/3°anno/STAGE/SEF/library/core/plugins/PluginRegistry.py)
- bootstrap UI del registry: [ui/services/registry_bootstrap.py](/Users/matteo/Documents/UNICAM/3°anno/STAGE/SEF/ui/services/registry_bootstrap.py)
