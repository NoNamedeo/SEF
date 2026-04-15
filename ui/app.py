"""
SEF — Signal Extraction Framework
Home page / entry point.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from library.core.plugins.PluginRegistry import PluginCategory  # noqa: E402
from ui.services.registry_bootstrap import get_registry  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEF — Signal Extraction Framework",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap registry (cached) ───────────────────────────────────────────────
registry = get_registry()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 SEF — Signal Extraction Framework")
st.caption("Piattaforma modulare per l'estrazione, l'analisi e la visualizzazione di segnali da video tramite tracking di oggetti.")
st.divider()

# ── Quick stats ───────────────────────────────────────────────────────────────
all_plugins = registry.list()
categories = list(PluginCategory)
cat_counts = {c: len(registry.list(c)) for c in categories}

st.subheader("Stato del sistema")
cols = st.columns(len(categories))
for col, cat in zip(cols, categories):
    col.metric(cat.value.replace("_", " ").title(), cat_counts[cat])

st.divider()

# ── Navigation cards ──────────────────────────────────────────────────────────
st.subheader("Pagine disponibili")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### Pipeline Runner")
    st.markdown(
        "Esegui una pipeline completa: seleziona il video, "
        "disegna la ROI e le barriere direttamente sull'immagine, "
        "configura i parametri e analizza i risultati."
    )
    st.page_link("pages/1_Pipeline_Runner.py", label="Apri →", icon="🚀")

with c2:
    st.markdown("### Config Builder")
    st.markdown("Costruisci e valida una pipeline tramite dizionario di configurazione. Ideale per scenari di deployment e test ripetibili.")
    st.page_link("pages/2_Config_Builder.py", label="Apri →", icon="⚙️")

with c3:
    st.markdown("### 🔌 Plugin Registry")
    st.markdown("Visualizza tutti i plugin registrati per categoria, controlla lo stato di caricamento e aggiungi plugin personalizzati a runtime.")
    st.page_link("pages/3_Plugin_Registry.py", label="Apri →", icon="🔌")

with c4:
    st.markdown("### Pipeline Monitor")
    st.markdown("Gestisci pipeline in esecuzione asincrona con il ThreadedPipelineRunner. Avvia, monitora e cancella pipeline in background.")
    st.page_link("pages/4_Pipeline_Monitor.py", label="Apri →", icon="📊")

st.divider()

# ── Architecture overview ─────────────────────────────────────────────────────
with st.expander("Architettura del sistema", expanded=False):
    st.markdown("""
```
Video
  └─▶ FrameExtractor  (OpenCVBufferedFrameExtractor)
        └─▶ [FrameCleaners]  (Gray, Smoothing, …)
              └─▶ FrameBuffer
                    └─▶ SignalExtractor  (SingleObject / MultiObject)
                          └─▶ [SignalCleaners]  (MovingAverage, OutlierRejection, …)
                                └─▶ Signal
                                      └─▶ [Analyzers]  →  list[IData]
                                            └─▶ [Visualizers]
```

**Modalità di esecuzione disponibili oggi**
- `PipelineOrchestrator.run(context)` — sincrono, senza EventBus obbligatorio
- `PipelineOrchestrator.submit(context, id)` — asincrono tramite runner e monitor

**Integrazione event-driven opzionale**
- `EventBus` + `PipelineEvent` — trigger esterni
- `BranchingCoordinator` — pipeline secondarie basate su eventi di dominio
    """)

# ── Plugin detail table ───────────────────────────────────────────────────────
with st.expander("Tutti i plugin caricati", expanded=False):
    if all_plugins:
        rows = [
            {"Categoria": p.category, "Nome": p.name, "Descrizione": p.description} for p in sorted(all_plugins, key=lambda p: (p.category, p.name))
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("Nessun plugin caricato.")
