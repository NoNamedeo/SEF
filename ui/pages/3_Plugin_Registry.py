"""
Plugin Registry — visualizza e gestisci i plugin SEF registrati.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.services.registry_bootstrap import get_registry          # noqa: E402
from library.core.plugins.PluginRegistry import PluginCategory   # noqa: E402

st.set_page_config(page_title="Plugin Registry — SEF", layout="wide", page_icon="🔌")
st.title("🔌 Plugin Registry")
st.caption("Visualizza i plugin disponibili per categoria e registra plugin personalizzati.")

registry = get_registry()

# ── Summary ───────────────────────────────────────────────────────────────────
all_plugins = registry.list()
categories  = list(PluginCategory)

st.subheader("Riepilogo")
cols = st.columns(len(categories))
for col, cat in zip(cols, categories):
    count = len(registry.list(cat))
    label = cat.value.replace("_", " ").title()
    col.metric(label, count)

st.divider()

# ── Per-category tables ───────────────────────────────────────────────────────
st.subheader("Plugin per categoria")

category_filter = st.selectbox(
    "Filtra per categoria",
    ["Tutte"] + [c.value for c in categories],
    key="pr_cat_filter",
)

shown_categories = categories if category_filter == "Tutte" else [PluginCategory(category_filter)]

for cat in shown_categories:
    plugins = registry.list(cat)
    label   = cat.value.replace("_", " ").title()

    with st.expander(f"**{label}** — {len(plugins)} plugin", expanded=True):
        if not plugins:
            st.info("Nessun plugin registrato in questa categoria.")
            continue
        rows = [
            {"Nome": p.name, "Descrizione": p.description, "Factory": p.factory.__name__}
            for p in sorted(plugins, key=lambda p: p.name)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()

# ── Runtime registration ──────────────────────────────────────────────────────
st.subheader("Registra plugin personalizzato")
st.caption(
    "Aggiungi un plugin a runtime inserendo il percorso completo della classe Python. "
    "Il plugin sarà disponibile fino al riavvio del server Streamlit."
)

with st.form("register_plugin_form"):
    r_cat   = st.selectbox("Categoria", [c.value for c in categories], key="rp_cat")
    r_name  = st.text_input("Nome univoco (es. `my_tracker`)", key="rp_name")
    r_cls   = st.text_input(
        "Classe Python (percorso completo, es. `mymodule.MyTracker`)", key="rp_cls"
    )
    r_desc  = st.text_input("Descrizione (opzionale)", key="rp_desc")
    submitted = st.form_submit_button("Registra")

if submitted:
    if not r_name:
        st.error("Il nome è obbligatorio.")
    elif not r_cls:
        st.error("Il percorso della classe è obbligatorio.")
    else:
        # Attempt dynamic import
        try:
            parts     = r_cls.rsplit(".", 1)
            mod_path  = parts[0]
            cls_name  = parts[1] if len(parts) == 2 else parts[0]
            import importlib
            mod    = importlib.import_module(mod_path)
            factory = getattr(mod, cls_name)
            registry.register(PluginCategory(r_cat), r_name, factory, r_desc)
            st.success(f"Plugin `{r_name}` registrato con successo nella categoria `{r_cat}`.")
            st.rerun()
        except ValueError as exc:
            st.error(f"Plugin già registrato o nome non valido: {exc}")
        except (ImportError, AttributeError) as exc:
            st.error(f"Impossibile importare `{r_cls}`: {exc}")
        except Exception as exc:
            st.error(f"Errore: {exc}")
