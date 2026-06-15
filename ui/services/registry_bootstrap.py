"""Registry bootstrap for SEF Studio.

The UI deliberately uses the same public registry factory exposed by SEF.
Keeping a second, hand-written list here caused stale plugin names to appear
in the composer after builtin components were refactored into ``sef.builtin``.
"""

from __future__ import annotations

import logging

import streamlit as st

from sef.api import default_registry
from sef.core.plugins.PluginRegistry import PluginRegistry

log = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Caricamento registry SEF...")
def get_registry() -> PluginRegistry:
    """Return the shared SEF registry used by the Streamlit server process."""
    registry = default_registry(include_builtins=True)
    log.info("SEF Studio registry loaded with %s plugin definitions.", len(registry.list()))
    return registry
