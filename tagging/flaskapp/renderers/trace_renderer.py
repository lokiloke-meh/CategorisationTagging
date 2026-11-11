from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.visualize_trace_html import (
    render_trace_html as _render_trace_html,  # noqa: E402
)


def render_trace_html(trace_id: str, log_dir: str) -> str:
    """Render trace Gantt HTML for iframe embedding (Flask route)."""
    return _render_trace_html(trace_id, log_dir)
