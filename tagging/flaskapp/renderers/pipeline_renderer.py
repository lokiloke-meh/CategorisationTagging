from __future__ import annotations

import sys
from pathlib import Path

# Ensure utils directory is on sys.path for visualise_pipeline_html import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_UTILS_DIR = _PROJECT_ROOT / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import visualise_pipeline_html as vph  # noqa: E402


def render_pipeline_html(pipeline_id: str, registry_source: str | Path) -> str:
    """Renders the standalone interactive pipeline HTML dashboard for a given pipeline_id.

    Mirrors the exact visualization from utils/visualise_pipeline_html.py.
    """
    registry_path = Path(registry_source)
    if not registry_path.exists():
        return f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #f9fafb; color: #6b7280;">
            <div style="text-align: center; max-width: 500px; padding: 24px;">
                <h3 style="color: #374151;">Pipeline Registry Not Found</h3>
                <p>Expected at: <code>{registry_path}</code></p>
            </div>
        </body>
        </html>
        """

    registry_data = vph.load_pipeline_registry(registry_path)
    if not registry_data:
        return """
        <!DOCTYPE html>
        <html>
        <body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #f9fafb; color: #6b7280;">
            <div style="text-align: center;">
                <h3>Empty Pipeline Registry</h3>
                <p>No pipeline definitions found in the registry.</p>
            </div>
        </body>
        </html>
        """

    # If the requested pipeline_id is not present, pick the first or newest pipeline in registry
    target_id = pipeline_id
    if target_id not in registry_data:
        sorted_ids = sorted(
            [k for k, v in registry_data.items() if isinstance(v, dict)],
            key=lambda k: registry_data[k].get("last_used", ""),
            reverse=True,
        )
        if sorted_ids:
            target_id = sorted_ids[0]
        else:
            target_id = next(iter(registry_data.keys()))

    # Build IR for all pipelines in the registry
    pipelines_ir: dict[str, vph.PipelineGraphIR] = {}
    for p_id, p_entry in registry_data.items():
        if isinstance(p_entry, dict):
            try:
                engine = vph.ChainGraphInferenceEngine(p_id, p_entry)
                pipelines_ir[p_id] = engine.infer()
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Failed to infer IR for pipeline {p_id}: {e}", file=sys.stderr)

    if not pipelines_ir:
        return f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #f9fafb; color: #6b7280;">
            <div style="text-align: center;">
                <h3>Unable to Infer Pipeline Graph</h3>
                <p>No valid pipeline graphs could be parsed for ID: <code>{pipeline_id}</code></p>
            </div>
        </body>
        </html>
        """

    return vph.render_html_dashboard(pipelines_ir, initial_pipeline_id=target_id)
