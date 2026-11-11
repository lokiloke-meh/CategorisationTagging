#!/usr/bin/env python3
"""Pipeline HTML Visualizer.

Generates a standalone, interactive HTML dashboard visualizing document tagging
pipelines stored in pipeline_registry.json.

Designed as a deep module: encapsulates JSON sanitization (Infinity/NaN handling),
structural DAG inference (accommodating any past or future chain architecture
without code changes), Mermaid diagram generation, and a reactive inspector UI.

Usage:
    python utils/visualise_pipeline_html.py <pipeline_id> --input data_store/nseix_circulars/pipeline_registry.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ==============================================================================
# 1. Structural Graph Intermediate Representation (IR)
# ==============================================================================


@dataclass
class NodeIR:
    """Intermediate representation of a chain/component node."""

    id: str
    name: str
    type: str
    category: str  # 'router', 'summarizer', 'distiller', 'pass_through', 'composite', 'generic'
    properties: dict[str, Any] = field(default_factory=dict)
    raw_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeIR:
    """Intermediate representation of a connection between two nodes."""

    source_id: str
    target_id: str
    label: str = ""
    style: str = "solid"


@dataclass
class PipelineGraphIR:
    """Complete graph IR for a single pipeline identity."""

    pipeline_id: str
    class_name: str
    file_count: int
    first_used: str
    last_used: str
    nodes: list[NodeIR] = field(default_factory=list)
    edges: list[EdgeIR] = field(default_factory=list)
    root_node_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 2. JSON & Float Sanitization
# ==============================================================================


def sanitize_json_values(obj: Any) -> Any:
    """Recursively converts non-standard float values (inf, -inf, nan) to JSON-safe representations."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_json_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_values(item) for item in obj]
    return obj


def load_pipeline_registry(source: str | Path | dict) -> dict[str, Any]:
    """Loads and sanitizes pipeline registry data from a filepath or dictionary."""
    if isinstance(source, dict):
        return sanitize_json_values(source)

    file_path = Path(source)
    if not file_path.exists():
        raise FileNotFoundError(f"Pipeline registry file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        # Standard Python json.load parses Infinity by default as float('inf')
        raw_data = json.load(f)

    return sanitize_json_values(raw_data)


# ==============================================================================
# 3. Structural Graph Inference Engine (Zero-Config for New Chains)
# ==============================================================================


class ChainGraphInferenceEngine:
    """Infers DAG structure from arbitrary serialized chain dictionaries.

    Uses structural discovery instead of class-specific branching:
    - Lists of chains -> Sequential or fan-out connections
    - Keyed dictionaries -> Named branch edges (e.g. router destination maps)
    - Route objects -> Conditional edges with formatted conditions
    - Fallback chains -> 'Default' / 'Fallback' edges
    - Any unrecognized chain -> Discovered dynamically and linked
    """

    # Keys that define graph linkages (excluded from node's displayed property table)
    CONTROL_LINKAGE_KEYS = {  # noqa: RUF012
        "routes",
        "chains",
        "steps",
        "stages",
        "default_chain",
        "fallback_chain",
        "destination_map",
        "branches",
        "mapping",
        "destination_chain",
    }

    # Structural keys that wrap children without being graph nodes themselves
    CONTAINER_KEYS = {"chain_graph", "pipeline", "graph"}

    # Keys that signal a nested adapter/chain worth visiting (no type/name required)
    ADAPTER_KEYS = {
        "distiller",
        "categorisation_chain",
        "summarisation_chain",
        "routing_chain",
    }

    # Keys treated as pass-through leaf indicators (visited but no children)
    LEAF_INDICATOR_KEYS = {"target_tags", "_target_tags"}

    def __init__(self, pipeline_id: str, reg_entry: dict[str, Any]):
        self.pipeline_id = pipeline_id
        self.reg_entry = reg_entry
        self.nodes: list[NodeIR] = []
        self.edges: list[EdgeIR] = []
        self._node_counter = 0
        self.metadata: dict[str, Any] = {}

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"node_{self._node_counter}"

    def _classify_category(
        self, type_str: str, name_str: str, node_dict: dict[str, Any]
    ) -> str:
        lower = f"{type_str} {name_str}".lower()
        if (
            "router" in lower
            or "route" in lower
            or "switch" in lower
            or "branch" in lower
        ):
            return "router"
        if (
            "summariz" in lower
            or "splitter" in lower
            or "refine" in lower
            or "mapreduce" in lower
        ):
            return "summarizer"
        if (
            "distill" in lower
            or "tag" in lower
            or "categor" in lower
            or "extract" in lower
        ):
            return "distiller"
        if "noop" in lower or "passthrough" in lower or "_target_tags" in node_dict:
            return "pass_through"
        if "sequential" in lower or "pipeline" in lower or "composite" in lower:
            return "composite"
        return "generic"

    def _format_condition(self, condition: Any, route_name: str = "") -> str:
        """Formats arbitrary route condition into a human-readable edge label."""
        if not condition:
            return route_name or "Match"

        if isinstance(condition, dict):
            # Text length condition
            min_c = condition.get("min_chars")
            max_c = condition.get("max_chars")
            if min_c is not None and max_c is not None:
                if max_c == "Infinity" or (
                    isinstance(max_c, (int, float)) and math.isinf(max_c)
                ):
                    return f"{route_name or 'Length'} (>= {min_c:,} chars)"
                return f"{route_name or 'Length'} ({min_c:,} - {max_c:,} chars)"
            if min_c is not None:
                return f"{route_name or 'Length'} (>= {min_c:,} chars)"
            if max_c is not None:
                return f"{route_name or 'Length'} (<= {max_c:,} chars)"

            # Buckets condition
            buckets = condition.get("_buckets") or condition.get("buckets")
            if buckets and isinstance(buckets, list):
                return f"{route_name or 'Classify'} ({len(buckets)} buckets)"

        if isinstance(condition, str):
            return condition

        return route_name or "Condition"

    def _is_chain_like(self, obj: Any) -> bool:
        """Checks if a dictionary represents an execution chain node.

        Recognises both explicit nodes (type/name), adapter containers
        that hold nested chains worth visiting, and leaf indicators.
        """
        if not isinstance(obj, dict):
            return False
        if any(k in obj for k in (
            "type", "name",
            "destination_map", "chains", "routes",
        )):
            return True
        if self.ADAPTER_KEYS & obj.keys():
            return True
        # Pure leaf dicts like {"target_tags": [...]} are chain-like pass-throughs
        if not any(k in obj for k in (
            "eval_max_retries", "evaluators", "invoke_max_retries",
            "include_parent_eval", "llm", "_llm", "condition",
        )):
            return bool(self.LEAF_INDICATOR_KEYS & obj.keys())
        return False

    def infer(self) -> PipelineGraphIR:
        chain_graph = self.reg_entry.get("chain_graph") or {}

        # Extract pipeline metadata (settings/config, not graph nodes)
        if isinstance(chain_graph, dict):
            for key in ("pipeline", "graph"):
                meta = chain_graph.get(key)
                if isinstance(meta, dict) and (
                    "type" in meta or "name" in meta
                ):
                    self.metadata = {key: meta}
                    break

        root_id = ""
        if isinstance(chain_graph, dict) and chain_graph:
            # Find the actual graph root by descending into container wrappers
            root_val = self._find_root_value(chain_graph)
            if root_val is not None:
                root_id = self._visit_node(root_val, parent_edge_label="")

        return PipelineGraphIR(
            pipeline_id=self.pipeline_id,
            class_name=str(self.reg_entry.get("class_name", "TagResult")),
            file_count=int(self.reg_entry.get("file_count", 0)),
            first_used=str(self.reg_entry.get("first_used", "N/A")),
            last_used=str(self.reg_entry.get("last_used", "N/A")),
            nodes=self.nodes,
            edges=self.edges,
            root_node_id=root_id,
            metadata=self.metadata,
        )

    def _find_root_value(self, obj: dict[str, Any]) -> dict[str, Any] | None:
        """Descends into container wrappers to find the first actual chain node.

        If the current object IS a concrete node (has type/name), returns it.
        Otherwise descends into structural containers until finding one.
        """
        if not isinstance(obj, dict):
            return None

        # If this is already a concrete node, use it directly
        if obj.get("type") or obj.get("name"):
            return obj

        # Otherwise descend into container wrappers (skip pipeline = metadata)
        for key in self.CONTAINER_KEYS:
            val = obj.get(key)
            if key == "pipeline":
                continue  # metadata only
            if isinstance(val, dict) and val:
                return self._find_root_value(val)

        # Fallback: first chain-like direct child
        for val in obj.values():
            if self._is_chain_like(val):
                return val

        return None

    def _create_pass_through(self, node_dict: dict[str, Any]) -> str:
        """Creates a pass-through leaf node for dicts with only metadata keys."""
        target_tags = node_dict.get("target_tags") or node_dict.get("_target_tags", [])
        node_name = "TargetTagsLeaf" if target_tags else f"Leaf {self._node_counter + 1}"

        self.nodes.append(
            NodeIR(
                id=self._next_node_id(),
                name=node_name,
                type="PassThrough",
                category="pass_through",
                properties={"target_tags": target_tags},
                raw_dict=node_dict,
            )
        )
        return self.nodes[-1].id

    def _visit_node(
        self, node_dict: dict[str, Any], parent_edge_label: str = ""
    ) -> str:
        """Visits a dictionary, registers a NodeIR, and recursively resolves edges to children."""

        # Unwrap pure adapter containers (no type/name, only ADAPTER/LEAF keys)
        if not node_dict.get("type") and not node_dict.get("name"):
            has_adapter = bool(self.ADAPTER_KEYS & node_dict.keys())
            is_leaf = (
                self.LEAF_INDICATOR_KEYS & node_dict.keys()
                and not any(k in node_dict for k in ("type", "name"))
            )
            if has_adapter or is_leaf:
                skip_keys = (
                    self.CONTAINER_KEYS
                    | {"eval_max_retries", "evaluators", "invoke_max_retries",
                       "include_parent_eval", "llm", "_llm", "condition",
                       "name", "category_tag_config", "decision_model",
                       "min_chars_for_summarization", "prompt_samples",
                       "text_chunk_size", "target_tag_examples"}
                )
                child_id = ""
                for key, val in node_dict.items():
                    if key in skip_keys:
                        continue
                    if self._is_chain_like(val):
                        child_id = self._visit_node(
                            val, parent_edge_label=parent_edge_label or key
                        )
                # If no children found (pure leaf), create a pass-through node
                if not child_id:
                    return self._create_pass_through(node_dict)
                return child_id

        node_id = self._next_node_id()

        # Deduce node identification
        raw_type = node_dict.get("type")
        raw_name = node_dict.get("name")

        if not raw_type and not raw_name:
            if "_target_tags" in node_dict:
                raw_type = "NoOpPassThrough"
                raw_name = "TargetTagsLeaf"
            else:
                raw_type = "ChainNode"
                raw_name = f"Node {self._node_counter}"

        node_type = str(raw_type or raw_name)
        node_name = str(raw_name or raw_type)
        category = self._classify_category(node_type, node_name, node_dict)

        # Extract non-linkage properties for node inspector
        properties = {
            k: v for k, v in node_dict.items() if k not in self.CONTROL_LINKAGE_KEYS
        }

        self.nodes.append(
            NodeIR(
                id=node_id,
                name=node_name,
                type=node_type,
                category=category,
                properties=properties,
                raw_dict=node_dict,
            )
        )

        # ----------------------------------------------------------------------
        # Structural Child Discovery (Handles standard and unseen chain types)
        # ----------------------------------------------------------------------

        # Pattern A: Sequential stages (e.g. chains, steps, stages)
        sequential_keys = [
            k
            for k in ("chains", "steps", "stages")
            if k in node_dict and isinstance(node_dict[k], list)
        ]
        for seq_key in sequential_keys:
            prev_id = node_id
            for child in node_dict[seq_key]:
                if self._is_chain_like(child):
                    child_id = self._visit_node(child, parent_edge_label="")
                    self.edges.append(
                        EdgeIR(source_id=prev_id, target_id=child_id)
                    )
                    prev_id = child_id

        # Pattern B: Conditional Routes (e.g. BoolRouter routes list)
        if "routes" in node_dict and isinstance(node_dict["routes"], list):
            for route in node_dict["routes"]:
                if isinstance(route, dict):
                    dest = route.get("destination_chain") or route.get("chain")
                    condition = route.get("condition")
                    route_name = route.get("name", "Route")
                    edge_label = self._format_condition(condition, route_name)
                    if self._is_chain_like(dest):
                        child_id = self._visit_node(dest, parent_edge_label=edge_label)
                        self.edges.append(
                            EdgeIR(
                                source_id=node_id, target_id=child_id, label=edge_label
                            )
                        )

        # Pattern C: Default / Fallback Chain
        for def_key in ("default_chain", "fallback_chain", "fallback"):
            if def_key in node_dict and self._is_chain_like(node_dict[def_key]):
                child_id = self._visit_node(
                    node_dict[def_key], parent_edge_label="Default / Fallback"
                )
                self.edges.append(
                    EdgeIR(
                        source_id=node_id,
                        target_id=child_id,
                        label=(
                            "Default (< 4000 chars)"
                            if "bool" in node_type.lower()
                            else "Default Fallback"
                        ),
                    )
                )

        # Pattern D: Keyed Destination Mappings (e.g. CategoricalRouter destination_map)
        mapping_keys = [
            k
            for k in ("destination_map", "branches", "mapping")
            if k in node_dict and isinstance(node_dict[k], dict)
        ]
        for map_key in mapping_keys:
            for branch_label, branch_target in node_dict[map_key].items():
                if self._is_chain_like(branch_target):
                    child_id = self._visit_node(
                        branch_target, parent_edge_label=str(branch_label)
                    )
                    self.edges.append(
                        EdgeIR(
                            source_id=node_id,
                            target_id=child_id,
                            label=str(branch_label),
                        )
                    )

        # Pattern E: Arbitrary child dictionary with chain-like structure
        handled_keys = (
            self.CONTROL_LINKAGE_KEYS
            | set(sequential_keys)
            | set(mapping_keys)
            | self.CONTAINER_KEYS
        )
        for key, val in node_dict.items():
            if key in handled_keys:
                continue
            if self._is_chain_like(val):
                child_id = self._visit_node(val, parent_edge_label=key)
                self.edges.append(
                    EdgeIR(source_id=node_id, target_id=child_id, label=key)
                )

        return node_id


# ==============================================================================
# 4. Mermaid Diagram Generation
# ==============================================================================


def generate_mermaid_code(graph: PipelineGraphIR) -> str:
    """Converts PipelineGraphIR into Mermaid flowchart code with interactive click bindings."""
    if not graph.nodes:
        return "graph TD;\n    Empty[No Pipeline Graph Found];"

    lines = ["graph TD;"]

    # Category styling definitions
    class_def = """
    classDef router fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#9a3412,font-weight:bold;
    classDef summarizer fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af,font-weight:bold;
    classDef distiller fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534,font-weight:bold;
    classDef pass_through fill:#f8fafc,stroke:#64748b,stroke-width:2px,color:#334155,font-style:italic;
    classDef composite fill:#fdf4ff,stroke:#c026d3,stroke-width:2px,color:#86198f,font-weight:bold;
    classDef generic fill:#f9fafb,stroke:#4b5563,stroke-width:1.5px,color:#1f2937;
    """

    # Node definitions
    for node in graph.nodes:
        clean_name = re.sub(r'["\n\r]', " ", node.name).strip()
        clean_type = re.sub(r'["\n\r]', " ", node.type).strip()

        # Display format
        if clean_name and clean_type and clean_name != clean_type:
            label = f"<b>{clean_name}</b><br/><small style='color:#6b7280;'>({clean_type})</small>"
        else:
            label = f"<b>{clean_name or clean_type}</b>"

        # Node shape by category
        if node.category == "router":
            shape = f'{node.id}{{"{label}"}}'
        elif node.category == "pass_through":
            shape = f'{node.id}(["{label}"])'
        elif node.category == "composite":
            shape = f'{node.id}[["{label}"]]'
        else:
            shape = f'{node.id}["{label}"]'

        lines.append(f"    {shape}:::{node.category}")

    # Edge definitions
    for edge in graph.edges:
        clean_label = re.sub(r'["\n\r]', " ", edge.label).strip()
        if clean_label:
            clean_label = clean_label.replace("|", "/")
            lines.append(f'    {edge.source_id} -->|"{clean_label}"| {edge.target_id}')
        else:
            lines.append(f"    {edge.source_id} --> {edge.target_id}")

    # Click callbacks for Mermaid interactive support
    for node in graph.nodes:
        lines.append(
            f'    click {node.id} selectNode "Click to inspect node configuration"'
        )

    lines.append(class_def)
    return "\n".join(lines)


# ==============================================================================
# 5. Standalone Interactive HTML Dashboard Template
# ==============================================================================


def render_html_dashboard(
    pipelines: dict[str, PipelineGraphIR],
    initial_pipeline_id: str,
) -> str:
    """Compiles pipeline graphs, metadata, and reactive scripts into standalone HTML."""

    pipelines_data: dict[str, Any] = {}
    for p_id, p_graph in pipelines.items():
        node_map: dict[str, Any] = {}
        for n in p_graph.nodes:
            node_map[n.id] = {
                "id": n.id,
                "name": n.name,
                "type": n.type,
                "category": n.category,
                "properties": n.properties,
                "raw_dict": n.raw_dict,
            }

        pipelines_data[p_id] = {
            "pipeline_id": p_id,
            "class_name": p_graph.class_name,
            "file_count": p_graph.file_count,
            "first_used": p_graph.first_used,
            "last_used": p_graph.last_used,
            "mermaid_code": generate_mermaid_code(p_graph),
            "nodes": node_map,
            "edges": [asdict(e) for e in p_graph.edges],
            "metadata": p_graph.metadata,
        }

    pipelines_json = json.dumps(pipelines_data, indent=2, ensure_ascii=False)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Pipeline Architecture Visualizer</title>
    <!-- Bootstrap Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
    <!-- Mermaid CDN with loose security for click handlers -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <style>
        :root {{
            --bg-canvas: #f8fafc;
            --bg-card: #ffffff;
            --border-color: #e2e8f0;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --accent: #2563eb;
            --accent-light: #eff6ff;
            --radius: 8px;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg-canvas);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        /* Header Navbar */
        header {{
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
            box-shadow: var(--shadow-sm);
            z-index: 10;
        }}
        .brand-title {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .brand-title h1 {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }}
        .badge {{
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 9999px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-blue {{ background: #dbeafe; color: #1e40af; }}
        .badge-green {{ background: #dcfce7; color: #166534; }}

        .pipeline-selector-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .pipeline-selector-group label {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
        }}
        select.pipeline-select {{
            padding: 6px 12px;
            font-size: 0.88rem;
            font-family: monospace;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-card);
            color: var(--text-primary);
            outline: none;
            cursor: pointer;
        }}
        select.pipeline-select:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(37,99,235,0.2);
        }}

        .meta-chips {{
            display: flex;
            gap: 16px;
            align-items: center;
            font-size: 0.82rem;
            color: var(--text-secondary);
        }}
        .meta-chip {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .meta-chip strong {{
            color: var(--text-primary);
        }}

        /* Main Workspace: Left Graph + Right Details */
        main {{
            flex: 1;
            display: flex;
            overflow-x: auto;
            position: relative;
        }}

        /* Graph Canvas Pane */
        #graph-pane {{
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            background: #ffffff;
            border-right: 1px solid var(--border-color);
            overflow: hidden;
        }}
        .graph-toolbar {{
            position: absolute;
            top: 14px;
            left: 14px;
            z-index: 5;
            display: flex;
            gap: 6px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(4px);
            padding: 4px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
        }}
        .toolbar-btn {{
            background: none;
            border: 1px solid transparent;
            padding: 5px 9px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .toolbar-btn:hover {{
            background: #f1f5f9;
            color: var(--text-primary);
        }}

        #mermaid-viewport {{
            flex: 1;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            cursor: grab;
        }}
        #mermaid-viewport:active {{
            cursor: grabbing;
        }}
        #mermaid-viewport svg {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        /* SVG Node Selection Highlighting */
        .node.active-node rect,
        .node.active-node polygon,
        .node.active-node path,
        .node.active-node circle {{
            stroke: #2563eb !important;
            stroke-width: 3.5px !important;
            filter: drop-shadow(0 0 6px rgba(37,99,235,0.45));
        }}

        /* Node Inspector Pane (Right Side) */
        #inspector-pane {{
            width: 460px;
            min-width: 300px;
            max-width: 80vw;
            background: var(--bg-card);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: -2px 0 6px -1px rgba(0,0,0,0.04);
        }}
        #inspector-resize-handle {{
            width: 6px; cursor: col-resize;
            border-left: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            flex-shrink: 0; transition: background 0.2s;
        }}
        #inspector-resize-handle:hover, #inspector-resize-handle.active {{
            background: var(--accent-light);
        }}
        .inspector-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            background: #fbfcfd;
        }}
        .inspector-title {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .inspector-title h2 {{
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .inspector-title .node-type-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: monospace;
        }}

        .inspector-body {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        /* Property Cards */
        .prop-section {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .prop-section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .prop-section-title {{
            font-size: 0.86rem;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .card {{
            background: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 12px 14px;
            font-size: 0.86rem;
        }}
        .card-row {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 6px 0;
            border-bottom: 1px solid #f1f5f9;
            gap: 24px;
        }}
        .card-row:last-child {{ border-bottom: none; }}
        .card-label {{ color: var(--text-secondary); font-weight: 500; min-width: 110px; }}
        .card-val {{ color: var(--text-primary); font-family: monospace; word-break: break-all; text-align: right; }}

        /* Tag Badges */
        .tag-pill-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .tag-pill {{
            background: #e0f2fe;
            color: #0369a1;
            border: 1px solid #bae6fd;
            padding: 4px 9px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-family: monospace;
            font-weight: 500;
        }}

        /* Raw JSON tab with copy button */
        .btn-copy {{
            background: #f1f5f9;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
        }}
        .btn-copy:hover {{
            background: #e2e8f0;
            color: var(--text-primary);
        }}
        pre.json-view {{
            background: #0f172a;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 6px;
            font-family: "Cascadia Code", Consolas, Monaco, monospace;
            font-size: 0.76rem;
            overflow: auto;
            max-height: 500px;
        }}
        .card-row-obj {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 8px 0;
            border-bottom: 1px solid #f1f5f9;
        }}
        .card-row-obj:last-child {{ border-bottom: none; }}
        .card-row-obj .card-label {{
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.82rem;
            margin-bottom: -2px;
        }}
        pre.obj-json {{
            background: #1e293b;
            color: #cbd5e1;
            padding: 10px 14px;
            border-radius: 6px;
            font-family: "Cascadia Code", Consolas, Monaco, monospace;
            font-size: 0.78rem;
            margin: 0;
            overflow-x: auto;
            max-height: 200px;
            line-height: 1.5;
        }}

        .placeholder-hint {{
            text-align: center;
            padding: 50px 20px;
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }}

        /* Settings modal */
        .modal-overlay {{
            position: fixed; inset: 0;
            background: rgba(0,0,0,0.5);
            display: flex; align-items: center; justify-content: center;
            z-index: 100;
        }}
        .modal-box {{
            background: var(--bg-card); border-radius: 12px;
            width: min(95vw, 1200px); max-height: 90vh; display: flex;
            flex-direction: column; box-shadow: var(--shadow-md);
        }}
        .modal-header {{
            padding: 16px 20px; border-bottom: 1px solid var(--border-color);
            display: flex; justify-content: space-between; align-items: center;
        }}
        .modal-header h3 {{ font-size: 1rem; font-weight: 700; }}
        .modal-close {{
            background: none; border: none; font-size: 1.2rem;
            cursor: pointer; color: var(--text-secondary);
            padding: 4px 8px; border-radius: 4px;
        }}
        .modal-close:hover {{ background: #f1f5f9; }}
        .modal-body {{
            padding: 20px; overflow-y: auto;
            flex: 1; font-size: 0.86rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="brand-title">
            <span class="badge badge-blue">Pipeline DAG</span>
            <h1>Tagging Pipeline Visualizer</h1>
        </div>

        <div class="pipeline-selector-group">
            <label for="pipeline-select">Pipeline ID:</label>
            <select id="pipeline-select" class="pipeline-select" onchange="onPipelineChange(this.value)">
                <!-- Dynamically filled -->
            </select>
            <button id="btn-settings" onclick="showSettingsModal()" title="View pipeline settings" style="background:var(--accent);color:#fff;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:0.85rem;font-weight:600;display:flex;align-items:center;gap:6px;"><i class="bi bi-gear-fill"></i> Pipeline Settings</button>
        </div>

        <div class="meta-chips">
            <div class="meta-chip"><span>Files Processed:</span> <strong id="meta-file-count">-</strong></div>
            <div class="meta-chip"><span>Class:</span> <strong id="meta-class-name">-</strong></div>
            <div class="meta-chip"><span>Last Used:</span> <strong id="meta-last-used">-</strong></div>
        </div>
    </header>

    <main>
        <!-- Left: Interactive Graph -->
        <div id="graph-pane">
            <div class="graph-toolbar">
                <button class="toolbar-btn" onclick="panZoomInstance?.zoomIn()">➕ Zoom In</button>
                <button class="toolbar-btn" onclick="panZoomInstance?.zoomOut()">➖ Zoom Out</button>
                <button class="toolbar-btn" onclick="panZoomInstance?.reset()">↺ Reset</button>
                <button class="toolbar-btn" onclick="panZoomInstance?.fit(); panZoomInstance?.center()">⛶ Fit View</button>
            </div><!-- /.graph-toolbar -->

            <div id="mermaid-viewport">
                <div id="mermaid-target"></div>
            </div>
        </div>

        <!-- Resizable Handle -->
        <div id="inspector-resize-handle"></div>

        <!-- Right: Universal Node Inspector -->
        <div id="inspector-pane">
            <div class="inspector-header">
                <div class="inspector-title">
                    <h2 id="inspect-name">Select a Node</h2>
                    <span class="node-type-label" id="inspect-type">Click any block in the flowchart to inspect configuration</span>
                </div>
                <span class="badge badge-green" id="inspect-category" style="display:none;"></span>
            </div>

            <div class="inspector-body" id="inspector-content">
                <div class="placeholder-hint">
                    👈 Click any block in the pipeline diagram to inspect its LLM configuration, routing conditions, target tags, and evaluators.
                </div>
            </div>
        </div>
    </main>

    <script type="text/javascript">
        const pipelinesData = {pipelines_json};
        let currentPipelineId = "{initial_pipeline_id}";
        let panZoomInstance = null;

        // Initialize Mermaid with loose security to enable native click callbacks
        mermaid.initialize({{
            startOnLoad: false,
            securityLevel: 'loose',
            theme: 'base',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis'
            }},
            themeVariables: {{
                fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif',
                primaryColor: '#eff6ff',
                primaryTextColor: '#1e3a8a',
                primaryBorderColor: '#3b82f6',
                lineColor: '#64748b'
            }}
        }});

        // Global node selection callback (callable by Mermaid's click directive and DOM listeners)
        window.selectNode = function(nodeId) {{
            if (!nodeId) return;

            // Normalize nodeId if passed as full element ID (e.g. flowchart-node_1-0 -> node_1)
            const match = String(nodeId).match(/(node_\\d+)/);
            const cleanId = match ? match[1] : nodeId;

            const currentGraph = pipelinesData[currentPipelineId];
            const node = currentGraph?.nodes[cleanId];
            if (!node) {{
                console.warn("Node not found:", cleanId, "available:", Object.keys(currentGraph?.nodes || {{}}));
                return;
            }}

            // Visual feedback: highlight selected node in SVG
            document.querySelectorAll('#mermaid-target .node').forEach(el => {{
                el.classList.remove('active-node');
                if (el.id.includes(cleanId)) {{
                    el.classList.add('active-node');
                }}
            }});

            renderNodeDetails(node);
        }};

        function initPipelineSelector() {{
            const selectEl = document.getElementById('pipeline-select');
            selectEl.innerHTML = '';
            for (const [pId, pData] of Object.entries(pipelinesData)) {{
                const opt = document.createElement('option');
                opt.value = pId;
                opt.textContent = `${{pId.slice(0, 8)}}... (${{pData.file_count}} files) - ${{pData.class_name}}`;
                if (pId === currentPipelineId) opt.selected = true;
                selectEl.appendChild(opt);
            }}
        }}

        function onPipelineChange(newId) {{
            currentPipelineId = newId;
            renderCurrentPipeline();
        }}

        async function renderCurrentPipeline() {{
            const data = pipelinesData[currentPipelineId];
            if (!data) return;

            // Update Metadata Header
            document.getElementById('meta-file-count').textContent = data.file_count + ' files';
            document.getElementById('meta-class-name').textContent = data.class_name;
            document.getElementById('meta-last-used').textContent = data.last_used.split('T')[0] || data.last_used;

            // Render Mermaid Graph
            const target = document.getElementById('mermaid-target');
            target.innerHTML = '';

            try {{
                const uniqueId = 'graph_' + Math.random().toString(36).substring(2, 9);
                const {{ svg }} = await mermaid.render(uniqueId, data.mermaid_code);
                target.innerHTML = svg;

                const svgElement = target.querySelector('svg');
                if (svgElement) {{
                    svgElement.style.width = '100%';
                    svgElement.style.height = '100%';

                    if (panZoomInstance) {{
                        panZoomInstance.destroy();
                    }}
                    panZoomInstance = svgPanZoom(svgElement, {{
                        zoomEnabled: true,
                        controlIconsEnabled: false,
                        fit: true,
                        center: true,
                        minZoom: 0.2,
                        maxZoom: 10
                    }});
                }}

                attachDelegatedClickHandlers();
            }} catch (err) {{
                console.error("Mermaid render error:", err);
                target.innerHTML = `<div style="color:red; padding:20px;">Render Error: ${{err.message}}</div>`;
            }}

            resetInspector();
        }}

        // Robust delegated click handler on SVG container
        function attachDelegatedClickHandlers() {{
            const target = document.getElementById('mermaid-target');
            target.addEventListener('click', (e) => {{
                const nodeGroup = e.target.closest('.node');
                if (!nodeGroup) return;

                const match = (nodeGroup.id || '').match(/(node_\\d+)/);
                if (match) {{
                    window.selectNode(match[1]);
                }}
            }});
        }}

        function renderNodeDetails(node) {{
            document.getElementById('inspect-name').textContent = node.name;
            document.getElementById('inspect-type').textContent = node.type;
            const catBadge = document.getElementById('inspect-category');
            catBadge.style.display = 'inline-block';
            catBadge.textContent = node.category;

            const container = document.getElementById('inspector-content');
            container.innerHTML = '';

            const props = node.properties || {{}};

            // 1. LLM Card (Direct or in condition)
            const llm = props.llm || (props.condition && props.condition._llm) || props._llm;
            if (llm && typeof llm === 'object') {{
                const sec = document.createElement('div');
                sec.className = 'prop-section';
                sec.innerHTML = `
                    <div class="prop-section-header">
                        <div class="prop-section-title">🧠 Language Model</div>
                    </div>
                    <div class="card">
                        <div class="card-row"><span class="card-label">Model</span><span class="card-val">${{llm.model || 'Unknown'}}</span></div>
                        <div class="card-row"><span class="card-label">API URL</span><span class="card-val">${{llm.url || 'Default'}}</span></div>
                    </div>
                `;
                container.appendChild(sec);
            }}

            // 2. Condition / Routing Buckets Card
            if (props.condition && typeof props.condition === 'object') {{
                const cond = props.condition;
                const buckets = cond._buckets || cond.buckets;
                const minChars = cond.min_chars;
                const maxChars = cond.max_chars;

                const sec = document.createElement('div');
                sec.className = 'prop-section';

                let innerHtml = `<div class="prop-section-header"><div class="prop-section-title">⚖️ Routing Criteria</div></div><div class="card">`;
                if (minChars !== undefined || maxChars !== undefined) {{
                    innerHtml += `<div class="card-row"><span class="card-label">Character Bounds</span><span class="card-val">${{minChars ?? 0}} to ${{maxChars ?? 'Infinity'}}</span></div>`;
                }}
                if (buckets && Array.isArray(buckets)) {{
                    const bucketPills = buckets.map(b => `<span class="tag-pill">${{b}}</span>`).join(' ');
                    innerHtml += `<div class="card-row" style="flex-direction:column; gap:6px;"><span class="card-label">Classification Buckets</span><div class="tag-pill-container">${{bucketPills}}</div></div>`;
                }}
                if (cond._samples) {{
                    innerHtml += `<div class="card-row"><span class="card-label">Prompt Samples</span><span class="card-val">Provided</span></div>`;
                }}
                innerHtml += `</div>`;
                sec.innerHTML = innerHtml;
                container.appendChild(sec);
            }}

            // 3. Text Splitter Card
            if (props.text_splitter && typeof props.text_splitter === 'object') {{
                const sec = document.createElement('div');
                sec.className = 'prop-section';
                sec.innerHTML = `
                    <div class="prop-section-header">
                        <div class="prop-section-title">✂️ Text Splitter</div>
                    </div>
                    <div class="card">
                        <div class="card-row"><span class="card-label">Chunk Size</span><span class="card-val">${{props.text_splitter.chunk_size}} chars</span></div>
                        <div class="card-row"><span class="card-label">Chunk Overlap</span><span class="card-val">${{props.text_splitter.chunk_overlap}} chars</span></div>
                    </div>
                `;
                container.appendChild(sec);
            }}

            // 4. Target Tags Pill Collection
            const tags = props.target_tags || props._target_tags;
            if (tags && Array.isArray(tags) && tags.length > 0) {{
                const sec = document.createElement('div');
                sec.className = 'prop-section';
                const pillsHtml = tags.map(t => `<span class="tag-pill">${{t}}</span>`).join('');
                sec.innerHTML = `
                    <div class="prop-section-header">
                        <div class="prop-section-title">🏷️ Target Tags (${{tags.length}})</div>
                    </div>
                    <div class="tag-pill-container">${{pillsHtml}}</div>
                `;
                container.appendChild(sec);
            }}

            // 5. Evaluators & Retries
            if (props.evaluators || props.eval_max_retries !== undefined || props.invoke_max_retries !== undefined) {{
                const sec = document.createElement('div');
                sec.className = 'prop-section';
                const evalsList = (props.evaluators && Array.isArray(props.evaluators) && props.evaluators.length > 0)
                    ? props.evaluators.join(', ')
                    : 'None';
                sec.innerHTML = `
                    <div class="prop-section-header">
                        <div class="prop-section-title">🛡️ Observability & Retries</div>
                    </div>
                    <div class="card">
                        <div class="card-row"><span class="card-label">Evaluators</span><span class="card-val">${{evalsList}}</span></div>
                        <div class="card-row"><span class="card-label">Eval Max Retries</span><span class="card-val">${{props.eval_max_retries ?? 0}}</span></div>
                        <div class="card-row"><span class="card-label">Invoke Max Retries</span><span class="card-val">${{props.invoke_max_retries ?? 0}}</span></div>
                    </div>
                `;
                container.appendChild(sec);
            }}

            // 6. Generic Attributes Table (Catches all unforeseen future properties!)
            const knownKeys = new Set(['name', 'type', 'llm', '_llm', 'condition', 'text_splitter', 'target_tags', '_target_tags', 'evaluators', 'eval_max_retries', 'invoke_max_retries']);
            const remainingKeys = Object.keys(props).filter(k => !knownKeys.has(k));
            if (remainingKeys.length > 0) {{
                const sec = document.createElement('div');
                sec.className = 'prop-section';
                let rowsHtml = '';
                for (const key of remainingKeys) {{
                    const val = props[key];
                    const jsonStr = JSON.stringify(val, null, 2);
                    // Dynamic height: ~15px per line, clamped between 40px and 300px
                    const lineCount = jsonStr.split('\\n').length;
                    const dynHeight = Math.min(300, Math.max(150, lineCount * 15));
                    rowsHtml += `
                        <div class="card-row-obj">
                            <span class="card-label">${{key}}</span>
                            <pre class="obj-json" style="max-height: ${{dynHeight}}px;">${{jsonStr}}</pre>
                        </div>`;
                }}
                sec.innerHTML = `
                    <div class="prop-section-header">
                        <div class="prop-section-title">🔧 Additional Parameters</div>
                    </div>
                    <div class="card">${{rowsHtml}}</div>
                `;
                container.appendChild(sec);
            }}

            // 7. Raw JSON Card with Copy Button
            const jsonSec = document.createElement('div');
            jsonSec.className = 'prop-section';
            jsonSec.innerHTML = `
                <div class="prop-section-header">
                    <div class="prop-section-title">📄 Raw Node Configuration</div>
                    <button class="btn-copy" onclick="copyRawJson(this)">Copy JSON</button>
                </div>
                <pre class="json-view" id="raw-json-block">${{JSON.stringify(node.raw_dict, null, 2)}}</pre>
            `;
            container.appendChild(jsonSec);
        }}

        function copyRawJson(btn) {{
            const block = document.getElementById('raw-json-block');
            if (block) {{
                navigator.clipboard.writeText(block.textContent).then(() => {{
                    btn.textContent = 'Copied!';
                    setTimeout(() => btn.textContent = 'Copy JSON', 1500);
                }});
            }}
        }}

        function resetInspector() {{
            document.getElementById('inspect-name').textContent = 'Select a Node';
            document.getElementById('inspect-type').textContent = 'Click any block in the flowchart to inspect configuration';
            document.getElementById('inspect-category').style.display = 'none';
            document.getElementById('inspector-content').innerHTML = `
                <div class="placeholder-hint">
                    👈 Click any block in the pipeline diagram to inspect its LLM configuration, routing conditions, target tags, and evaluators.
                </div>
            `;
        }}

        function showSettingsModal() {{
            const data = pipelinesData[currentPipelineId];
            if (!data || !data.metadata) return;

            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay';
            overlay.onclick = (e) => {{ if (e.target === overlay) overlay.remove(); }};

            let innerHtml = '';
            for (const [metaKey, metaVal] of Object.entries(data.metadata)) {{
                const typeName = metaVal.type || metaVal.name || 'Unknown';
                innerHtml += `
                    <div class="prop-section">
                        <div class="prop-section-header">
                            <div class="prop-section-title">⚙️ ${{metaKey}}: ${{typeName}}</div>
                        </div>
                        <pre class="json-view">${{JSON.stringify(metaVal, null, 2)}}</pre>
                    </div>`;
            }}

            overlay.innerHTML = `
                <div class="modal-box">
                    <div class="modal-header">
                        <h3>Pipeline Settings</h3>
                        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
                    </div>
                    <div class="modal-body">${{innerHtml}}</div>
                </div>`;

            document.body.appendChild(overlay);
        }}

        // Resizable sidebar
        (function initResize() {{
            const handle = document.getElementById('inspector-resize-handle');
            const pane = document.getElementById('inspector-pane');
            let isResizing = false;
            let startX = 0;
            let startWidth = 0;

            handle.addEventListener('mousedown', (e) => {{
                isResizing = true;
                handle.classList.add('active');
                startX = e.clientX;
                startWidth = pane.offsetWidth;
                e.preventDefault();
            }});

            document.addEventListener('mousemove', (e) => {{
                if (!isResizing) return;
                const dx = startX - e.clientX;  // drag left = positive dx = wider sidebar
                const newWidth = Math.max(300, Math.min(startWidth + dx, window.innerWidth * 0.8));
                pane.style.width = newWidth + 'px';
            }});

            document.addEventListener('mouseup', () => {{
                if (isResizing) {{
                    isResizing = false;
                    handle.classList.remove('active');
                }}
            }});
        }}());

        window.addEventListener('DOMContentLoaded', () => {{
            initPipelineSelector();
            renderCurrentPipeline();
        }});
    </script>
</body>
</html>
"""
    return html_template


# ==============================================================================
# 6. Deep Module Public API
# ==============================================================================


def generate_pipeline_html(
    registry_source: str | Path | dict,
    pipeline_id: str | None = None,
) -> str:
    """Generates an interactive HTML visualization for one or all pipelines in a registry.

    Args:
        registry_source: Path to pipeline_registry.json, or pre-loaded dictionary.
        pipeline_id: Optional pipeline ID to select/visualize. If None, defaults
            to the latest used pipeline and enables switching between all pipelines.

    Returns:
        The absolute path to the generated HTML file.
    """
    registry_data = load_pipeline_registry(registry_source)

    if not registry_data:
        raise ValueError("Pipeline registry is empty.")

    available_ids = list(registry_data.keys())
    if pipeline_id:
        if pipeline_id not in registry_data:
            raise KeyError(
                f"Pipeline ID '{pipeline_id}' not found in registry. "
                f"Available IDs: {', '.join(available_ids)}"
            )
        initial_id = pipeline_id
    else:
        # Sort by last_used descending if available, else first key
        sorted_ids = sorted(
            available_ids,
            key=lambda k: registry_data[k].get("last_used", ""),
            reverse=True,
        )
        initial_id = sorted_ids[0]

    # Build IR for all pipelines in the registry
    pipelines_ir: dict[str, PipelineGraphIR] = {}
    for p_id, p_entry in registry_data.items():
        if isinstance(p_entry, dict):
            engine = ChainGraphInferenceEngine(p_id, p_entry)
            pipelines_ir[p_id] = engine.infer()

    filename = f"pipeline_{initial_id}.html"
    if isinstance(registry_source, (str, Path)):
        output_path = Path(registry_source).parent / filename
    else:
        output_path = Path.cwd() / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Render HTML
    html_content = render_html_dashboard(pipelines_ir, initial_pipeline_id=initial_id)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    resolved_path = str(output_path.resolve())
    print(resolved_path)


# ==============================================================================
# 7. Command-Line Interface
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML visualization of document tagging pipelines."
    )
    parser.add_argument(
        "pipeline_id",
        nargs="?",
        default=None,
        help="Pipeline ID to visualize.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data_store/nseix_circulars/pipeline_registry.json",
        help="Path to pipeline_registry.json (default: data_store/nseix_circulars/pipeline_registry.json)",
    )

    args = parser.parse_args()

    try:
        generate_pipeline_html(
            registry_source=args.input,
            pipeline_id=args.pipeline_id,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error generating pipeline visualization: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
