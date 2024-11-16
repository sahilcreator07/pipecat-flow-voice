from fastapi import APIRouter, HTTPException

from .models import FlowConfig, VisualFlow
from .validation import validate_flow

router = APIRouter()


@router.post("/import")
async def import_flow(flow_config: FlowConfig) -> VisualFlow:
    """Convert a Pipecat flow configuration to a visual representation"""
    try:
        # Create a basic visual layout
        nodes = {}
        x, y = 100, 100  # Starting position

        for node_id, config in flow_config.nodes.items():
            nodes[node_id] = {"id": node_id, "position": {"x": x, "y": y}, "config": config}
            x += 250  # Simple horizontal layout

        return VisualFlow(nodes=nodes, initial_node=flow_config.initial_node)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/validate")
async def validate_flow_config(flow_config: FlowConfig) -> dict:
    """Validate a flow configuration"""
    errors = validate_flow(flow_config)
    return {"valid": len(errors) == 0, "errors": errors}


@router.post("/export")
async def export_flow(visual_flow: VisualFlow) -> FlowConfig:
    """Convert a visual flow back to Pipecat configuration"""
    try:
        nodes = {node_id: node.config for node_id, node in visual_flow.nodes.items()}

        return FlowConfig(initial_node=visual_flow.initial_node, nodes=nodes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
