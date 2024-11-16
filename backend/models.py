from typing import Dict, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class Function(BaseModel):
    type: str = "function"
    function: Dict


class Action(BaseModel):
    type: str
    text: Optional[str] = None  # For TTS actions


class NodeConfig(BaseModel):
    messages: List[Dict]
    functions: List[Dict]
    pre_actions: Optional[List[Dict]] = None
    post_actions: Optional[List[Dict]] = None


class Node(BaseModel):
    id: str
    position: Dict[str, float]
    config: NodeConfig


class FlowConfig(BaseModel):
    initial_node: str
    nodes: Dict[str, NodeConfig]


class VisualFlow(BaseModel):
    nodes: Dict[str, Node]
    initial_node: str
