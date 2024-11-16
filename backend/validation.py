from typing import List

from .models import FlowConfig


class FlowValidator:
    def __init__(self, flow: FlowConfig):
        self.flow = flow
        self.errors: List[str] = []

    def validate(self) -> List[str]:
        """Run all validation checks and return list of errors"""
        self.errors = []

        self._validate_initial_node()
        self._validate_node_references()
        self._validate_node_contents()

        return self.errors

    def _validate_initial_node(self):
        """Ensure initial node exists and is valid"""
        if not self.flow.initial_node:
            self.errors.append("Initial node must be specified")
        elif self.flow.initial_node not in self.flow.nodes:
            self.errors.append(f"Initial node '{self.flow.initial_node}' not found in nodes")

    def _validate_node_references(self):
        """Ensure all function references point to valid nodes"""
        for node_id, node in self.flow.nodes.items():
            function_names = {
                func["function"]["name"]
                for func in node.functions
                if "function" in func and "name" in func["function"]
            }

            # Check if function names reference valid nodes
            for func_name in function_names:
                if func_name not in self.flow.nodes and func_name != "end":
                    self.errors.append(
                        f"Node '{node_id}' has function '{func_name}' that doesn't reference a valid node"
                    )

    def _validate_node_contents(self):
        """Validate individual node configurations"""
        for node_id, node in self.flow.nodes.items():
            # Validate messages
            if not node.messages:
                self.errors.append(f"Node '{node_id}' must have at least one message")

            for msg in node.messages:
                if "role" not in msg:
                    self.errors.append(f"Message in node '{node_id}' missing 'role'")
                if "content" not in msg:
                    self.errors.append(f"Message in node '{node_id}' missing 'content'")

            # Validate functions
            for func in node.functions:
                if "function" not in func:
                    self.errors.append(f"Function in node '{node_id}' missing 'function' object")
                elif "name" not in func["function"]:
                    self.errors.append(f"Function in node '{node_id}' missing 'name'")


def validate_flow(flow: FlowConfig) -> List[str]:
    """Convenience function to validate a flow configuration"""
    validator = FlowValidator(flow)
    return validator.validate()
