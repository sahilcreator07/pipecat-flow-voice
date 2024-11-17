class FlowValidator {
  constructor(flowConfig) {
    this.flow = flowConfig;
    this.errors = [];
  }

  validate() {
    this.errors = [];

    this._validateInitialNode();
    this._validateNodeReferences();
    this._validateNodeContents();

    return this.errors;
  }

  _validateInitialNode() {
    if (!this.flow.initial_node) {
      this.errors.push('Initial node must be specified');
    } else if (!this.flow.nodes[this.flow.initial_node]) {
      this.errors.push(
        `Initial node '${this.flow.initial_node}' not found in nodes`
      );
    }
  }

  _validateNodeReferences() {
    Object.entries(this.flow.nodes).forEach(([nodeId, node]) => {
      if (node.functions) {
        const functionNames = node.functions
          .map((func) => func.function?.name)
          .filter(Boolean);

        functionNames.forEach((funcName) => {
          if (!this.flow.nodes[funcName] && funcName !== 'end') {
            this.errors.push(
              `Node '${nodeId}' has function '${funcName}' that doesn't reference a valid node`
            );
          }
        });
      }
    });
  }

  _validateNodeContents() {
    Object.entries(this.flow.nodes).forEach(([nodeId, node]) => {
      // Validate messages
      if (!node.messages || node.messages.length === 0) {
        this.errors.push(`Node '${nodeId}' must have at least one message`);
      }

      node.messages?.forEach((msg) => {
        if (!msg.role) {
          this.errors.push(`Message in node '${nodeId}' missing 'role'`);
        }
        if (!msg.content) {
          this.errors.push(`Message in node '${nodeId}' missing 'content'`);
        }
      });

      // Validate functions
      node.functions?.forEach((func) => {
        if (!func.function) {
          this.errors.push(
            `Function in node '${nodeId}' missing 'function' object`
          );
        } else if (!func.function.name) {
          this.errors.push(`Function in node '${nodeId}' missing 'name'`);
        }
      });
    });
  }
}

// Helper function to validate a flow configuration
function validateFlow(flowConfig) {
  const validator = new FlowValidator(flowConfig);
  return {
    valid: validator.validate().length === 0,
    errors: validator.errors,
  };
}

// Example usage in main.js:
document.getElementById('export-flow').onclick = async () => {
  try {
    const flowConfig = generateFlowConfig(graph);

    // Validate before export
    const validation = validateFlow(flowConfig);
    if (!validation.valid) {
      console.error('Flow validation errors:', validation.errors);
      if (!confirm('Flow has validation errors. Export anyway?')) {
        return;
      }
    }

    // Continue with export...
    console.log('Generated Flow Configuration:');
    console.log(JSON.stringify(flowConfig, null, 2));
    // ... rest of export code ...
  } catch (error) {
    console.error('Error generating flow configuration:', error);
    alert('Error generating flow configuration: ' + error.message);
  }
};

// Also use during import
document.getElementById('import-flow').onclick = () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  input.onchange = (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const flowConfig = JSON.parse(event.target.result);

        // Validate imported flow
        const validation = validateFlow(flowConfig);
        if (!validation.valid) {
          console.error('Flow validation errors:', validation.errors);
          if (!confirm('Imported flow has validation errors. Import anyway?')) {
            return;
          }
        }

        createFlowFromConfig(graph, flowConfig);
        console.log('Successfully imported flow configuration');
      } catch (error) {
        console.error('Error importing flow:', error);
        alert('Error importing flow: ' + error.message);
      }
    };
    reader.readAsText(file);
  };
  input.click();
};
