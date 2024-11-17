// Clear all default node types
LiteGraph.clearRegisteredTypes();

// Helper functions
const formatActions = (actions) => {
  return actions
    .map((action) => {
      if (action.text) {
        return `${action.type}: "${action.text}"`;
      }
      const { type, ...rest } = action;
      return `${type}: ${JSON.stringify(rest)}`;
    })
    .join('\n');
};

// Base node class with common functionality
class PipecatBaseNode extends LiteGraph.LGraphNode {
  constructor(title, color, defaultContent) {
    super();
    this.title = title;
    this.color = color;
    this.size = [400, 200];

    // Default properties - fix the content structure
    this.properties = {
      messages: [
        {
          role: 'system',
          content: defaultContent || 'Enter message...',
        },
      ],
      pre_actions: [],
      post_actions: [],
    };

    // Force minimum width
    this.computeSize = function () {
      return [400, this.size[1]];
    };
  }

  onDrawForeground(ctx) {
    const padding = 15;
    const textColor = '#ddd';
    const labelColor = '#aaa';

    const drawWrappedText = (text, startY, label) => {
      ctx.fillStyle = labelColor;
      ctx.font = '12px Arial';
      ctx.fillText(label, padding, startY + 5);

      ctx.fillStyle = textColor;
      ctx.font = '12px monospace';

      const words = text.split(' ');
      let line = '';
      let y = startY + 25;
      const maxWidth = this.size[0] - padding * 3;

      words.forEach((word) => {
        const testLine = line + word + ' ';
        const metrics = ctx.measureText(testLine);
        if (metrics.width > maxWidth) {
          ctx.fillText(line, padding * 1.5, y);
          line = word + ' ';
          y += 20;
        } else {
          line = testLine;
        }
      });
      ctx.fillText(line, padding * 1.5, y);

      return y + 25;
    };

    let currentY = 40;
    currentY = drawWrappedText(
      this.properties.messages[0].content,
      currentY,
      'System Message'
    );

    if (this.properties.pre_actions.length > 0) {
      currentY = drawWrappedText(
        formatActions(this.properties.pre_actions),
        currentY + 15,
        'Pre-actions'
      );
    }

    if (this.properties.post_actions.length > 0) {
      currentY = drawWrappedText(
        formatActions(this.properties.post_actions),
        currentY + 15,
        'Post-actions'
      );
    }

    const desiredHeight = currentY + padding * 2;
    if (Math.abs(this.size[1] - desiredHeight) > 10) {
      this.size[1] = desiredHeight;
      this.setDirtyCanvas(true, true);
    }
  }

  onSelected() {
    updateSidePanel(this);
  }
}

// Specific node implementations
class PipecatStartNode extends PipecatBaseNode {
  constructor() {
    super('Start Node', '#2ecc71', 'Enter initial system message...');
    this.addOutput('Out', 'flow');
  }
}

class PipecatFlowNode extends PipecatBaseNode {
  constructor() {
    super('Flow Node', '#3498db', 'Enter message content...');
    this.addInput('In', 'flow');
    this.addOutput('Out', 'flow');
  }
}

class PipecatEndNode extends PipecatBaseNode {
  constructor() {
    super('End Node', '#e74c3c', 'Enter final message...');
    this.addInput('In', 'flow');
  }
}

class PipecatFunctionNode extends LiteGraph.LGraphNode {
  constructor() {
    super();
    this.title = 'Function';
    this.addInput('From', 'flow');
    this.addOutput('To', 'flow');

    this.properties = {
      type: 'function',
      function: {
        name: 'function_name',
        description: 'Function description',
        parameters: {
          type: 'object',
          properties: {},
        },
      },
      isTerminal: false,
    };

    this.color = '#9b59b6';
    this.size = [400, 150];

    // Force minimum width
    this.computeSize = function () {
      return [400, this.size[1]];
    };
  }

  // New method to check if the function is terminal
  updateTerminalStatus() {
    const hasOutputConnection =
      this.outputs[0].links && this.outputs[0].links.length > 0;
    const newStatus = !hasOutputConnection;

    if (this.properties.isTerminal !== newStatus) {
      this.properties.isTerminal = newStatus;
      this.setDirtyCanvas(true, true);
    }
  }

  // Override connection methods to update terminal status
  connect(slot, targetNode, targetSlot) {
    const result = super.connect(slot, targetNode, targetSlot);
    this.updateTerminalStatus();
    return result;
  }

  disconnectOutput(slot) {
    super.disconnectOutput(slot);
    this.updateTerminalStatus();
  }

  disconnectInput(slot) {
    super.disconnectInput(slot);
    this.updateTerminalStatus();
  }

  // Called when a connection is removed from any side
  onConnectionsChange(type, slot, connected, link_info) {
    super.onConnectionsChange &&
      super.onConnectionsChange(type, slot, connected, link_info);
    this.updateTerminalStatus();
  }

  onDrawForeground(ctx) {
    // Update terminal status before drawing
    this.updateTerminalStatus();

    const padding = 15;
    const textColor = '#ddd';
    const labelColor = '#aaa';

    // Update node color based on terminal status
    this.color = this.properties.isTerminal ? '#e67e22' : '#9b59b6';

    // Draw terminal/transitional indicator
    ctx.fillStyle = labelColor;
    ctx.font = '11px Arial';
    const typeText = this.properties.isTerminal
      ? '[Terminal Function]'
      : '[Transitional Function]';
    ctx.fillText(typeText, padding, 35);

    // Draw function name
    ctx.fillStyle = textColor;
    ctx.font = '14px monospace';
    ctx.fillText(this.properties.function.name, padding, 60);

    // Draw description
    ctx.fillStyle = labelColor;
    ctx.font = '12px Arial';
    const description = this.properties.function.description;

    // Word wrap description
    const words = description.split(' ');
    let line = '';
    let y = 80;
    const maxWidth = this.size[0] - padding * 3;

    words.forEach((word) => {
      const testLine = line + word + ' ';
      const metrics = ctx.measureText(testLine);
      if (metrics.width > maxWidth) {
        ctx.fillText(line, padding, y);
        line = word + ' ';
        y += 20;
      } else {
        line = testLine;
      }
    });
    ctx.fillText(line, padding, y);

    // Draw parameters indicator if they exist
    const hasParameters =
      Object.keys(this.properties.function.parameters.properties).length > 0;
    if (hasParameters) {
      ctx.fillStyle = '#666';
      ctx.font = '11px Arial';
      ctx.fillText('Has Parameters ⚙️', padding, y + 25);
    }

    // Adjust node height based on content
    const desiredHeight = y + (hasParameters ? 45 : 25);
    if (Math.abs(this.size[1] - desiredHeight) > 10) {
      this.size[1] = desiredHeight;
      this.setDirtyCanvas(true, true);
    }
  }

  onSelected() {
    updateSidePanel(this);
  }

  // Add method to handle graph changes
  onAfterGraphChange() {
    this.updateTerminalStatus();
  }
}

class PipecatMergeNode extends LiteGraph.LGraphNode {
  constructor() {
    super();
    this.title = 'Merge';

    // Start with two input slots (minimum)
    this.addInput('In 1', 'flow');
    this.addInput('In 2', 'flow');

    this.addOutput('Out', 'flow');
    this.color = '#95a5a6';
    this.size = [140, 60]; // Increased height for buttons

    // Add buttons
    this.addWidget('button', '+ Add input', null, () => {
      this.addInput(`In ${this.inputs.length + 1}`, 'flow');
      this.size[1] += 20; // Increase height to accommodate new input
    });

    this.addWidget('button', '- Remove input', null, () => {
      if (this.inputs.length > 2) {
        // Maintain minimum of 2 inputs
        // Disconnect any existing connection to the last input
        if (this.inputs[this.inputs.length - 1].link != null) {
          this.disconnectInput(this.inputs.length - 1);
        }
        // Remove the last input
        this.removeInput(this.inputs.length - 1);
        this.size[1] -= 20; // Decrease height
      }
    });
  }

  onDrawForeground(ctx) {
    const activeConnections = this.inputs.filter(
      (input) => input.link != null
    ).length;
    if (activeConnections > 0) {
      ctx.fillStyle = '#ddd';
      ctx.font = '11px Arial';
    }
  }
}

// Side panel management
function updateSidePanel(node) {
  const content = document.querySelector('.editor-content');
  const noSelection = document.querySelector('.no-selection-message');
  const title = document.getElementById('editor-title');
  const messageEditor = document.querySelector('.message-editor');
  const functionEditor = document.querySelector('.function-editor');

  if (!node) {
    content.style.display = 'none';
    noSelection.style.display = 'block';
    title.textContent = 'Node Editor';
    return;
  }

  content.style.display = 'block';
  noSelection.style.display = 'none';
  title.textContent = `Edit ${node.title}`;

  // Handle different node types
  if (node instanceof PipecatFunctionNode) {
    // Show function editor, hide message editor
    messageEditor.style.display = 'none';
    functionEditor.style.display = 'block';

    // Update function editor with the complete function structure
    const functionData = {
      type: 'function',
      function: node.properties.function,
    };
    document.getElementById('function-editor').value = JSON.stringify(
      functionData,
      null,
      2
    );

    // Add change listener
    document.getElementById('function-editor').onchange = (e) => {
      try {
        const parsed = JSON.parse(e.target.value);
        if (parsed.type === 'function' && parsed.function) {
          node.properties.function = parsed.function;
          node.setDirtyCanvas(true);
        } else {
          throw new Error('Invalid function format');
        }
      } catch (error) {
        console.error('Invalid JSON in function');
        // Revert to previous valid value
        const currentData = {
          type: 'function',
          function: node.properties.function,
        };
        e.target.value = JSON.stringify(currentData, null, 2);
      }
    };
  } else if (node.properties.messages) {
    // Show message editor, hide function editor
    messageEditor.style.display = 'block';
    functionEditor.style.display = 'none';

    // Update message and action editors
    document.getElementById('message-editor').value =
      node.properties.messages[0].content;
    document.getElementById('pre-actions-editor').value = JSON.stringify(
      node.properties.pre_actions || [],
      null,
      2
    );
    document.getElementById('post-actions-editor').value = JSON.stringify(
      node.properties.post_actions || [],
      null,
      2
    );

    // Add change listeners
    document.getElementById('message-editor').onchange = (e) => {
      node.properties.messages[0].content = e.target.value;
      node.setDirtyCanvas(true);
    };

    const setupActionEditor = (elementId, propertyName) => {
      document.getElementById(elementId).onchange = (e) => {
        try {
          const parsed = JSON.parse(e.target.value);
          node.properties[propertyName] = Array.isArray(parsed) ? parsed : [];
          node.setDirtyCanvas(true);
        } catch (error) {
          console.error(`Invalid JSON in ${propertyName}`);
          e.target.value = JSON.stringify(
            node.properties[propertyName],
            null,
            2
          );
        }
      };
    };

    setupActionEditor('pre-actions-editor', 'pre_actions');
    setupActionEditor('post-actions-editor', 'post_actions');
  }
}

// Register node types
LiteGraph.registerNodeType('nodes/Start', PipecatStartNode);
LiteGraph.registerNodeType('nodes/Flow', PipecatFlowNode);
LiteGraph.registerNodeType('nodes/End', PipecatEndNode);
LiteGraph.registerNodeType('nodes/Function', PipecatFunctionNode);
LiteGraph.registerNodeType('flow/Merge', PipecatMergeNode);

// Add this function to handle saving
function generateFlowConfig(graphInstance) {
  if (!graphInstance) {
    throw new Error('No graph instance provided');
  }

  if (!graphInstance._nodes || !Array.isArray(graphInstance._nodes)) {
    throw new Error('No nodes found in the graph');
  }

  const nodes = graphInstance._nodes;
  let startNode = nodes.find((node) => node instanceof PipecatStartNode);

  if (!startNode) {
    throw new Error('No start node found in the flow');
  }

  // Helper function to get node ID
  function getNodeId(node) {
    if (node instanceof PipecatStartNode) return 'start';
    if (node instanceof PipecatEndNode) return 'end';
    // For flow nodes, use the incoming function name
    const incomingFunction = findIncomingFunction(node);
    return incomingFunction ? incomingFunction.properties.function.name : null;
  }

  // Helper function to find incoming function node
  function findIncomingFunction(node) {
    return nodes.find(
      (n) =>
        n instanceof PipecatFunctionNode &&
        n.outputs[0].links &&
        n.outputs[0].links.some((linkId) => {
          const link = graphInstance.links[linkId];
          return link && link.target_id === node.id;
        })
    );
  }

  // Helper function to find all function nodes connected to a node
  function findConnectedFunctions(node) {
    const functions = [];

    // If the node has outputs, check for function nodes
    if (node.outputs && node.outputs[0] && node.outputs[0].links) {
      node.outputs[0].links.forEach((linkId) => {
        const link = graphInstance.links[linkId];
        if (link) {
          const targetNode = nodes.find((n) => n.id === link.target_id);
          if (targetNode && targetNode instanceof PipecatFunctionNode) {
            functions.push({
              type: 'function',
              function: targetNode.properties.function,
            });
          }
        }
      });
    }

    return functions;
  }

  // Build the flow configuration
  const flowConfig = {
    initial_node: 'start',
    nodes: {},
  };

  // Process each node
  nodes.forEach((node) => {
    if (node instanceof PipecatFunctionNode) {
      return; // Skip function nodes, they'll be handled as part of their source nodes
    }

    const nodeId = getNodeId(node);
    if (!nodeId) return; // Skip nodes without valid IDs

    // Get all functions for this node
    const functions = findConnectedFunctions(node);

    // Build node configuration
    flowConfig.nodes[nodeId] = {
      messages: node.properties.messages,
      functions: functions, // Use the connected functions
    };

    // Add actions if they exist and aren't empty
    if (node.properties.pre_actions && node.properties.pre_actions.length > 0) {
      flowConfig.nodes[nodeId].pre_actions = node.properties.pre_actions;
    }
    if (
      node.properties.post_actions &&
      node.properties.post_actions.length > 0
    ) {
      flowConfig.nodes[nodeId].post_actions = node.properties.post_actions;
    }
  });

  return flowConfig;
}

function createFlowFromConfig(graph, flowConfig) {
  // Clear existing graph
  graph.clear();

  const nodeSpacing = {
    horizontal: 400, // Space between main nodes
    vertical: 150, // Space between function nodes
  };
  const startX = 100;
  const startY = 100;
  const nodes = {};

  // First pass: Create all main nodes and establish basic layout
  let currentX = startX;
  let currentY = startY;

  // Create start node first
  const startNode = new PipecatStartNode();
  startNode.properties = {
    messages: flowConfig.nodes.start.messages,
    pre_actions: flowConfig.nodes.start.pre_actions || [],
    post_actions: flowConfig.nodes.start.post_actions || [],
  };
  startNode.pos = [currentX, currentY];
  graph.add(startNode);
  nodes.start = { node: startNode, config: flowConfig.nodes.start };
  currentX += nodeSpacing.horizontal;

  // Create intermediate nodes (not start or end)
  Object.entries(flowConfig.nodes).forEach(([nodeId, nodeConfig]) => {
    if (nodeId !== 'start' && nodeId !== 'end') {
      const node = new PipecatFlowNode();
      node.properties = {
        messages: nodeConfig.messages,
        pre_actions: nodeConfig.pre_actions || [],
        post_actions: nodeConfig.post_actions || [],
      };
      node.pos = [currentX, currentY];
      graph.add(node);
      nodes[nodeId] = { node: node, config: nodeConfig };
      currentX += nodeSpacing.horizontal;
    }
  });

  // Create end node last
  if (flowConfig.nodes.end) {
    const endNode = new PipecatEndNode();
    endNode.properties = {
      messages: flowConfig.nodes.end.messages,
      pre_actions: flowConfig.nodes.end.pre_actions || [],
      post_actions: flowConfig.nodes.end.post_actions || [],
    };
    endNode.pos = [currentX, currentY];
    graph.add(endNode);
    nodes.end = { node: endNode, config: flowConfig.nodes.end };
  }

  // Second pass: Create function nodes and connections
  Object.entries(nodes).forEach(([sourceId, { node: sourceNode, config }]) => {
    if (!config.functions) return;

    // Calculate vertical spacing for multiple functions
    const functionCount = config.functions.length;
    const totalHeight = functionCount * nodeSpacing.vertical;
    const startingY = currentY - totalHeight / 2;

    config.functions.forEach((funcConfig, index) => {
      // Create function node
      const functionNode = new PipecatFunctionNode();
      functionNode.properties.function = funcConfig.function;
      // Set terminal status based on whether the function name matches a node
      functionNode.properties.isTerminal = !nodes[funcConfig.function.name];

      // Position function node
      const sourceX = sourceNode.pos[0];
      const targetNodeId = funcConfig.function.name;
      const targetInfo = nodes[targetNodeId];

      let functionX, functionY;
      if (targetInfo) {
        // This is a connecting function
        const targetX = targetInfo.node.pos[0];
        functionX = sourceX + (targetX - sourceX) / 2;
        functionY = startingY + index * nodeSpacing.vertical;

        // Connect to target node
        functionNode.pos = [functionX, functionY];
        graph.add(functionNode);
        sourceNode.connect(0, functionNode, 0);
        functionNode.connect(0, targetInfo.node, 0);
      } else {
        // This is a terminal function
        functionX = sourceX + nodeSpacing.horizontal / 2;
        functionY = startingY + index * nodeSpacing.vertical;

        // Add terminal function node
        functionNode.pos = [functionX, functionY];
        graph.add(functionNode);
        sourceNode.connect(0, functionNode, 0);
      }
    });
  });

  // Center the graph in the canvas
  graph.arrange();
  graph.setDirtyCanvas(true, true);
}

// Initialize the graph when the document is ready
document.addEventListener('DOMContentLoaded', function () {
  const graph = new LGraph();
  const canvas = new LGraphCanvas('#graph-canvas', graph);

  graph.onNodeSelected = updateSidePanel;
  graph.onNodeDeselected = () => updateSidePanel(null);

  function resizeCanvas() {
    const canvas = document.getElementById('graph-canvas');
    const container = document.getElementById('graph-container');
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
  }

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Add a custom event handler to update function nodes
  graph.onAfterChange = () => {
    graph._nodes.forEach((node) => {
      if (node instanceof PipecatFunctionNode) {
        node.onAfterGraphChange();
      }
    });
  };

  // Button handlers
  document.getElementById('new-flow').onclick = () => graph.clear();

  // Import handler
  document.getElementById('import-flow').onclick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          // Clean the input string
          const cleanInput = event.target.result
            .replace(/[\u0000-\u001F\u007F-\u009F]/g, '') // Remove control characters
            .replace(/\r\n/g, '\n') // Normalize line endings
            .replace(/\r/g, '\n'); // Replace any remaining \r

          console.log('Cleaned input:', cleanInput);

          const flowConfig = JSON.parse(cleanInput);
          console.log('Parsed config:', flowConfig);

          createFlowFromConfig(graph, flowConfig);
          console.log('Successfully imported flow configuration');
        } catch (error) {
          console.error('Error importing flow:', error);
          console.error('Error details:', {
            message: error.message,
            position: error.position,
            stack: error.stack,
          });
          alert('Error importing flow: ' + error.message);
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  // Export handler
  document.getElementById('export-flow').onclick = async () => {
    try {
      const flowConfig = generateFlowConfig(graph);
      console.log('Generated Flow Configuration:');
      console.log(JSON.stringify(flowConfig, null, 2));

      // Generate timestamp
      const timestamp = new Date()
        .toISOString()
        .replace(/[:.]/g, '-')
        .replace('T', '_')
        .slice(0, -5);

      // Create a clean JSON string
      const cleanJson = JSON.stringify(flowConfig, null, 2)
        .replace(/\\n/g, '\n') // Convert \n to actual newlines
        .replace(/\\"/g, '"') // Fix escaped quotes
        .replace(/"\[\s*\{/g, '[{') // Fix array openings
        .replace(/\}\s*\]"/g, '}]') // Fix array closings
        .replace(/,(\s*}])/g, '$1') // Remove trailing commas before }]
        .replace(/\n\s*,\s*}/g, '\n}') // Fix trailing commas
        .replace(/}]\s*,\s*}/g, '}]}') // Fix nested array closings
        .replace(/\n\s{20,}/g, '\n    ') // Fix over-indentation
        .trim(); // Trim the whole output

      const blob = new Blob([cleanJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `flow_config_${timestamp}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error generating flow configuration:', error);
      alert('Error generating flow configuration: ' + error.message);
    }
  };

  graph.start();
});
