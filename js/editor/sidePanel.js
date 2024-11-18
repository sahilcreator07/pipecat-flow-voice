//
// Copyright (c) 2024, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

export class SidePanel {
  constructor(graph) {
    this.graph = graph;
    this.panel = document.querySelector('#side-panel');
    this.content = document.querySelector('.editor-content');
    this.noSelection = document.querySelector('.no-selection-message');
    this.title = document.getElementById('editor-title');
    this.messageEditor = document.querySelector('.message-editor');
    this.functionEditor = document.querySelector('.function-editor');

    this.setupEventListeners();
  }

  setupEventListeners() {
    // Message editor change handler
    document.getElementById('message-editor').onchange = (e) => {
      const selectedNode = this.graph.getSelectedNode();
      if (selectedNode && selectedNode.properties.messages) {
        selectedNode.properties.messages[0].content = e.target.value;
        selectedNode.setDirtyCanvas(true);
      }
    };

    // Setup action editors
    const setupActionEditor = (elementId, propertyName) => {
      document.getElementById(elementId).onchange = (e) => {
        const selectedNode = this.graph.getSelectedNode();
        if (selectedNode) {
          try {
            const parsed = JSON.parse(e.target.value);
            selectedNode.properties[propertyName] = Array.isArray(parsed)
              ? parsed
              : [];
            selectedNode.setDirtyCanvas(true);
          } catch (error) {
            console.error(`Invalid JSON in ${propertyName}`);
            e.target.value = JSON.stringify(
              selectedNode.properties[propertyName],
              null,
              2
            );
          }
        }
      };
    };

    setupActionEditor('pre-actions-editor', 'pre_actions');
    setupActionEditor('post-actions-editor', 'post_actions');

    // Function editor change handler
    document.getElementById('function-editor').onchange = (e) => {
      const selectedNode = this.graph.getSelectedNode();
      if (selectedNode) {
        try {
          const parsed = JSON.parse(e.target.value);
          if (parsed.type === 'function' && parsed.function) {
            selectedNode.properties.function = parsed.function;
            selectedNode.setDirtyCanvas(true);
          } else {
            throw new Error('Invalid function format');
          }
        } catch (error) {
          console.error('Invalid JSON in function');
          const currentData = {
            type: 'function',
            function: selectedNode.properties.function,
          };
          e.target.value = JSON.stringify(currentData, null, 2);
        }
      }
    };
  }

  updatePanel(node) {
    if (!node) {
      this.content.style.display = 'none';
      this.noSelection.style.display = 'block';
      this.title.textContent = 'Node Editor';
      return;
    }

    this.content.style.display = 'block';
    this.noSelection.style.display = 'none';
    this.title.textContent = `Edit ${node.title}`;

    // Handle different node types
    if (node.constructor.name === 'PipecatFunctionNode') {
      // Show function editor, hide message editor
      this.messageEditor.style.display = 'none';
      this.functionEditor.style.display = 'block';

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
    } else if (node.properties.messages) {
      // Show message editor, hide function editor
      this.messageEditor.style.display = 'block';
      this.functionEditor.style.display = 'none';

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
    }
  }
}
