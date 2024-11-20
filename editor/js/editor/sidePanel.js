/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

import { editorState } from "./editorState.js";

/**
 * @typedef {Object} SidePanelElements
 * @property {HTMLElement} panel - The main panel element
 * @property {HTMLElement} content - The content container
 * @property {HTMLElement} noSelection - The no-selection message element
 * @property {HTMLElement} title - The panel title element
 * @property {HTMLElement} messageEditor - The message editor container
 * @property {HTMLElement} functionEditor - The function editor container
 */

/**
 * Manages the side panel UI and interactions
 */
export class SidePanel {
  /**
   * Creates a new SidePanel instance
   * @param {LGraph} graph - The LiteGraph instance
   */
  constructor(graph) {
    this.graph = graph;
    this.selectedNode = null;

    /** @type {SidePanelElements} */
    this.elements = {
      panel: document.querySelector("#side-panel"),
      content: document.querySelector(".editor-content"),
      noSelection: document.querySelector(".no-selection-message"),
      title: document.getElementById("editor-title"),
      messageEditor: document.querySelector(".message-editor"),
      functionEditor: document.querySelector(".function-editor"),
    };

    this.setupEventListeners();
  }

  /**
   * Sets up event listeners for the editors
   */
  setupEventListeners() {
    // First verify elements exist
    const messageEditor = document.getElementById("message-editor");
    const preActionsEditor = document.getElementById("pre-actions-editor");
    const postActionsEditor = document.getElementById("post-actions-editor");
    const functionEditor = document.getElementById("function-editor");

    console.log("Found editors:", {
      messageEditor,
      preActionsEditor,
      postActionsEditor,
      functionEditor,
    });

    // Prevent node deselection when clicking in the side panel
    this.elements.panel.addEventListener("mousedown", (e) => {
      e.stopPropagation();
    });

    if (!messageEditor) {
      console.error("Message editor element not found!");
      return;
    }

    // Message editor change handler
    messageEditor.addEventListener("change", (e) => {
      console.log("Message editor change event fired");
      if (!this.selectedNode) {
        console.log("No node selected");
        return;
      }

      try {
        const messages = JSON.parse(e.target.value);
        console.log("Parsed messages:", messages);

        if (Array.isArray(messages)) {
          console.log("Updating node messages");
          this.selectedNode.properties.messages = messages;
          this.selectedNode.setDirtyCanvas(true);
          this.graph.change();
          console.log("Updated node properties:", this.selectedNode.properties);
        }
      } catch (error) {
        console.error("JSON parse error:", error);
        e.target.value = JSON.stringify(
          this.selectedNode.properties.messages,
          null,
          2,
        );
      }
    });

    // Also try the input event
    messageEditor.addEventListener("input", (e) => {
      console.log("Message editor input event fired");
    });

    // Action editors
    const setupActionEditor = (element, propertyName) => {
      if (!element) {
        console.error(`${propertyName} editor element not found!`);
        return;
      }

      element.addEventListener("change", (e) => {
        console.log(`${propertyName} editor change event fired`);
        if (!this.selectedNode) return;

        try {
          const parsed = JSON.parse(e.target.value);
          this.selectedNode.properties[propertyName] = Array.isArray(parsed)
            ? parsed
            : [];
          this.selectedNode.setDirtyCanvas(true);
          this.graph.change();
        } catch (error) {
          console.error(`Invalid JSON in ${propertyName}`);
          e.target.value = JSON.stringify(
            this.selectedNode.properties[propertyName],
            null,
            2,
          );
        }
      });
    };

    setupActionEditor(preActionsEditor, "pre_actions");
    setupActionEditor(postActionsEditor, "post_actions");

    // Function editor
    if (functionEditor) {
      functionEditor.addEventListener("change", (e) => {
        console.log("Function editor change event fired");
        if (!this.selectedNode) return;

        try {
          const parsed = JSON.parse(e.target.value);
          if (parsed.type === "function" && parsed.function) {
            this.selectedNode.properties.function = parsed.function;
            this.selectedNode.setDirtyCanvas(true);
            this.graph.change();
          }
        } catch (error) {
          console.error("Invalid JSON in function");
          const currentData = {
            type: "function",
            function: this.selectedNode.properties.function,
          };
          e.target.value = JSON.stringify(currentData, null, 2);
        }
      });
    }
  }

  /**
   * Updates the side panel with node data
   * @param {PipecatBaseNode|null} node
   */
  updatePanel(node) {
    console.log("updatePanel called with node:", node);

    // Update the selected node reference
    this.selectedNode = node;

    if (!node) {
      console.log("No node provided, hiding panel");
      this.elements.content.style.display = "none";
      this.elements.noSelection.style.display = "block";
      this.elements.title.textContent = "Node Editor";
      return;
    }

    console.log("Node type:", node.constructor.name);
    console.log("Node properties:", node.properties);

    this.elements.content.style.display = "block";
    this.elements.noSelection.style.display = "none";
    this.elements.title.textContent = `Edit ${node.title}`;

    // Handle different node types
    if (node.constructor.name === "PipecatFunctionNode") {
      // Show function editor, hide message editor
      this.elements.messageEditor.style.display = "none";
      this.elements.functionEditor.style.display = "block";

      // Update function editor with the complete function structure
      const functionData = {
        type: "function",
        function: node.properties.function,
      };
      document.getElementById("function-editor").value = JSON.stringify(
        functionData,
        null,
        2,
      );
    } else if (node.properties.messages) {
      // Show message editor, hide function editor
      this.elements.messageEditor.style.display = "block";
      this.elements.functionEditor.style.display = "none";

      // Update message and action editors
      document.getElementById("message-editor").value = JSON.stringify(
        node.properties.messages,
        null,
        2,
      );

      // Update action editors
      document.getElementById("pre-actions-editor").value = JSON.stringify(
        node.properties.pre_actions || [],
        null,
        2,
      );
      document.getElementById("post-actions-editor").value = JSON.stringify(
        node.properties.post_actions || [],
        null,
        2,
      );
    }
  }
}
