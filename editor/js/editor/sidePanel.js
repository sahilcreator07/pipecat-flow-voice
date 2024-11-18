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
    // Message editor change handler
    document.getElementById("message-editor").onchange = (e) => {
      const selectedNode = this.graph.getSelectedNode();
      if (selectedNode && selectedNode.properties.messages) {
        selectedNode.properties.messages[0].content = e.target.value;
        selectedNode.setDirtyCanvas(true);
      }
    };

    /**
     * Sets up an action editor
     * @param {string} elementId - ID of the editor element
     * @param {string} propertyName - Name of the property to update
     */
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
              2,
            );
          }
        }
      };
    };

    setupActionEditor("pre-actions-editor", "pre_actions");
    setupActionEditor("post-actions-editor", "post_actions");

    // Function editor change handler
    document.getElementById("function-editor").onchange = (e) => {
      const selectedNode = this.graph.getSelectedNode();
      if (selectedNode) {
        try {
          const parsed = JSON.parse(e.target.value);
          if (parsed.type === "function" && parsed.function) {
            selectedNode.properties.function = parsed.function;
            selectedNode.setDirtyCanvas(true);
          } else {
            throw new Error("Invalid function format");
          }
        } catch (error) {
          console.error("Invalid JSON in function");
          const currentData = {
            type: "function",
            function: selectedNode.properties.function,
          };
          e.target.value = JSON.stringify(currentData, null, 2);
        }
      }
    };
  }

  /**
   * Updates the side panel with node data
   * @param {PipecatBaseNode|null}
   */
  updatePanel(node) {
    if (!node) {
      this.elements.content.style.display = "none";
      this.elements.noSelection.style.display = "block";
      this.elements.title.textContent = "Node Editor";
      return;
    }

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
      document.getElementById("message-editor").value =
        node.properties.messages[0].content;
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
