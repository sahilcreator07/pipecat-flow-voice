/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * Singleton class to manage global editor state
 */
class EditorState {
  /** @type {EditorState|null} */
  static instance = null;

  /**
   * Creates or returns the EditorState singleton
   */
  constructor() {
    if (EditorState.instance) {
      return EditorState.instance;
    }
    /** @type {import('./sidePanel').SidePanel|null} */
    this.sidePanel = null;
    EditorState.instance = this;
  }

  /**
   * Sets the side panel instance
   * @param {import('./sidePanel').SidePanel} sidePanel - The side panel instance
   */
  setSidePanel(sidePanel) {
    this.sidePanel = sidePanel;
  }

  /**
   * Updates the side panel with node data
   * @param {import('../nodes/baseNode').PipecatBaseNode|null} node - The selected node or null
   */
  updateSidePanel(node) {
    if (this.sidePanel) {
      this.sidePanel.updatePanel(node);
    }
  }
}

export const editorState = new EditorState();
