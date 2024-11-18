/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * Sets up the canvas and its event handlers
 * @param {LGraph} graph - The LiteGraph instance
 * @returns {LGraphCanvas} The configured canvas instance
 */
export function setupCanvas(graph) {
  const canvas = new LGraphCanvas("#graph-canvas", graph);

  /**
   * Resizes the canvas to fit its container
   */
  function resizeCanvas() {
    const canvasElement = document.getElementById("graph-canvas");
    const container = document.getElementById("graph-container");
    canvasElement.width = container.offsetWidth;
    canvasElement.height = container.offsetHeight;
  }

  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  return canvas;
}
