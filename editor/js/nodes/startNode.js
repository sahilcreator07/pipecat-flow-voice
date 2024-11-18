/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

import { PipecatBaseNode } from "./baseNode.js";

/**
 * Represents the starting node of a flow
 * @extends PipecatBaseNode
 */
export class PipecatStartNode extends PipecatBaseNode {
  /**
   * Creates a new start node
   */
  constructor() {
    super("Start Node", "#2ecc71", "Enter initial system message...");
    this.addOutput("Out", "flow");
  }
}
