//
// Copyright (c) 2024, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

export class PipecatMergeNode extends LiteGraph.LGraphNode {
  constructor() {
    super();
    this.title = 'Merge';

    // Start with two input slots
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
