//
// Copyright (c) 2024, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

import { editorState } from '../editor/editorState.js';

export class PipecatFunctionNode extends LiteGraph.LGraphNode {
  constructor() {
    super();
    this.title = 'Function';
    this.addInput('From', 'flow');
    this.addOutput('To', 'flow', {
      linkType: LiteGraph.MULTIPLE_LINK,
    });

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
  }

  computeSize() {
    return [400, this.size[1]];
  }

  updateTerminalStatus() {
    const hasOutputConnection =
      this.outputs[0].links && this.outputs[0].links.length > 0;
    const newStatus = !hasOutputConnection;

    if (this.properties.isTerminal !== newStatus) {
      this.properties.isTerminal = newStatus;
      this.setDirtyCanvas(true, true);
    }
  }

  connect(slot, targetNode, targetSlot) {
    if (slot === 1 && this.outputs[slot].links == null) {
      this.outputs[slot].links = [];
    }
    const result = super.connect(slot, targetNode, targetSlot);
    this.updateTerminalStatus();
    return result;
  }

  disconnectOutput(slot) {
    if (this.outputs[slot].links == null) {
      this.outputs[slot].links = [];
    }
    super.disconnectOutput(slot);
    this.updateTerminalStatus();
  }

  disconnectInput(slot) {
    super.disconnectInput(slot);
    this.updateTerminalStatus();
  }

  onConnectionsChange(type, slot, connected, link_info) {
    if (type === LiteGraph.OUTPUT && this.outputs[slot].links == null) {
      this.outputs[slot].links = [];
    }
    super.onConnectionsChange &&
      super.onConnectionsChange(type, slot, connected, link_info);
    this.updateTerminalStatus();
  }

  onDrawForeground(ctx) {
    this.updateTerminalStatus();

    const padding = 15;
    const textColor = '#ddd';
    const labelColor = '#aaa';

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

    // Draw parameters indicator
    const hasParameters =
      Object.keys(this.properties.function.parameters.properties).length > 0;
    if (hasParameters) {
      ctx.fillStyle = '#666';
      ctx.font = '11px Arial';
      ctx.fillText('Has Parameters ⚙️', padding, y + 25);
    }

    // Adjust node height
    const desiredHeight = y + (hasParameters ? 45 : 25);
    if (Math.abs(this.size[1] - desiredHeight) > 10) {
      this.size[1] = desiredHeight;
      this.setDirtyCanvas(true, true);
    }
  }

  onSelected() {
    editorState.updateSidePanel(this);
  }

  onAfterGraphChange() {
    this.updateTerminalStatus();
  }
}
