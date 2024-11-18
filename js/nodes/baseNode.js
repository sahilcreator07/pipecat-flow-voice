//
// Copyright (c) 2024, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

import { editorState } from '../editor/editorState.js';
import { formatActions } from '../utils/helpers.js';

export class PipecatBaseNode extends LiteGraph.LGraphNode {
  constructor(title, color, defaultContent) {
    super();
    this.title = title;
    this.color = color;
    this.size = [400, 200];

    // Default properties
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
      'Message'
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
    editorState.updateSidePanel(this);
  }
}
