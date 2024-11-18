//
// Copyright (c) 2024, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

import { PipecatBaseNode } from './baseNode.js';

export class PipecatFlowNode extends PipecatBaseNode {
  constructor() {
    super('Flow Node', '#3498db', 'Enter message content...');
    this.addInput('In', 'flow');
    this.addOutput('Out', 'flow');
  }
}
