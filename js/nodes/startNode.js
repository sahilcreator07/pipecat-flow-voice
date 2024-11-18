import { PipecatBaseNode } from './baseNode.js';

export class PipecatStartNode extends PipecatBaseNode {
  constructor() {
    super('Start Node', '#2ecc71', 'Enter initial system message...');
    this.addOutput('Out', 'flow');
  }
}
