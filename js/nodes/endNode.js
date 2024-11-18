import { PipecatBaseNode } from './baseNode.js';

export class PipecatEndNode extends PipecatBaseNode {
  constructor() {
    super('End Node', '#e74c3c', 'Enter final message...');
    this.addInput('In', 'flow', {
      multipleConnections: true,
      linkType: LiteGraph.MULTIPLE_LINK,
    });
  }

  // Override the default connection behavior
  onConnectInput(targetSlot, type, output, input_node, input_slot) {
    if (this.inputs[0].link == null) {
      this.inputs[0].link = [];
    }
    return true;
  }

  // Override connection methods
  connect(slot, targetNode, targetSlot) {
    if (this.inputs[slot].link == null) {
      this.inputs[slot].link = [];
    }
    return super.connect(slot, targetNode, targetSlot);
  }

  onConnectionsChange(type, slot, connected, link_info) {
    if (type === LiteGraph.INPUT && this.inputs[slot].link == null) {
      this.inputs[slot].link = [];
    }
    super.onConnectionsChange &&
      super.onConnectionsChange(type, slot, connected, link_info);
  }
}
