/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

import {
  PipecatStartNode,
  PipecatFlowNode,
  PipecatEndNode,
  PipecatFunctionNode,
  PipecatMergeNode,
} from "../nodes/index.js";

/**
 * Creates a graph from a flow configuration
 * @param {LGraph} graph - The LiteGraph instance
 * @param {FlowConfig} flowConfig - The flow configuration
 */
export function createFlowFromConfig(graph, flowConfig) {
  // Clear existing graph
  graph.clear();

  const nodeSpacing = {
    horizontal: 400,
    vertical: 150,
  };
  const startX = 100;
  const startY = 100;
  /** @type {Object.<string, { node: LGraphNode, config: any }>} */
  const nodes = {};

  // First pass: Create all main nodes
  let currentX = startX;
  let currentY = startY;

  // Create start node
  const startNode = new PipecatStartNode();
  startNode.properties = {
    messages: flowConfig.nodes.start.messages,
    pre_actions: flowConfig.nodes.start.pre_actions || [],
    post_actions: flowConfig.nodes.start.post_actions || [],
  };
  startNode.pos = [currentX, currentY];
  graph.add(startNode);
  nodes.start = { node: startNode, config: flowConfig.nodes.start };
  currentX += nodeSpacing.horizontal;

  // Create intermediate nodes
  Object.entries(flowConfig.nodes).forEach(([nodeId, nodeConfig]) => {
    if (nodeId !== "start" && nodeId !== "end") {
      const node = new PipecatFlowNode();
      node.properties = {
        messages: nodeConfig.messages,
        pre_actions: nodeConfig.pre_actions || [],
        post_actions: nodeConfig.post_actions || [],
      };
      node.pos = [currentX, currentY];
      graph.add(node);
      nodes[nodeId] = { node: node, config: nodeConfig };
      currentX += nodeSpacing.horizontal;
    }
  });

  // Create end node
  if (flowConfig.nodes.end) {
    const endNode = new PipecatEndNode();
    endNode.properties = {
      messages: flowConfig.nodes.end.messages,
      pre_actions: flowConfig.nodes.end.pre_actions || [],
      post_actions: flowConfig.nodes.end.post_actions || [],
    };
    endNode.pos = [currentX, currentY];
    graph.add(endNode);
    nodes.end = { node: endNode, config: flowConfig.nodes.end };
  }

  // Analyze function targets
  const functionTargets = new Map();
  Object.entries(flowConfig.nodes).forEach(([sourceNodeId, nodeConfig]) => {
    if (nodeConfig.functions) {
      nodeConfig.functions.forEach((funcConfig) => {
        const targetName = funcConfig.function.name;
        if (!functionTargets.has(targetName)) {
          functionTargets.set(targetName, []);
        }
        functionTargets.get(targetName).push({
          sourceNodeId,
          funcConfig,
        });
      });
    }
  });

  /**
   * Creates a merge node
   * @param {Array<LGraphNode>} sourceNodes - Source nodes to merge
   * @param {LGraphNode} targetNode - Target node
   * @returns {PipecatMergeNode} The created merge node
   */
  function createMergeNode(sourceNodes, targetNode) {
    const mergeNode = new PipecatMergeNode();
    graph.add(mergeNode);

    const avgX =
      sourceNodes.reduce((sum, n) => sum + n.pos[0], 0) / sourceNodes.length;
    const avgY =
      sourceNodes.reduce((sum, n) => sum + n.pos[1], 0) / sourceNodes.length;

    mergeNode.pos = [(avgX + targetNode.pos[0]) / 2, avgY];

    while (mergeNode.inputs.length < sourceNodes.length) {
      mergeNode.addInput(`In ${mergeNode.inputs.length + 1}`, "flow");
      mergeNode.size[1] += 20;
    }

    return mergeNode;
  }

  // Second pass: Create function nodes and handle merging
  functionTargets.forEach((sourceFunctions, targetName) => {
    const targetNode = nodes[targetName]?.node;
    const functionNodes = [];
    let currentY = startY;

    // Create function nodes
    sourceFunctions.forEach(({ sourceNodeId, funcConfig }) => {
      const sourceNode = nodes[sourceNodeId].node;
      const functionNode = new PipecatFunctionNode();
      functionNode.properties.function = funcConfig.function;
      functionNode.properties.isTerminal = !targetNode;

      functionNode.pos = [
        sourceNode.pos[0] + nodeSpacing.horizontal / 2,
        currentY,
      ];
      graph.add(functionNode);
      sourceNode.connect(0, functionNode, 0);
      functionNodes.push(functionNode);
      currentY += nodeSpacing.vertical;
    });

    // Handle merging if needed
    if (functionNodes.length > 1 && targetNode) {
      const mergeNode = createMergeNode(functionNodes, targetNode);

      functionNodes.forEach((functionNode, index) => {
        functionNode.connect(0, mergeNode, index);
      });

      mergeNode.connect(0, targetNode, 0);
    } else if (targetNode) {
      functionNodes[0].connect(0, targetNode, 0);
    }
  });

  // Center the graph
  graph.arrange();
  graph.setDirtyCanvas(true, true);
}
