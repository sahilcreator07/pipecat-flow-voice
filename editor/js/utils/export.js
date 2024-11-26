/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * Generates a flow configuration from the graph
 * @param {LGraph} graphInstance - The LiteGraph instance
 * @returns {FlowConfig} The generated flow configuration
 * @throws {Error} If the graph is invalid
 */
export function generateFlowConfig(graphInstance) {
  if (!graphInstance) {
    throw new Error("No graph instance provided");
  }

  if (!graphInstance._nodes || !Array.isArray(graphInstance._nodes)) {
    throw new Error("No nodes found in the graph");
  }

  const nodes = graphInstance._nodes;
  let startNode = nodes.find(
    (node) => node.constructor.name === "PipecatStartNode",
  );

  if (!startNode) {
    throw new Error("No start node found in the flow");
  }

  /**
   * Finds all functions connected to a node
   * @param {LGraphNode} node - The node to find functions for
   * @returns {Array<Object>} Array of function configurations
   */
  function findConnectedFunctions(node) {
    const functions = [];

    if (node.outputs && node.outputs[0] && node.outputs[0].links) {
      node.outputs[0].links.forEach((linkId) => {
        const link = graphInstance.links[linkId];
        if (!link) return;

        const targetNode = nodes.find((n) => n.id === link.target_id);
        if (!targetNode) return;

        if (targetNode.constructor.name === "PipecatFunctionNode") {
          functions.push({
            type: "function",
            function: targetNode.properties.function,
          });
        } else if (targetNode.constructor.name === "PipecatMergeNode") {
          // Find where this merge node connects to
          const mergeOutput = targetNode.outputs[0].links?.[0];
          if (!mergeOutput) return;

          const mergeLink = graphInstance.links[mergeOutput];
          if (!mergeLink) return;

          const finalNode = nodes.find((n) => n.id === mergeLink.target_id);
          if (!finalNode) return;

          // Find all functions that connect to this merge node
          const connectedFunctions = nodes.filter(
            (n) =>
              n.constructor.name === "PipecatFunctionNode" &&
              n.outputs[0].links?.some((l) => {
                const funcLink = graphInstance.links[l];
                return funcLink && funcLink.target_id === targetNode.id;
              }),
          );

          // Add all functions with their correct target
          connectedFunctions.forEach((funcNode) => {
            functions.push({
              type: "function",
              function: funcNode.properties.function,
            });
          });
        }
      });
    }

    return functions;
  }

  // Build the flow configuration using the start node's title as initial_node
  const flowConfig = {
    initial_node: startNode.title,
    nodes: {},
  };

  // Process all nodes
  nodes.forEach((node) => {
    if (
      node.constructor.name === "PipecatFunctionNode" ||
      node.constructor.name === "PipecatMergeNode"
    ) {
      return;
    }

    // Use node.title directly as the node ID
    flowConfig.nodes[node.title] = {
      messages: node.properties.messages,
      functions: findConnectedFunctions(node),
    };

    if (node.properties.pre_actions?.length > 0) {
      flowConfig.nodes[node.title].pre_actions = node.properties.pre_actions;
    }
    if (node.properties.post_actions?.length > 0) {
      flowConfig.nodes[node.title].post_actions = node.properties.post_actions;
    }
  });

  return flowConfig;
}
