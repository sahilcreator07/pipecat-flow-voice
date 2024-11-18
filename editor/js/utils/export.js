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
   * Gets the ID for a node
   * @param {LGraphNode} node - The node to get ID for
   * @returns {string|null} The node ID or null
   */
  function getNodeId(node) {
    if (node.constructor.name === "PipecatStartNode") return "start";
    if (node.constructor.name === "PipecatEndNode") return "end";
    const incomingFunction = findIncomingFunction(node);
    return incomingFunction ? incomingFunction.properties.function.name : null;
  }

  /**
   * Finds the incoming function node
   * @param {LGraphNode} node - The node to find incoming function for
   * @returns {LGraphNode|null} The incoming function node or null
   */
  function findIncomingFunction(node) {
    return nodes.find(
      (n) =>
        n.constructor.name === "PipecatFunctionNode" &&
        n.outputs[0].links &&
        n.outputs[0].links.some((linkId) => {
          const link = graphInstance.links[linkId];
          return link && link.target_id === node.id;
        }),
    );
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
        if (link) {
          const targetNode = nodes.find((n) => n.id === link.target_id);
          if (targetNode.constructor.name === "PipecatFunctionNode") {
            functions.push({
              type: "function",
              function: targetNode.properties.function,
            });
          } else if (targetNode.constructor.name === "PipecatMergeNode") {
            if (targetNode.outputs[0].links) {
              const mergeLink =
                graphInstance.links[targetNode.outputs[0].links[0]];
              if (mergeLink) {
                const finalNode = nodes.find(
                  (n) => n.id === mergeLink.target_id,
                );
                if (finalNode.constructor.name === "PipecatEndNode") {
                  functions.push({
                    type: "function",
                    function: {
                      name: "end",
                      description:
                        "Complete the order (use only after user confirms)",
                      parameters: { type: "object", properties: {} },
                    },
                  });
                }
              }
            }
          }
        }
      });
    }

    return functions;
  }

  // Build the flow configuration
  const flowConfig = {
    initial_node: "start",
    nodes: {},
  };

  // Process each node
  nodes.forEach((node) => {
    if (
      node.constructor.name === "PipecatFunctionNode" ||
      node.constructor.name === "PipecatMergeNode"
    ) {
      return;
    }

    const nodeId = getNodeId(node);
    if (!nodeId) return;

    const functions = findConnectedFunctions(node);

    flowConfig.nodes[nodeId] = {
      messages: node.properties.messages,
      functions: functions,
    };

    if (node.properties.pre_actions?.length > 0) {
      flowConfig.nodes[nodeId].pre_actions = node.properties.pre_actions;
    }
    if (node.properties.post_actions?.length > 0) {
      flowConfig.nodes[nodeId].post_actions = node.properties.post_actions;
    }
  });

  return flowConfig;
}
