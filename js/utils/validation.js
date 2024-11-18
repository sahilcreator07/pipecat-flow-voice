/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * @typedef {NodeProperties}
 * @typedef {FunctionDefinition}
 */

/**
 * @typedef {Object} FlowConfig
 * @property {string} initial_node - ID of the starting node
 * @property {Object.<string, NodeProperties>} nodes - Map of node IDs to node configurations
 */

/**
 * @typedef {Object} ValidationResult
 * @property {boolean} valid - Whether the flow is valid
 * @property {string[]} errors - Array of validation error messages
 */

/**
 * Validates flow configurations
 */
export class FlowValidator {
  /**
   * Creates a new flow validator
   * @param {FlowConfig} flowConfig - The flow configuration to validate
   */
  constructor(flowConfig) {
    this.flow = flowConfig;
    this.errors = [];
  }

  /**
   * Performs all validation checks
   * @returns {string[]} Array of validation error messages
   */
  validate() {
    this.errors = [];

    this._validateInitialNode();
    this._validateNodeReferences();
    this._validateNodeContents();

    return this.errors;
  }

  /**
   * Validates the initial node configuration
   * @private
   */
  _validateInitialNode() {
    if (!this.flow.initial_node) {
      this.errors.push("Initial node must be specified");
    } else if (!this.flow.nodes[this.flow.initial_node]) {
      this.errors.push(
        `Initial node '${this.flow.initial_node}' not found in nodes`,
      );
    }
  }

  /**
   * Determines if a function is terminal based on its parameters
   * @param {string} funcName - Name of the function to check
   * @returns {boolean} Whether the function is terminal
   * @private
   */
  isTerminalFunction(funcName) {
    // Find the function definition in any node
    for (const node of Object.values(this.flow.nodes)) {
      const func = node.functions?.find((f) => f.function.name === funcName);
      if (func) {
        // Terminal functions are those that:
        // 1. Have parameters with properties (collecting data)
        // 2. Have required fields (must collect specific data)
        // 3. Have constraints (enum, min/max, etc.)
        const params = func.function.parameters;
        const hasProperties = Object.keys(params.properties || {}).length > 0;
        const hasRequired =
          Array.isArray(params.required) && params.required.length > 0;
        const hasConstraints = Object.values(params.properties || {}).some(
          (prop) =>
            prop.enum ||
            prop.minimum !== undefined ||
            prop.maximum !== undefined,
        );

        return hasProperties && (hasRequired || hasConstraints);
      }
    }
    return false;
  }

  /**
   * Validates node references in functions
   * @private
   */
  _validateNodeReferences() {
    Object.entries(this.flow.nodes).forEach(([nodeId, node]) => {
      if (node.functions) {
        const functionNames = node.functions
          .map((func) => func.function?.name)
          .filter(Boolean);

        functionNames.forEach((funcName) => {
          if (
            !this.isTerminalFunction(funcName) &&
            funcName !== "end" &&
            !this.flow.nodes[funcName]
          ) {
            this.errors.push(
              `Node '${nodeId}' has function '${funcName}' that doesn't reference a valid node`,
            );
          }
        });
      }
    });
  }

  /**
   * Validates node contents
   * @private
   */
  _validateNodeContents() {
    Object.entries(this.flow.nodes).forEach(([nodeId, node]) => {
      // Validate messages
      if (!node.messages || node.messages.length === 0) {
        this.errors.push(`Node '${nodeId}' must have at least one message`);
      }

      node.messages?.forEach((msg) => {
        if (!msg.role) {
          this.errors.push(`Message in node '${nodeId}' missing 'role'`);
        }
        if (!msg.content) {
          this.errors.push(`Message in node '${nodeId}' missing 'content'`);
        }
      });

      // Validate functions
      node.functions?.forEach((func) => {
        if (!func.function) {
          this.errors.push(
            `Function in node '${nodeId}' missing 'function' object`,
          );
        } else if (!func.function.name) {
          this.errors.push(`Function in node '${nodeId}' missing 'name'`);
        }
      });
    });
  }
}

/**
 * Validates a flow configuration
 * @param {FlowConfig} flowConfig - The flow configuration to validate
 * @returns {ValidationResult} Validation result
 */
export function validateFlow(flowConfig) {
  const validator = new FlowValidator(flowConfig);
  return {
    valid: validator.validate().length === 0,
    errors: validator.errors,
  };
}
