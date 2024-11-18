/**
 * Copyright (c) 2024, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * Formats actions for display
 * @param {Array<{type: string, text?: string}>} actions - Array of actions to format
 * @returns {string} Formatted string representation of actions
 */
export function formatActions(actions) {
  return actions
    .map((action) => {
      if (action.text) {
        return `${action.type}: "${action.text}"`;
      }
      const { type, ...rest } = action;
      return `${type}: ${JSON.stringify(rest)}`;
    })
    .join("\n");
}
