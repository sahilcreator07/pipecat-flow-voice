import { generateFlowConfig } from '../utils/export.js';
import { createFlowFromConfig } from '../utils/import.js';
import { validateFlow } from '../utils/validation.js';

export class Toolbar {
  constructor(graph) {
    this.graph = graph;
    this.setupButtons();
  }

  setupButtons() {
    document.getElementById('new-flow').onclick = () => this.handleNew();
    document.getElementById('import-flow').onclick = () => this.handleImport();
    document.getElementById('export-flow').onclick = () => this.handleExport();
  }

  handleNew() {
    this.graph.clear();
  }

  handleImport() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          // Clean the input string
          const cleanInput = event.target.result
            .replace(/[\u0000-\u001F\u007F-\u009F]/g, '')
            .replace(/\r\n/g, '\n')
            .replace(/\r/g, '\n');

          console.debug('Cleaned input:', cleanInput);

          const flowConfig = JSON.parse(cleanInput);
          console.debug('Parsed config:', flowConfig);

          // Validate imported flow
          const validation = validateFlow(flowConfig);
          if (!validation.valid) {
            console.error('Flow validation errors:', validation.errors);
            if (
              !confirm('Imported flow has validation errors. Import anyway?')
            ) {
              return;
            }
          }

          createFlowFromConfig(this.graph, flowConfig);
          console.debug('Successfully imported flow configuration');
        } catch (error) {
          console.error('Error importing flow:', error);
          console.error('Error details:', {
            message: error.message,
            position: error.position,
            stack: error.stack,
          });
          alert('Error importing flow: ' + error.message);
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }

  handleExport() {
    try {
      const flowConfig = generateFlowConfig(this.graph);

      // Validate before export
      const validation = validateFlow(flowConfig);
      if (!validation.valid) {
        console.error('Flow validation errors:', validation.errors);
        if (!confirm('Flow has validation errors. Export anyway?')) {
          return;
        }
      }

      console.debug('Generated Flow Configuration:');
      console.debug(JSON.stringify(flowConfig, null, 2));

      // Generate timestamp
      const timestamp = new Date()
        .toISOString()
        .replace(/[:.]/g, '-')
        .replace('T', '_')
        .slice(0, -5);

      // Create a clean JSON string
      const cleanJson = JSON.stringify(flowConfig, null, 2)
        .replace(/\\n/g, '\n')
        .replace(/\\"/g, '"')
        .replace(/"\[\s*\{/g, '[{')
        .replace(/\}\s*\]"/g, '}]')
        .replace(/,(\s*}])/g, '$1')
        .replace(/\n\s*,\s*}/g, '\n}')
        .replace(/}]\s*,\s*}/g, '}]}')
        .replace(/\n\s{20,}/g, '\n    ')
        .trim();

      const blob = new Blob([cleanJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `flow_config_${timestamp}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error generating flow configuration:', error);
      alert('Error generating flow configuration: ' + error.message);
    }
  }
}
