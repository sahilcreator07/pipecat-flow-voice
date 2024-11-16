# Pipecat Flow Editor

A visual editor for Pipecat conversation flows.

## Backend Development

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running the Server

```bash
python -m backend.main
```

The server will start at `http://localhost:8000`

### API Endpoints

- `POST /api/import` - Convert flow configuration to visual representation
- `POST /api/export` - Convert visual representation to flow configuration
- `POST /api/validate` - Validate flow configuration
- `GET /api/templates` - Get predefined node templates

### Testing

With the server running, execute individual tests:

```bash
python -m backend.tests.test_import
python -m backend.tests.test_validation
python -m backend.tests.test_templates
```

## Frontend Development

### Setup

```bash
cd frontend
npm init -y
npm install litegraph.js
```

### Running the Frontend

```bash
python frontend/server.py
```

The frontend will be available at `http://localhost:8080`
