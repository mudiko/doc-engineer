# Doc Engineer

[![Tests](https://github.com/yourusername/doc-engineer/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/doc-engineer/actions/workflows/tests.yml)
[![Lint](https://github.com/yourusername/doc-engineer/actions/workflows/lint.yml/badge.svg)](https://github.com/yourusername/doc-engineer/actions/workflows/lint.yml)

A powerful single-shot document generation system that leverages AI to create comprehensive, well-structured documents on any topic within seconds.

## Overview

Doc Engineer streamlines document creation by allowing users to specify their requirements through simple templates and parameters. The system handles the rest, generating complete documents in one go. Simply define what you're looking for, and Doc Engineer produces the result almost instantly.

## Features

- **Single-Shot Document Generation**: Create complete documents in one go with minimal input
- **Template-Based Approach**: Choose from prebuilt templates for various document types
- **AI-Powered Content Generation**: Generates comprehensive, structured documents on any topic
- **Customizable Document Parameters**: Specify length, structure, and complexity as needed
- **Document-Wide Coherence**: AI evaluates the entire document for consistency and flow
- **Chunking Support**: Handles generation of large documents through intelligent chunking
- **API Quota Management**: Built-in retry mechanism with exponential backoff for API rate limits
- **Robust Error Handling**: Graceful recovery from model errors and empty responses
- **Multiple Output Formats**: Export documents as Markdown, HTML, or plain text
- **Modular Architecture**: Highly extensible system for easy customization

## Coming Soon

- **Citation Support**: Automatic citation generation and management
- **Search Integration**: Find and incorporate relevant information from specified sources
- **Additional Templates**: More specialized document templates for various use cases

## Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| ✅ 100-page documents | **COMPLETED** | Generate consistent 100-page documents in a single shot with intelligent chunking |
| ⏳ Concurrent generation | PLANNED | Improve performance with parallel processing of document sections |
| ⏳ Search integration | PLANNED | Incorporate external knowledge through search capabilities |
| ⏳ Citation tools | PLANNED | Add automatic citation generation and management |
| ⏳ Additional output formats | PLANNED | Support for more document output formats (PDF, DOCX, LaTeX, AsciiDoc, reStructuredText) |
| ⏳ Template library | PLANNED | Expand the collection of document templates for specific use cases |
| ⏳ Web frontend | PLANNED | Create a user-friendly web interface for document generation and management |

## Modular Architecture

The system is designed with a modular architecture that separates concerns into distinct components:

1. **Document Parser**: Handles parsing and structural representation of documents
2. **Content Generator**: Manages AI-based content generation using various model providers
3. **Template Handler**: Provides document templates and formatting structures
4. **Output Formatter**: Handles document formatting and export to different formats

This architecture allows for easy extension and customization of each component.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/doc-engineer.git
   cd doc-engineer
   ```

2. Install with Poetry (recommended):
   ```bash
   # Install Poetry if you don't have it
   # curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

   Alternatively, use pip with a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   Create a `.env` file in the root directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

### Command Line Interface

Generate a document using the CLI:

```bash
# If using Poetry
poetry run doc-engineer "The Future of Quantum Computing"

# Or directly if in Poetry shell or other environment
./doc_engineer.py "The Future of Quantum Computing"
```

Or with custom parameters:

```bash
./doc_engineer.py "Comprehensive Guide to Machine Learning" --sections 6 --pages 15 --template report --output ml_guide.md
```

### Command Line Options

- `title`: Document title (positional argument)
- `--sections`: Number of main sections to generate (default: 5)
- `--pages`: Approximate length in pages (1 page ≈ 500 words) (default: 5)
- `--template`: Template to use for document formatting (choices: academic, report, blog; default: academic)
- `--format`: Output format (choices: markdown, html, text; default: markdown)
- `--output`: Output file path (default: generated_document.md)
- `--mock`: Use mock provider for testing without API key
- `--api-key`: Directly provide Google API key (overrides environment variable)
- `--hide-tokens`: Hide detailed token usage statistics (tokens are shown by default)

## Development

### Running Tests

Run the tests using the provided script:

```bash
./run_tests.sh
```

Or manually with Poetry:

```bash
poetry run pytest -xvs
```

To run tests with coverage reporting:

```bash
poetry run pytest -xvs --cov=. --cov-report=term-missing
```

### Linting and Code Quality

Run linters to check code quality:

```bash
./run_lint.sh
```

This will run:
- **flake8**: For basic code quality checks
- **black**: To verify code formatting
- **mypy**: For static type checking

### Continuous Integration

This project uses GitHub Actions for continuous integration, with two main workflows:

1. **Tests**: Runs the test suite on multiple Python versions
2. **Lint**: Performs code quality checks

These workflows run automatically on push to main branches and on pull requests.

### Directory Structure

```
doc-engineer/
├── core/               # Core functionality
├── tests/              # Test suite
├── .github/            # GitHub Actions workflows
├── doc_engineer.py     # CLI interface
├── pyproject.toml      # Poetry configuration
├── requirements.txt    # Pip requirements (for non-Poetry users)
├── setup.sh            # Setup script
├── run_tests.sh        # Test runner script
└── run_lint.sh         # Linting script
```

## Using the API

You can also use the document generation system programmatically:

```python
from core.document_generator import DocumentGenerator

# Initialize the document generator
generator = DocumentGenerator(api_key="your_google_api_key")

# Generate a document
document = generator.generate_document(
    title="Climate Change Mitigation Strategies",
    num_sections=4,
    template_name="report",
    output_format="html",
    output_path="climate_report.html",
    target_length_words=7500  # Target length of ~15 pages
)
```

## Extending the System

The modular architecture makes it easy to extend the system:

- Add new model providers by implementing the `ModelProvider` protocol
- Create custom templates by extending the `Template` class
- Add new output formats by implementing additional formatting options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the Google Gemini API for content generation
- Built with a focus on efficiency and usability for rapid document creation
