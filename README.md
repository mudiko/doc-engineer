# Doc Engineer

A modular document generation system that leverages AI to create well-structured, coherent documents on any topic.

## Features

- **AI-Powered Content Generation**: Generates comprehensive, structured documents on any topic
- **Document-Wide Critique & Improvement**: AI evaluates the entire document for coherence and improves each section
- **Section-by-Section Evaluation**: AI reviews each section for quality and coherence
- **Template Support**: Apply different templates for academic papers, business reports, or blog posts
- **Multiple Output Formats**: Export documents as Markdown, HTML, or plain text
- **Flexible Document Length**: Specify document length in pages (15 pages, 30 pages, etc.)
- **Modular Architecture**: Highly extensible system with clean separation of concerns

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

2. Create a virtual environment and install dependencies:
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

### Basic Usage

Generate a document with default settings:

```bash
python example.py
```

### Advanced Usage

Customize your document generation:

```bash
python example.py --title "The Future of Quantum Computing" --sections 6 --template academic --format markdown --output quantum_paper.md
```

Generate a longer document with a specific page length:

```bash
python example.py --title "Comprehensive Guide to Machine Learning" --pages 15 --template report --format markdown --output ml_guide.md
```

### Command Line Options

- `--title`: Document title (default: "The Impact of Artificial Intelligence on Healthcare")
- `--sections`: Number of main sections to generate (default: 5)
- `--pages`: Approximate length in pages (1 page ≈ 500 words)
- `--template`: Template to use for document formatting (choices: academic, report, blog; default: academic)
- `--format`: Output format (choices: markdown, html, text; default: markdown)
- `--output`: Output file path (default: generated_document.md)
- `--mock`: Use mock provider for testing without API key
- `--api-key`: Directly provide Google API key (overrides environment variable)

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
- Create custom templates by extending the `TemplateStrategy` class
- Add new output formats by implementing the `OutputFormatter` interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the Google Gemini API for content generation
- Built with inspiration from various academic paper generation systems and document engineering concepts
