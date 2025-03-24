#!/usr/bin/env python3
"""
Document Generator - Example with Verbose Output

This script demonstrates the document generation process with detailed output
showing all steps using the mock provider for testing.
"""

import os
import argparse

from core.document_generator import DocumentGenerator
from core.modules.content_generator import MockProvider

def main():
    """Main function to demonstrate document generation with verbose output."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a document with detailed progress output")
    parser.add_argument("--title", type=str, default="The Impact of Artificial Intelligence on Healthcare",
                        help="Title of the document to generate")
    parser.add_argument("--sections", type=int, default=5,
                        help="Number of main sections to generate (default: 5)")
    parser.add_argument("--template", type=str, default="academic",
                        choices=["academic", "report", "blog"],
                        help="Template to use for document formatting (default: academic)")
    parser.add_argument("--format", type=str, default="markdown",
                        choices=["markdown", "html", "text"],
                        help="Output format (default: markdown)")
    parser.add_argument("--output", type=str, default="generated_document.md",
                        help="Output file path (default: generated_document.md)")
    
    args = parser.parse_args()
    
    print("\n==============================================")
    print("=== Document Generator Process Demonstration ===")
    print("==============================================\n")
    
    print("Using mock provider for demonstration...")
    # Create a MockProvider and DocumentGenerator
    model_provider = MockProvider()
    generator = DocumentGenerator(model_provider=model_provider)
    
    # Generate document with detailed output
    print(f"\nGenerating document: '{args.title}'")
    print(f"Template: {args.template}")
    print(f"Format: {args.format}")
    print(f"Number of sections: {args.sections}")
    print("-" * 50)
    
    document = generator.generate_document(
        title=args.title,
        num_sections=args.sections,
        template_name=args.template,
        output_format=args.format,
        output_path=args.output
    )
        
    print(f"Full document saved to: {args.output}")
    print("\nProcess completed successfully!")
    
if __name__ == "__main__":
    main() 