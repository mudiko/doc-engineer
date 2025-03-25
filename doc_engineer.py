#!/usr/bin/env python3
"""
Doc Engineer CLI

A command-line interface for the document generation system.
"""

import os
import argparse
from typing import Optional

from core.document_generator import DocumentGenerator


def setup_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Doc Engineer - Generate well-structured documents on any topic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "title", type=str, nargs="?", default=None, help="Title of the document to generate"
    )

    # Optional arguments
    parser.add_argument(
        "--sections", type=int, default=5, help="Number of main sections to generate"
    )
    parser.add_argument(
        "--pages", type=int, default=5, help="Approximate length in pages (1 page ≈ 500 words)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="academic",
        choices=["academic", "report", "blog"],
        help="Template to use for document formatting",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "html", "text"],
        help="Output format",
    )
    parser.add_argument(
        "--output", type=str, default="generated_document.md", help="Output file path"
    )
    parser.add_argument(
        "--api-key", type=str, help="Google API key (overrides environment variable)"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use mock provider instead of Gemini API"
    )

    return parser


def get_model_provider(use_mock: bool, api_key: Optional[str] = None):
    """Get the appropriate model provider based on arguments."""
    if use_mock:
        from core.modules.content_generator import MockProvider

        print("Using mock provider for demonstration purposes")
        return MockProvider()

    # Validate API key exists
    if not api_key:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print(
            "ERROR: No API key provided. Please set GOOGLE_API_KEY in .env file or use --api-key option."
        )
        print("Alternatively, use --mock to run with mock data for demonstration purposes.")
        return None

    try:
        from core.modules.content_generator import GeminiProvider

        print(f"Initializing Gemini provider with API key: {api_key[:5]}...")
        return GeminiProvider(api_key=api_key, model_name="gemini-2.0-flash-thinking-exp-01-21")
    except ImportError:
        print("Error: Google Generative AI package not found.")
        print("Please install it with: poetry add google-generativeai")
        print("Or use --mock to run with mock data for demonstration.")
        return None
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        print("Please check your API key and permissions.")
        print("You can use --mock flag to run with mock data for demonstration purposes.")
        return None


def main():
    """Main function for the Doc Engineer CLI."""
    # Parse command line arguments
    parser = setup_parser()
    args = parser.parse_args()

    # Prompt for title if not provided
    if not args.title:
        args.title = input("Enter document title: ")

    # Get model provider
    model_provider = get_model_provider(args.mock, args.api_key)
    if not model_provider:
        return

    # Initialize generator
    generator = DocumentGenerator(model_provider=model_provider)

    # Generate document
    print(f"\nDoc Engineer - Generating document: '{args.title}'")
    print(f"Using template: {args.template}")
    print(f"Output format: {args.format}")

    # Calculate words based on pages
    total_words = args.pages * 500  # Standard approximation

    try:
        generator.generate_document(
            title=args.title,
            num_sections=args.sections,
            template_name=args.template,
            output_format=args.format,
            output_path=args.output,
            target_length_words=total_words,
        )

        print(f"\nDocument successfully generated!")
        print(f"Saved to: {args.output}")
    except Exception as e:
        print(f"Error generating document: {e}")
        print("You can use --mock flag to run with mock data for demonstration purposes.")


if __name__ == "__main__":
    main()
