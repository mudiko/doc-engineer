#!/usr/bin/env python3
"""
Simple Document Generator Example

This script demonstrates how to use the content generation components directly.
"""

import os
from dotenv import load_dotenv
import argparse

from core.modules.content_generator import GeminiProvider, MockProvider, ContentGenerator
from core.modules.document_parser import Section, GeneratedSection

# Load environment variables
load_dotenv()

def generate_content(topic: str, use_mock: bool = False, api_key: str = None):
    """Generate content about a topic."""
    # Setup the content generator with the appropriate provider
    if use_mock:
        print("Using mock provider...")
        provider = MockProvider()
    else:
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY or use --api-key option.")
        print(f"Using Gemini flash-thinking model with API key: {api_key[:5]}...")
        provider = GeminiProvider(api_key=api_key, model_name="gemini-2.0-flash-thinking-exp-01-21")
    
    content_generator = ContentGenerator(provider)
    
    # Create a sample section
    section = Section(
        title="Introduction to " + topic,
        description="An overview of " + topic,
        subsections=["Background", "Key Concepts", "Importance"],
        estimated_length=500
    )
    
    # Generate content for the section
    print(f"Generating content for: {section.title}")
    generated_section = content_generator.generate_section_content(
        title=topic,
        section=section
    )
    
    return f"# {generated_section.title}\n\n{generated_section.content}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate content for a specific topic")
    parser.add_argument("--topic", type=str, default="The Impact of Artificial Intelligence on Healthcare",
                        help="Topic to generate content about")
    parser.add_argument("--mock", action="store_true", help="Use mock provider instead of Gemini API")
    parser.add_argument("--api-key", type=str, help="Google API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    # Example topic
    topic = args.topic
    
    print(f"Generating content about: {topic}")
    print("-" * 50)
    
    try:
        content = generate_content(topic, use_mock=args.mock, api_key=args.api_key)
        print(content)
    except Exception as e:
        print(f"Error generating content: {e}")
        print("Please check your API key and permissions.")
        print("You can use --mock flag to use the mock provider for demonstration.")

if __name__ == "__main__":
    main() 