#!/usr/bin/env python3
"""
Test Content Generator

Tests for the core content generation components.
"""

import os
import pytest
from unittest.mock import patch

from core.modules.content_generator import GeminiProvider, MockProvider, ContentGenerator
from core.modules.document_parser import Section, GeneratedSection


class TestContentGenerator:
    """Test the content generator functionality."""

    def test_mock_provider(self):
        """Test content generation with mock provider."""
        provider = MockProvider()
        content_generator = ContentGenerator(provider)

        section = Section(
            title="Introduction to Test Topic",
            description="An overview of Test Topic",
            subsections=["Background", "Key Concepts", "Importance"],
            estimated_length=500,
        )

        # Generate content for the section
        generated_section = content_generator.generate_section_content(
            title="Test Topic", section=section
        )

        # Check if content was generated
        assert generated_section.title == "Introduction to Test Topic"
        assert len(generated_section.content) > 0
        assert "mock content" in generated_section.content.lower()

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="No API key available")
    def test_gemini_provider(self):
        """Test content generation with Gemini provider (requires API key)."""
        api_key = os.getenv("GOOGLE_API_KEY")
        provider = GeminiProvider(api_key=api_key, model_name="gemini-2.0-flash-thinking-exp-01-21")
        content_generator = ContentGenerator(provider)

        section = Section(
            title="Introduction to AI",
            description="An overview of Artificial Intelligence",
            subsections=["Background", "Key Concepts", "Importance"],
            estimated_length=500,
        )

        # Generate content for the section
        generated_section = content_generator.generate_section_content(
            title="Artificial Intelligence", section=section
        )

        # Check if content was generated
        assert generated_section.title == "Introduction to AI"
        assert len(generated_section.content) > 0

    def test_document_plan_creation(self):
        """Test document plan creation."""
        provider = MockProvider()
        content_generator = ContentGenerator(provider)

        # Create a document plan
        document_plan = content_generator.create_document_plan(
            title="Test Document", num_sections=5, target_length_words=2500
        )

        # Check if plan was created properly
        assert document_plan.topic == "Test Document"
        assert len(document_plan.sections) > 0
        assert document_plan.total_estimated_length > 0

    def test_section_evaluation(self):
        """Test section evaluation and critique."""
        provider = MockProvider()
        content_generator = ContentGenerator(provider)

        # Create some test sections
        sections = [
            GeneratedSection(
                title="Introduction", content="This is a mock introduction.", subsections=[]
            ),
            GeneratedSection(
                title="Main Section", content="This is a mock main section.", subsections=[]
            ),
        ]

        # Evaluate the document sections
        critiques = content_generator.evaluate_document_sections(
            title="Test Document", sections=sections
        )

        # Check if critiques were generated
        assert critiques is not None
