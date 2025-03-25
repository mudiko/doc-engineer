#!/usr/bin/env python3
"""
Test Document Generator

Tests for the document generator functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from tempfile import NamedTemporaryFile

from core.document_generator import DocumentGenerator
from core.modules.content_generator import MockProvider


class TestDocumentGenerator:
    """Test the document generator functionality."""

    def test_document_generation_with_mock(self):
        """Test document generation with mock provider."""
        model_provider = MockProvider()
        generator = DocumentGenerator(model_provider=model_provider)

        with NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            output_path = tmp.name

        try:
            # Generate document
            document = generator.generate_document(
                title="Test Document",
                num_sections=5,
                template_name="academic",
                output_format="markdown",
                output_path=output_path,
            )

            # Check if file was created
            assert os.path.exists(output_path)

            # Check file content
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "Test Document" in content
                assert "Introduction" in content
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_different_templates(self):
        """Test document generation with different templates."""
        model_provider = MockProvider()
        generator = DocumentGenerator(model_provider=model_provider)

        templates = ["academic", "report", "blog"]

        for template in templates:
            with NamedTemporaryFile(suffix=".md", delete=False) as tmp:
                output_path = tmp.name

            try:
                # Generate document with this template
                document = generator.generate_document(
                    title=f"Test {template.capitalize()} Document",
                    num_sections=4,
                    template_name=template,
                    output_format="markdown",
                    output_path=output_path,
                )

                # Check if file was created
                assert os.path.exists(output_path)

                # Different templates should produce different results
                with open(output_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    assert f"Test {template.capitalize()} Document" in content
            finally:
                # Clean up temp file
                if os.path.exists(output_path):
                    os.unlink(output_path)

    def test_different_output_formats(self):
        """Test document generation with different output formats."""
        model_provider = MockProvider()
        generator = DocumentGenerator(model_provider=model_provider)

        formats = ["markdown", "html", "text"]

        for output_format in formats:
            suffix = (
                ".md"
                if output_format == "markdown"
                else ".html"
                if output_format == "html"
                else ".txt"
            )

            with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                output_path = tmp.name

            try:
                # Generate document with this format
                document = generator.generate_document(
                    title=f"Test Format Document",
                    num_sections=3,
                    template_name="academic",
                    output_format=output_format,
                    output_path=output_path,
                )

                # Check if file was created
                assert os.path.exists(output_path)

                # Different formats should produce different markers
                with open(output_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if output_format == "html":
                        assert "<html" in content.lower() or "<!doctype" in content.lower()
                    elif output_format == "markdown":
                        assert "#" in content  # Markdown headers

            finally:
                # Clean up temp file
                if os.path.exists(output_path):
                    os.unlink(output_path)

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="No API key available")
    def test_with_real_api(self):
        """Test document generation with real API (skipped if no API key)."""
        from core.modules.content_generator import GeminiProvider

        api_key = os.getenv("GOOGLE_API_KEY")
        provider = GeminiProvider(api_key=api_key, model_name="gemini-2.0-flash-thinking-exp-01-21")
        generator = DocumentGenerator(model_provider=provider)

        with NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            output_path = tmp.name

        try:
            # Generate a small document (just to test basic functionality)
            document = generator.generate_document(
                title="API Test Document",
                num_sections=2,  # Keep this small to avoid long test runs
                template_name="academic",
                output_format="markdown",
                output_path=output_path,
                target_length_words=500,  # Keep this small too
            )

            # Check if file was created
            assert os.path.exists(output_path)

            # Check file content
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "API Test Document" in content
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
