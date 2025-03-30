#!/usr/bin/env python3
"""
Test Document Generator

Tests for the document generator functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from tempfile import NamedTemporaryFile

# Updated imports after refactoring
from core.orchestration.document_generator import DocumentGenerator
from core.generation.content_generator import MockProvider, ContentGenerator, GeminiProvider # Added ContentGenerator, GeminiProvider
from core.planning.document_planner import DocumentPlanner # Added
from core.citations.citation_manager import CitationManager # Added


class TestDocumentGenerator:
    """Test the document generator functionality."""

    @patch("core.planning.document_planner.DocumentPlanner")
    @patch("core.citations.citation_manager.CitationManager")
    def test_document_generation_with_mock(self, mock_citation_manager_class, mock_planner_class):
        """Test document generation with mock provider."""
        # Setup mocks for dependencies
        mock_planner = MagicMock()
        mock_planner_class.return_value = mock_planner
        # Mock the create_plan method to return a basic DocumentPlan structure
        mock_plan = MagicMock()
        mock_plan.sections = [MagicMock(title="Introduction"), MagicMock(title="Body"), MagicMock(title="Conclusion")]
        mock_plan.introduction = mock_plan.sections[0]
        mock_plan.main_sections = [mock_plan.sections[1]]
        mock_plan.conclusion = mock_plan.sections[2]
        mock_plan.total_estimated_length = 1000
        mock_planner.create_plan.return_value = mock_plan

        mock_citation_manager = MagicMock()
        mock_citation_manager_class.return_value = mock_citation_manager
        mock_citation_manager.search_papers.return_value = [] # No citations for basic mock test
        mock_citation_manager.select_citations_for_section.return_value = []

        # Create the actual ContentGenerator with a MockProvider
        mock_model_provider = MockProvider()
        content_generator = ContentGenerator(model_provider=mock_model_provider)

        # Instantiate DocumentGenerator with mocks and real ContentGenerator
        generator = DocumentGenerator(
            document_planner=mock_planner,
            content_generator=content_generator,
            citation_manager=mock_citation_manager
        )

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

    @patch("core.planning.document_planner.DocumentPlanner")
    @patch("core.citations.citation_manager.CitationManager")
    def test_different_templates(self, mock_citation_manager_class, mock_planner_class):
        """Test document generation with different templates."""
        # Setup mocks
        mock_planner = MagicMock()
        mock_planner_class.return_value = mock_planner
        mock_plan = MagicMock()
        mock_plan.sections = [MagicMock(title="Intro"), MagicMock(title="Body"), MagicMock(title="Concl")]
        mock_plan.introduction = mock_plan.sections[0]
        mock_plan.main_sections = [mock_plan.sections[1]]
        mock_plan.conclusion = mock_plan.sections[2]
        mock_plan.total_estimated_length = 1000
        mock_planner.create_plan.return_value = mock_plan

        mock_citation_manager = MagicMock()
        mock_citation_manager_class.return_value = mock_citation_manager
        mock_citation_manager.search_papers.return_value = []
        mock_citation_manager.select_citations_for_section.return_value = []

        mock_model_provider = MockProvider()
        content_generator = ContentGenerator(model_provider=mock_model_provider)

        generator = DocumentGenerator(
            document_planner=mock_planner,
            content_generator=content_generator,
            citation_manager=mock_citation_manager
        )

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

    @patch("core.planning.document_planner.DocumentPlanner")
    @patch("core.citations.citation_manager.CitationManager")
    def test_different_output_formats(self, mock_citation_manager_class, mock_planner_class):
        """Test document generation with different output formats."""
        # Setup mocks
        mock_planner = MagicMock()
        mock_planner_class.return_value = mock_planner
        mock_plan = MagicMock()
        mock_plan.sections = [MagicMock(title="Intro"), MagicMock(title="Body"), MagicMock(title="Concl")]
        mock_plan.introduction = mock_plan.sections[0]
        mock_plan.main_sections = [mock_plan.sections[1]]
        mock_plan.conclusion = mock_plan.sections[2]
        mock_plan.total_estimated_length = 1000
        mock_planner.create_plan.return_value = mock_plan

        mock_citation_manager = MagicMock()
        mock_citation_manager_class.return_value = mock_citation_manager
        mock_citation_manager.search_papers.return_value = []
        mock_citation_manager.select_citations_for_section.return_value = []

        mock_model_provider = MockProvider()
        content_generator = ContentGenerator(model_provider=mock_model_provider)

        generator = DocumentGenerator(
            document_planner=mock_planner,
            content_generator=content_generator,
            citation_manager=mock_citation_manager
        )

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
    @patch("core.citations.citation_manager.CitationManager") # Mock citation manager for API test too
    def test_with_real_api(self, mock_citation_manager_class):
        """Test document generation with real API (skipped if no API key)."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
             pytest.skip("GOOGLE_API_KEY not set")

        # Use real providers/managers where possible, mock others
        try:
             model_provider = GeminiProvider(api_key=api_key, model_name="gemini-pro") # Use standard model name
             content_generator = ContentGenerator(model_provider=model_provider)
             document_planner = DocumentPlanner() # Use real planner
        except Exception as e:
             pytest.skip(f"Skipping real API test due to initialization error: {e}")


        mock_citation_manager = MagicMock()
        mock_citation_manager_class.return_value = mock_citation_manager
        mock_citation_manager.search_papers.return_value = [] # Keep citations mocked for speed/simplicity
        mock_citation_manager.select_citations_for_section.return_value = []

        generator = DocumentGenerator(
            document_planner=document_planner,
            content_generator=content_generator,
            citation_manager=mock_citation_manager
        )

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
