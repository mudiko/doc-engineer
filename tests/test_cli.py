#!/usr/bin/env python3
"""
Test CLI Interface

Tests for the command-line interface functionality.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
import tempfile
import shutil

# Import the CLI module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import doc_engineer


class TestCLI:
    """Test the CLI functionality."""
    
    def test_setup_parser(self):
        """Test the argument parser setup."""
        parser = doc_engineer.setup_parser()
        
        # Check that the parser has the expected arguments
        args = parser.parse_args(['Test Title'])
        assert args.title == 'Test Title'
        assert args.sections == 5
        assert args.pages == 5
        assert args.template == 'academic'
        assert args.format == 'markdown'
        assert args.output == 'generated_document.md'
        assert not args.mock
        assert args.api_key is None
    
    def test_get_model_provider_mock(self):
        """Test getting the mock provider."""
        provider = doc_engineer.get_model_provider(True)
        assert provider is not None
        from core.modules.content_generator import MockProvider
        assert isinstance(provider, MockProvider)
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-api-key"})
    def test_get_model_provider_with_env_var(self):
        """Test getting the provider using environment variable."""
        with patch('doc_engineer.GeminiProvider') as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance
            
            provider = doc_engineer.get_model_provider(False)
            
            # Check that provider was created with correct parameters
            mock_provider.assert_called_once()
            assert provider == mock_instance
    
    def test_get_model_provider_with_direct_key(self):
        """Test getting the provider using a directly provided key."""
        with patch('doc_engineer.GeminiProvider') as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance
            
            provider = doc_engineer.get_model_provider(False, "direct-api-key")
            
            # Check that provider was created with correct parameters
            mock_provider.assert_called_once()
            assert provider == mock_instance
    
    def test_get_model_provider_no_key(self):
        """Test getting the provider with no API key."""
        # Temporarily unset GOOGLE_API_KEY if it exists
        old_key = os.environ.pop('GOOGLE_API_KEY', None)
        try:
            # Mock dotenv.load_dotenv to do nothing
            with patch('doc_engineer.load_dotenv'):
                provider = doc_engineer.get_model_provider(False)
                assert provider is None
        finally:
            # Restore the original environment
            if old_key is not None:
                os.environ['GOOGLE_API_KEY'] = old_key
    
    @patch('doc_engineer.setup_parser')
    @patch('doc_engineer.get_model_provider')
    @patch('doc_engineer.DocumentGenerator')
    def test_main_with_mock(self, mock_generator_class, mock_get_provider, mock_setup_parser):
        """Test the main function with mock provider."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.title = "Test Document"
        mock_args.mock = True
        mock_args.api_key = None
        mock_args.sections = 3
        mock_args.pages = 2
        mock_args.template = "academic"
        mock_args.format = "markdown"
        mock_args.output = "test_output.md"
        
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        # Run the main function
        with patch('sys.stdout', new=StringIO()) as captured_output:
            doc_engineer.main()
            
            # Check that the right methods were called
            mock_setup_parser.assert_called_once()
            mock_get_provider.assert_called_once_with(True, None)
            mock_generator_class.assert_called_once_with(model_provider=mock_provider)
            
            # Check that generate_document was called with correct parameters
            mock_generator.generate_document.assert_called_once()
            call_args = mock_generator.generate_document.call_args[1]
            assert call_args['title'] == "Test Document"
            assert call_args['num_sections'] == 3
            assert call_args['template_name'] == "academic"
            assert call_args['output_format'] == "markdown"
            assert call_args['output_path'] == "test_output.md"
            assert call_args['target_length_words'] == 1000  # 2 pages * 500 words per page
            
            # Check console output
            output = captured_output.getvalue()
            assert "Generating document" in output
    
    @patch('doc_engineer.input', return_value="Prompted Title")
    @patch('doc_engineer.setup_parser')
    @patch('doc_engineer.get_model_provider')
    @patch('doc_engineer.DocumentGenerator')
    def test_main_with_title_prompt(self, mock_generator_class, mock_get_provider, 
                                   mock_setup_parser, mock_input):
        """Test the main function with title prompt."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.title = None  # No title provided, should prompt
        mock_args.mock = True
        mock_args.api_key = None
        mock_args.sections = 3
        mock_args.pages = 2
        mock_args.template = "academic"
        mock_args.format = "markdown"
        mock_args.output = "test_output.md"
        
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        # Run the main function
        with patch('sys.stdout', new=StringIO()):
            doc_engineer.main()
            
            # Check that input was called for title prompt
            mock_input.assert_called_once()
            
            # Check that generate_document was called with prompted title
            call_args = mock_generator.generate_document.call_args[1]
            assert call_args['title'] == "Prompted Title" 