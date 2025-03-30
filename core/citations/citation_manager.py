"""
Citation Manager Module

This module handles bibliographic research and citation management for Doc Engineer.
It can use either the findpapers library or Semantic Scholar for academic papers and citations.
"""

import os
import json
import datetime
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set

# Import statements moved to the methods that use them
# to ensure proper error handling and dependency flexibility

class CitationManager:
    """Manages bibliographic research and citations for document generation."""

    def __init__(
        self, 
        cache_dir: Optional[str] = None,
        scopus_api_token: Optional[str] = None,
        ieee_api_token: Optional[str] = None,
        proxy: Optional[str] = None,
        use_semantic_scholar: bool = True,
    ):
        """
        Initialize the citation manager.
        
        Args:
            cache_dir: Directory to store search results and papers (default: ~/.doc-engineer/citations)
            scopus_api_token: API token for Scopus database
            ieee_api_token: API token for IEEE database
            proxy: Proxy URL for downloading papers
            use_semantic_scholar: Whether to use Semantic Scholar instead of findpapers
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(os.getcwd(), "citations")
        else:
            self.cache_dir = cache_dir
            
        # Store API tokens and proxy
        self.scopus_api_token = scopus_api_token
        self.ieee_api_token = ieee_api_token
        self.proxy = proxy
        
        # Default parameters
        self.default_databases = ["acm", "arxiv", "pubmed", "ieee", "scopus"]
        self.default_publication_types = ["journal", "conference proceedings"]
        
        # Map to store citation keys for each document
        self.document_citations: Dict[str, Set[str]] = {}
        
        # Store the preference for citation source
        self.use_semantic_scholar = use_semantic_scholar
        if self.use_semantic_scholar:
            self.logger.info("Will attempt to use Semantic Scholar for citations")
        else:
            self.logger.info("Will attempt to use findpapers for citations")
        
        # In-memory storage for vector indexes
        self.vector_indexes = {}
        
        # In-memory storage for mock citation data
        self.mock_citations = {
            "neural_networks": [
                {
                    "bibtex_key": "krizhevsky2012imagenet",
                    "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                    "authors": ["Krizhevsky, A.", "Sutskever, I.", "Hinton, G.E."],
                    "year": "2012",
                    "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes.",
                    "database": "arXiv"
                },
                {
                    "bibtex_key": "he2016deep",
                    "title": "Deep Residual Learning for Image Recognition",
                    "authors": ["He, K.", "Zhang, X.", "Ren, S.", "Sun, J."],
                    "year": "2016",
                    "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
                    "database": "IEEE"
                },
                {
                    "bibtex_key": "goodfellow2014generative",
                    "title": "Generative Adversarial Networks",
                    "authors": ["Goodfellow, I.", "Pouget-Abadie, J.", "Mirza, M.", "Xu, B.", "Warde-Farley, D.", "Ozair, S.", "Courville, A.", "Bengio, Y."],
                    "year": "2014",
                    "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
                    "database": "arXiv"
                },
                {
                    "bibtex_key": "simonyan2014very",
                    "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition",
                    "authors": ["Simonyan, K.", "Zisserman, A."],
                    "year": "2014",
                    "abstract": "In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting.",
                    "database": "arXiv"
                },
                {
                    "bibtex_key": "lecun2015deep",
                    "title": "Deep Learning",
                    "authors": ["LeCun, Y.", "Bengio, Y.", "Hinton, G."],
                    "year": "2015",
                    "abstract": "Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction.",
                    "database": "Nature"
                }
            ],
            "computer_vision": [
                {
                    "bibtex_key": "redmon2016you",
                    "title": "You Only Look Once: Unified, Real-Time Object Detection",
                    "authors": ["Redmon, J.", "Divvala, S.", "Girshick, R.", "Farhadi, A."],
                    "year": "2016",
                    "abstract": "We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities.",
                    "database": "IEEE"
                },
                {
                    "bibtex_key": "ronneberger2015u",
                    "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
                    "authors": ["Ronneberger, O.", "Fischer, P.", "Brox, T."],
                    "year": "2015",
                    "abstract": "There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently.",
                    "database": "arXiv"
                },
                {
                    "bibtex_key": "chen2018encoder",
                    "title": "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation",
                    "authors": ["Chen, L.C.", "Zhu, Y.", "Papandreou, G.", "Schroff, F.", "Adam, H."],
                    "year": "2018",
                    "abstract": "Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information.",
                    "database": "IEEE"
                }
            ],
            "citizenship": [
                {
                    "bibtex_key": "schachar2009birthright",
                    "title": "The Birthright Lottery: Citizenship and Global Inequality",
                    "authors": ["Schachar, A."],
                    "year": "2009",
                    "abstract": "The book develops a new principle of jus nexi to challenge conventional understandings of birthright citizenship and to advance a more equitable distribution of access to citizenship.",
                    "database": "Harvard University Press"
                },
                {
                    "bibtex_key": "smith1997civic",
                    "title": "Civic Ideals: Conflicting Visions of Citizenship in U.S. History",
                    "authors": ["Smith, R.M."],
                    "year": "1997",
                    "abstract": "This book examines the conflicts and tensions that have surrounded the legal definition of American citizenship from colonial times to the present.",
                    "database": "Yale University Press"
                },
                {
                    "bibtex_key": "brubaker1992citizenship",
                    "title": "Citizenship and Nationhood in France and Germany",
                    "authors": ["Brubaker, R."],
                    "year": "1992",
                    "abstract": "This study examines the historical development and contemporary significance of citizenship and nationality laws in France and Germany, two nations with contrasting approaches to citizenship.",
                    "database": "Harvard University Press"
                },
                {
                    "bibtex_key": "aleinikoff2002citizenship",
                    "title": "Citizenship Policies for an Age of Migration",
                    "authors": ["Aleinikoff, T.A.", "Klusmeyer, D."],
                    "year": "2002",
                    "abstract": "This book analyzes citizenship laws and policies worldwide, focusing on the challenges posed by international migration.",
                    "database": "Carnegie Endowment for International Peace"
                },
                {
                    "bibtex_key": "spiro2008beyond",
                    "title": "Beyond Citizenship: American Identity After Globalization",
                    "authors": ["Spiro, P.J."],
                    "year": "2008",
                    "abstract": "This book examines the declining significance of citizenship status in an age of global migration and transnational identities.",
                    "database": "Oxford University Press"
                },
                {
                    "bibtex_key": "bosniak2006citizen",
                    "title": "The Citizen and the Alien: Dilemmas of Contemporary Membership",
                    "authors": ["Bosniak, L."],
                    "year": "2006",
                    "abstract": "This book explores the tensions and contradictions in liberal democratic approaches to citizenship and alienage.",
                    "database": "Princeton University Press"
                },
                {
                    "bibtex_key": "shuck1998citizens",
                    "title": "Citizens, Strangers, and In-Betweens: Essays on Immigration and Citizenship",
                    "authors": ["Shuck, P.H."],
                    "year": "1998",
                    "abstract": "A collection of essays examining the legal and political aspects of immigration and citizenship in the United States.",
                    "database": "Westview Press"
                },
                {
                    "bibtex_key": "kymlicka2001politics",
                    "title": "Politics in the Vernacular: Nationalism, Multiculturalism, and Citizenship",
                    "authors": ["Kymlicka, W."],
                    "year": "2001",
                    "abstract": "This book defends the importance of nationality, argues that cultural membership should be recognized as an important aspect of citizenship.",
                    "database": "Oxford University Press"
                }
            ],
            "immigration": [
                {
                    "bibtex_key": "zolberg2006nation",
                    "title": "A Nation by Design: Immigration Policy in the Fashioning of America",
                    "authors": ["Zolberg, A.R."],
                    "year": "2006",
                    "abstract": "This book provides a historical account of how immigration has continuously shaped and reshaped American society.",
                    "database": "Harvard University Press"
                },
                {
                    "bibtex_key": "calavita2005immigrants",
                    "title": "Immigrants at the Margins: Law, Race, and Exclusion in Southern Europe",
                    "authors": ["Calavita, K."],
                    "year": "2005",
                    "abstract": "This book examines the legal, social, and economic marginalization of immigrants in Spain and Italy.",
                    "database": "Cambridge University Press"
                },
                {
                    "bibtex_key": "dauvergne2008making",
                    "title": "Making People Illegal: What Globalization Means for Migration and Law",
                    "authors": ["Dauvergne, C."],
                    "year": "2008",
                    "abstract": "This book examines how globalization has affected migration patterns and the legal frameworks governing migration.",
                    "database": "Cambridge University Press"
                },
                {
                    "bibtex_key": "motomura2006americans",
                    "title": "Americans in Waiting: The Lost Story of Immigration and Citizenship in the United States",
                    "authors": ["Motomura, H."],
                    "year": "2006",
                    "abstract": "This book recovers an important conception of immigration as a transition to citizenship, which was once central to U.S. immigration policy.",
                    "database": "Oxford University Press"
                }
            ]
        }
    
    def search_papers(
        self,
        topic: str,
        document_id: str,
        since_year: Optional[int] = None,
        until_year: Optional[int] = None,
        limit: int = 30,
        limit_per_database: int = 10,
        databases: Optional[List[str]] = None,
        publication_types: Optional[List[str]] = None,
        use_mock: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on a specific topic.
        
        Args:
            topic: The topic to search for
            document_id: Unique identifier for the document
            since_year: Start year for search range
            until_year: End year for search range
            limit: Maximum number of papers to return
            limit_per_database: Maximum papers per database
            databases: List of databases to search
            publication_types: Types of publications to include
            use_mock: If True, use mock data instead of real search
            
        Returns:
            List of papers metadata
        """
        # If mock mode is explicitly requested, use the fallback method directly
        if use_mock:
            self.logger.info(f"Mock mode requested for '{topic}'")
            return self._fallback_search(topic, document_id, limit)
        
        # If Semantic Scholar is enabled, use it
        if self.use_semantic_scholar:
            self.logger.info(f"Using Semantic Scholar to search for papers on '{topic}'")
            return self._search_semantic_scholar(topic, document_id, limit, since_year)
            
        # Try using findpapers if available
        try:
            import findpapers
            self.logger.info(f"Using findpapers to search for papers on '{topic}'")
            return self._search_findpapers(
                topic, document_id, since_year, until_year, 
                limit, limit_per_database, databases, publication_types
            )
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Failed to use findpapers: {str(e)}")
            
        # Fall back to mock data if neither is available
        self.logger.warning("No citation search provider available, using mock data")
        return self._fallback_search(topic, document_id, limit)
            
    def _search_semantic_scholar(
        self, 
        topic: str, 
        document_id: str, 
        limit: int = 30,
        since_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers using Semantic Scholar and create a vector index.
        
        Args:
            topic: The search query
            document_id: Unique identifier for the document
            limit: Maximum number of papers to return
            since_year: Start year for search range
            
        Returns:
            List of papers metadata
        """
        self.logger.info(f"Attempting to search Semantic Scholar for '{topic}'")
        
        try:
            # Dynamically import required components
            from llama_index.readers.semanticscholar import SemanticScholarReader
            from llama_index.core import VectorStoreIndex, ServiceContext
            from llama_index.core.query_engine import CitationQueryEngine
            from llama_index.core.schema import Document
            from llama_index.llms.gemini import Gemini
            
            # Create Semantic Scholar reader
            s2_reader = SemanticScholarReader()
            
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Create document directory
            document_dir = os.path.join(self.cache_dir, document_id)
            os.makedirs(document_dir, exist_ok=True)
            
            # Set path for search results
            result_path = os.path.join(document_dir, "result.json")
            vector_path = os.path.join(document_dir, "vector_index")
            
            # Get documents from Semantic Scholar
            self.logger.info(f"Executing Semantic Scholar search for '{topic}'")
            semantic_documents = s2_reader.load_data(query=topic, limit=limit, full_text=True)
            self.logger.info(f"Found {len(semantic_documents)} documents in Semantic Scholar")
            
            # Convert documents to our metadata format and download PDFs if available
            papers = []
            enhanced_documents = []  # Documents with enhanced content from PDFs
            
            for doc in semantic_documents:
                # Extract metadata from the document
                metadata = doc.metadata
                
                # Create a unique bibtex key if not available
                bibtex_key = metadata.get('paperId', '')
                if not bibtex_key:
                    # Generate a key based on first author and year
                    authors = metadata.get('authors', [])
                    year = metadata.get('year', '')
                    first_author = authors[0].split()[-1] if authors else 'unknown'
                    bibtex_key = f"{first_author.lower()}{year}{str(uuid.uuid4())[:8]}"
                else:
                    # Use a shorter version of the paper ID
                    bibtex_key = bibtex_key.split('/')[-1] if '/' in bibtex_key else bibtex_key[:12]
                
                # Create a properly formatted author list
                author_list = []
                for author in metadata.get('authors', []):
                    # Format authors as "Last, First"
                    parts = author.split()
                    if len(parts) > 1:
                        last_name = parts[-1]
                        first_names = ' '.join(parts[:-1])
                        author_list.append(f"{last_name}, {first_names}")
                    else:
                        author_list.append(author)
                
                # Extract DOI if available in the URL
                url = metadata.get('url', '')
                doi = ''
                if 'doi.org' in url:
                    doi = url.split('doi.org/')[-1]
                
                # Format the paper info
                paper_info = {
                    "bibtex_key": bibtex_key,
                    "title": metadata.get('title', 'Untitled'),
                    "authors": author_list,
                    "year": str(metadata.get('year', '')),
                    "abstract": doc.text,
                    "database": metadata.get('venue', 'Semantic Scholar'),
                    "citation_count": metadata.get('citationCount', 0),
                    "url": url,
                    "doi": doi,
                    "open_access_pdf": metadata.get('openAccessPdf', None),
                    "full_text": doc.text  # Initially store the abstract
                }
                
                # Download PDF if available and extract text
                if metadata.get('openAccessPdf'):
                    self.logger.info(f"Found open access PDF for '{paper_info['title']}'")
                    pdf_text = self.download_pdf(paper_info, document_id)
                    if pdf_text:
                        # Use the full PDF text instead of just the abstract
                        paper_info["full_text"] = pdf_text
                        
                        # Create an enhanced document with full text
                        enhanced_doc = Document(
                            text=pdf_text, 
                            metadata={
                                **metadata,
                                "bibtex_key": bibtex_key,
                                "title": paper_info["title"],
                                "authors": paper_info["authors"],
                                "year": paper_info["year"],
                                "source_type": "pdf"
                            }
                        )
                        enhanced_documents.append(enhanced_doc)
                
                papers.append(paper_info)
            
            # Save the results for future use
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump({"papers": papers}, f, ensure_ascii=False, indent=2)
            
            # Create vector index of documents (use enhanced versions when available)
            try:
                # Set up Gemini as the LLM
                self.logger.info("Creating vector index of papers...")
                try:
                    # Try with gemini-pro model name first (for older versions of llama-index)
                    llm = Gemini(model="gemini-pro", temperature=0)
                except Exception as e:
                    # If that fails, try with the proper model format
                    self.logger.info(f"Retrying with full model path: {e}")
                    llm = Gemini(model="models/gemini-pro", temperature=0)
                
                service_context = ServiceContext.from_defaults(llm=llm)
                
                # Use original documents plus enhanced documents with PDF content
                all_documents = semantic_documents
                if enhanced_documents:
                    self.logger.info(f"Adding {len(enhanced_documents)} documents with full PDF text")
                    all_documents = semantic_documents + enhanced_documents
                
                # Create vector index
                vector_index = VectorStoreIndex.from_documents(
                    all_documents, service_context=service_context
                )
                
                # Store the index for later use
                self.vector_indexes[document_id] = vector_index
                
                # Save the index to disk (if llama_index supports this)
                try:
                    vector_index.storage_context.persist(persist_dir=vector_path)
                    self.logger.info(f"Vector index saved to {vector_path}")
                except Exception as e:
                    self.logger.warning(f"Could not save vector index: {str(e)}")
                
                self.logger.info("Vector index created successfully")
            except Exception as e:
                self.logger.error(f"Error creating vector index: {str(e)}")
                # Continue even if vector index creation fails
            
            # Store citation keys for this document
            self.document_citations[document_id] = set()
            for paper in papers:
                if "bibtex_key" in paper:
                    self.document_citations[document_id].add(paper["bibtex_key"])
            
            self.logger.info(f"Successfully processed {len(papers)} papers from Semantic Scholar for '{topic}'")
            return papers
            
        except Exception as e:
            self.logger.error(f"Error searching Semantic Scholar: {str(e)}")
            # If error contains ImportError or ModuleNotFoundError, indicate installation issue
            if "ImportError" in str(e) or "ModuleNotFoundError" in str(e):
                self.logger.error("Semantic Scholar modules are not properly installed.")
                self.logger.error("Please install with: pip install llama-index-readers-semanticscholar llama-index-core llama-index-llms-gemini pypdf")
            self.logger.info("Falling back to mock citation data")
            return self._fallback_search(topic, document_id, limit)
    
    def _search_findpapers(
        self,
        topic: str,
        document_id: str,
        since_year: Optional[int] = None,
        until_year: Optional[int] = None,
        limit: int = 30,
        limit_per_database: int = 10,
        databases: Optional[List[str]] = None,
        publication_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers using findpapers.
        
        Args:
            topic: The topic to search for
            document_id: Unique identifier for the document
            since_year: Start year for search range
            until_year: End year for search range
            limit: Maximum number of papers to return
            limit_per_database: Maximum papers per database
            databases: List of databases to search
            publication_types: Types of publications to include
            
        Returns:
            List of papers metadata
        """
        # Create search query in proper findpapers format
        query = self._create_search_query(topic)
        
        # Create date objects for since and until if years are provided
        since_date = None
        until_date = None
        if since_year:
            since_date = datetime.datetime(since_year, 1, 1).date()
        if until_year:
            until_date = datetime.datetime(until_year, 12, 31).date()
            
        self.logger.info(f"Searching for papers on '{topic}' (from {since_year or 'earliest'} to {until_year or 'latest'})")
        
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Create document directory
            document_dir = os.path.join(self.cache_dir, document_id)
            os.makedirs(document_dir, exist_ok=True)
            
            # Set path for search results
            result_path = os.path.join(document_dir, "result.json")
            
            # Set API tokens if provided
            if self.scopus_api_token:
                findpapers.set_scopus_token(self.scopus_api_token)
            if self.ieee_api_token:
                findpapers.set_ieee_token(self.ieee_api_token)
                
            # Execute the search using the proper findpapers API
            self.logger.info(f"Running search with query: {query}")
            
            # Call findpapers.search with the proper parameters
            search_result = findpapers.search(
                outputpath=result_path,
                query=query,
                since=since_date, 
                until=until_date,
                limit=limit,
                limit_per_database=limit_per_database,
                databases=databases,
                publication_types=publication_types,
                scopus_api_token=self.scopus_api_token,
                ieee_api_token=self.ieee_api_token,
                proxy=self.proxy,
                verbose=True  # Enable verbose to log more details
            )
            
            # Parse the result file to extract papers
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    
                papers = []
                if "papers" in result_data:
                    for paper_data in result_data["papers"]:
                        # Extract relevant information
                        paper_info = {
                            "bibtex_key": paper_data.get("key", ""),
                            "title": paper_data.get("title", ""),
                            "authors": paper_data.get("authors", []),
                            "year": str(paper_data.get("year", "")),
                            "abstract": paper_data.get("abstract", ""),
                            "database": paper_data.get("database", "")
                        }
                        papers.append(paper_info)
                        
                # Store citation keys for this document
                self.document_citations[document_id] = set()
                for paper in papers:
                    if "bibtex_key" in paper:
                        self.document_citations[document_id].add(paper["bibtex_key"])
                
                self.logger.info(f"Found {len(papers)} papers for '{topic}'")
                return papers
            else:
                self.logger.warning(f"No result file found at {result_path}, using fallback method")
                return self._fallback_search(topic, document_id, limit)
                
        except Exception as e:
            self.logger.error(f"Error searching for papers: {str(e)}")
            return self._fallback_search(topic, document_id, limit)
    
    def _fallback_search(self, topic: str, document_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Fallback method to use mock citations when actual search fails.
        
        Args:
            topic: The topic to search for
            document_id: Unique identifier for the document
            limit: Maximum number of papers to return
            
        Returns:
            List of papers metadata
        """
        topic_lower = topic.lower()
        self.logger.info(f"Using fallback search for '{topic}'")
        
        # Choose the most relevant mock dataset based on keywords in the topic
        mock_data = []
        
        # Citizenship related topics
        if any(keyword in topic_lower for keyword in ["citizen", "birthright", "nationality", "jus soli", "naturalization"]):
            self.logger.info(f"Found citizenship-related topic: {topic}")
            mock_data.extend(self.mock_citations["citizenship"])
            # Also include immigration papers as they're closely related
            if len(mock_data) < limit:
                mock_data.extend(self.mock_citations["immigration"])
        
        # Immigration related topics
        elif any(keyword in topic_lower for keyword in ["immigra", "migra", "refugee", "alien", "foreign"]):
            self.logger.info(f"Found immigration-related topic: {topic}")
            mock_data.extend(self.mock_citations["immigration"])
            # Also include some citizenship papers as they're related
            if len(mock_data) < limit:
                mock_data.extend(self.mock_citations["citizenship"][:3])
        
        # Neural network and computer vision topics
        elif any(keyword in topic_lower for keyword in ["neural", "network", "deep learning", "machine learning", "ai"]):
            self.logger.info(f"Found neural network-related topic: {topic}")
            mock_data.extend(self.mock_citations["neural_networks"])
            
        elif any(keyword in topic_lower for keyword in ["vision", "image", "computer vision", "recognition"]):
            self.logger.info(f"Found computer vision-related topic: {topic}")
            mock_data.extend(self.mock_citations["computer_vision"])
            # Also include neural network papers as they're related to CV
            if len(mock_data) < limit:
                mock_data.extend(self.mock_citations["neural_networks"])
            
        # If no relevant mock data found, use a generic approach
        if not mock_data:
            self.logger.info(f"No specific dataset for topic: {topic}. Using mixed citations.")
            # Provide a mix of citations from all available categories
            for dataset in self.mock_citations.values():
                mock_data.extend(dataset[:3])  # Take a few from each category
        
        # Remove duplicates while preserving order
        seen = set()
        unique_mock_data = []
        for item in mock_data:
            if item["bibtex_key"] not in seen:
                seen.add(item["bibtex_key"])
                unique_mock_data.append(item)
        
        # Limit the number of results
        selected_papers = unique_mock_data[:limit]
        
        # Store citation keys for this document
        self.document_citations[document_id] = set()
        for paper in selected_papers:
            if "bibtex_key" in paper:
                self.document_citations[document_id].add(paper["bibtex_key"])
        
        self.logger.info(f"Found {len(selected_papers)} fallback citations for '{topic}'")
        return selected_papers
    
    def get_bibtex_path(self, document_id: str) -> str:
        """
        Get the path to the BibTeX file for a document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Path to the BibTeX file
        """
        bibtex_path = os.path.join(self.cache_dir, document_id, "bibliography.bib")
        
        # If the BibTeX file doesn't exist, create it
        if not os.path.exists(bibtex_path) and document_id in self.document_citations:
            try:
                # Get the result.json path
                result_path = os.path.join(self.cache_dir, document_id, "result.json")
                
                # Check if the result file exists
                if os.path.exists(result_path):
                    # Ensure the parent directory exists
                    os.makedirs(os.path.dirname(bibtex_path), exist_ok=True)
                    
                    # Use findpapers to generate the BibTeX file
                    # Note: findpapers expects the output path to be a directory, not a file
                    output_dir = os.path.dirname(bibtex_path)
                    findpapers.generate_bibtex(
                        filepath=result_path,
                        outputpath=output_dir,
                        only_selected_papers=False,
                        categories_by_facet=None,
                        add_findpapers_citation=False,
                        verbose=True
                    )
                    
                    # Rename the file if needed
                    generated_bibtex = os.path.join(output_dir, "bibliography.bib")
                    if os.path.exists(generated_bibtex) and generated_bibtex != bibtex_path:
                        import shutil
                        shutil.move(generated_bibtex, bibtex_path)
                else:
                    # Create a simple BibTeX file with mock data
                    self._create_mock_bibtex(document_id, bibtex_path)
            except Exception as e:
                self.logger.error(f"Error generating BibTeX file: {str(e)}")
                # Create a simple BibTeX file with mock data as fallback
                self._create_mock_bibtex(document_id, bibtex_path)
                
        return bibtex_path
        
    def _create_mock_bibtex(self, document_id: str, bibtex_path: str) -> None:
        """
        Create a mock BibTeX file for a document.
        
        Args:
            document_id: Unique identifier for the document
            bibtex_path: Path to write the BibTeX file
        """
        if document_id not in self.document_citations:
            return
            
        try:
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(bibtex_path), exist_ok=True)
            
            # Get citation summaries
            citation_summaries = self.get_citations_summary(document_id)
            
            # Write BibTeX file
            with open(bibtex_path, 'w', encoding='utf-8') as f:
                f.write("% Bibliography created by Doc Engineer\n\n")
                
                for citation in citation_summaries:
                    key = citation.get("bibtex_key", "unknown")
                    title = citation.get("title", "Untitled")
                    authors = " and ".join(citation.get("authors", ["Unknown Author"]))
                    year = citation.get("year", "2023")
                    
                    # Determine the type of publication
                    venue = citation.get("database", "Unknown Journal")
                    pub_type = "article"
                    if "conference" in venue.lower() or "proceedings" in venue.lower():
                        pub_type = "inproceedings"
                    elif "book" in venue.lower() or "press" in venue.lower():
                        pub_type = "book"
                    
                    f.write(f"@{pub_type}{{{key},\n")
                    f.write(f"  title = {{{title}}},\n")
                    f.write(f"  author = {{{authors}}},\n")
                    f.write(f"  year = {{{year}}},\n")
                    
                    # Add venue-specific fields
                    if pub_type == "article":
                        f.write(f"  journal = {{{venue}}},\n")
                    elif pub_type == "inproceedings":
                        f.write(f"  booktitle = {{{venue}}},\n")
                    elif pub_type == "book":
                        f.write(f"  publisher = {{{venue}}},\n")
                        
                    # Add DOI if available
                    if "doi" in citation:
                        f.write(f"  doi = {{{citation['doi']}}},\n")
                    
                    # Add URL if available
                    if "url" in citation:
                        f.write(f"  url = {{{citation['url']}}},\n")
                        
                    # Add citation count if available (as note)
                    if "citation_count" in citation and citation["citation_count"]:
                        f.write(f"  note = {{Cited by {citation['citation_count']} papers}},\n")
                    
                    f.write("}\n\n")
        
        except Exception as e:
            self.logger.error(f"Error creating mock BibTeX file: {str(e)}")
    
    def get_citation_keys(self, document_id: str) -> List[str]:
        """
        Get the list of citation keys for a document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            List of citation keys
        """
        return list(self.document_citations.get(document_id, set()))
    
    def get_citations_summary(self, document_id: str) -> List[Dict[str, str]]:
        """
        Get a summary of citations for a document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            List of citation summaries
        """
        if document_id not in self.document_citations:
            return []
            
        # Path to the document's findpapers result JSON file
        result_path = os.path.join(self.cache_dir, document_id, "result.json")
        
        # Get the citation keys for this document
        citation_keys = self.document_citations[document_id]
        
        # First try to get citations from the result.json file
        if os.path.exists(result_path):
            try:
                # Read the search result from the JSON file
                with open(result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    
                # Extract papers from the result data
                papers = []
                if "papers" in result_data:
                    for paper_data in result_data["papers"]:
                        if paper_data.get("key", "") in citation_keys:
                            paper_info = {
                                "bibtex_key": paper_data.get("key", ""),
                                "title": paper_data.get("title", ""),
                                "authors": paper_data.get("authors", []),
                                "year": str(paper_data.get("year", "")),
                                "abstract": paper_data.get("abstract", ""),
                                "database": paper_data.get("database", "")
                            }
                            papers.append(paper_info)
                
                return papers
                
            except Exception as e:
                self.logger.error(f"Error retrieving citation summary from file: {str(e)}")
                # Fall back to in-memory citations
        
        # If file not found or couldn't be read, use in-memory mock citations
        self.logger.info(f"Using in-memory citations for document {document_id}")
        
        # Find the full citation details from mock data
        citations = []
        for dataset in self.mock_citations.values():
            for paper in dataset:
                if paper.get("bibtex_key", "") in citation_keys:
                    citations.append(paper)
        
        return citations
    
    def _create_search_query(self, topic: str) -> str:
        """
        Create a search query from a topic.
        
        Args:
            topic: The topic to create a query for
            
        Returns:
            Formatted search query for findpapers
        """
        # Split the topic into keywords
        keywords = [kw.strip() for kw in topic.split() if len(kw.strip()) > 3]
        
        # Create a findpapers-compatible query
        # The format should be: [term A] AND ([term B] OR [term C]) AND NOT [term D]
        
        # Start with the whole topic enclosed in brackets
        query = f"[{topic}]"
        
        # Add important individual keywords with OR logic if there are more than 2 keywords
        if len(keywords) > 2:
            keyword_part = " OR ".join([f"[{kw}]" for kw in keywords])
            query += f" AND ({keyword_part})"
        
        return query

    def select_citations_for_section(self, citations: List[Dict[str, Any]], section_title: str, section_description: str, document_id: str = None) -> List[Dict[str, Any]]:
        """
        Select citations that are relevant to a specific section using vector search when available.

        Args:
            citations: List of citation metadata (potentially pre-filtered)
            section_title: Title of the section
            section_description: Description of the section
            document_id: Unique identifier for the document

        Returns:
            List of relevant citations with text chunks
        """
        # If document_id is provided and vector index exists, use semantic search
        if document_id and hasattr(self, 'query_vector_index'): # Changed self.citation_manager to self
            try:
                # Create a query from the section title and description
                query = f"{section_title}: {section_description}"

                # Query the vector index
                print(f"Using vector search for '{section_title}' section")
                vector_results = self.query_vector_index(document_id, query, top_k=5) # Changed self.citation_manager to self

                if vector_results:
                    # Process vector search results
                    enhanced_citations = []
                    seen_keys = set()

                    for result in vector_results:
                        # Get metadata from the result
                        metadata = result.get('metadata', {})
                        text = result.get('text', '')

                        # Find the corresponding citation
                        citation_key = metadata.get('paperId', '')
                        matching_citation = None

                        for citation in citations:
                            if citation.get('bibtex_key') == citation_key or citation.get('title') == metadata.get('title'):
                                matching_citation = citation.copy()
                                break

                        if not matching_citation:
                            # If no exact match, create a new citation from metadata
                            matching_citation = {
                                "bibtex_key": citation_key,
                                "title": metadata.get('title', 'Unknown Title'),
                                "authors": metadata.get('authors', []),
                                "year": metadata.get('year', ''),
                                "abstract": metadata.get('abstract', ''),
                            }

                        # Add the text chunk to the citation
                        if 'text_chunks' not in matching_citation:
                            matching_citation['text_chunks'] = []

                        # Add this text chunk if it's not too long
                        if len(text) > 50:  # Only add substantive chunks
                            matching_citation['text_chunks'].append(text)

                        # Only add each citation once
                        if matching_citation.get('bibtex_key') not in seen_keys:
                            seen_keys.add(matching_citation.get('bibtex_key'))
                            enhanced_citations.append(matching_citation)

                    if enhanced_citations:
                        print(f"Found {len(enhanced_citations)} relevant citations using vector search")
                        return enhanced_citations

            except Exception as e:
                print(f"Vector search failed: {e}. Falling back to keyword matching.")

        # Fall back to keyword matching if vector search failed or is not available
        print("Using keyword matching for citation selection")
        keywords = set([word.lower() for word in section_title.split() + section_description.split() if len(word) > 3])

        scored_citations = []
        for citation in citations:
            # Calculate relevance score based on keyword matches in title and abstract
            score = 0
            title = citation.get("title", "").lower()
            abstract = citation.get("abstract", "").lower()

            # Check title matches
            for keyword in keywords:
                if keyword in title:
                    score += 3  # Title matches are more important
                if keyword in abstract:
                    score += 1

            # Only include citations with at least some relevance
            if score > 0:
                scored_citations.append((citation, score))

        # Sort by relevance score
        scored_citations.sort(key=lambda x: x[1], reverse=True)

        # Return the citations without scores
        return [citation for citation, _ in scored_citations]

    def query_vector_index(self, document_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the vector index for relevant paper chunks.
        
        Args:
            document_id: Unique identifier for the document
            query: Query string (e.g., section title or description)
            top_k: Number of top results to return
            
        Returns:
            List of relevant text chunks with metadata
        """
        if document_id not in self.vector_indexes:
            self.logger.warning(f"No vector index found for document {document_id}")
            return []
            
        try:
            # Get the vector index
            vector_index = self.vector_indexes[document_id]
            
            # Create citation query engine
            query_engine = vector_index.as_query_engine(
                similarity_top_k=top_k,
                citation_chunk_size=512,
                response_mode="no_text"  # Just return the nodes, not a synthesized response
            )
            
            # Execute the query
            response = query_engine.query(query)
            
            # Extract relevant nodes
            results = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    metadata = node.node.metadata.copy()
                    text = node.node.text
                    
                    # Add a snippet to make it easier to use
                    results.append({
                        "text": text,
                        "metadata": metadata,
                        "score": node.score if hasattr(node, 'score') else 0
                    })
                    
            return results
        except Exception as e:
            self.logger.error(f"Error querying vector index: {str(e)}")
            return [] 

    def download_pdf(self, paper_info: Dict[str, Any], document_id: str) -> Optional[str]:
        """
        Download PDF for a paper if available and extract text.
        
        Args:
            paper_info: Paper metadata
            document_id: Unique identifier for the document
            
        Returns:
            Extracted text from the PDF, or None if download failed
        """
        if not paper_info.get("open_access_pdf"):
            return None
            
        try:
            from pypdf import PdfReader
            import requests
            from io import BytesIO
            
            # Get PDF URL - handle different formats of open_access_pdf
            pdf_url = None
            open_access_info = paper_info["open_access_pdf"]
            
            # Handle case where it's a string (direct URL)
            if isinstance(open_access_info, str):
                pdf_url = open_access_info
            # Handle case where it's a dictionary with url field
            elif isinstance(open_access_info, dict) and open_access_info.get("url"):
                pdf_url = open_access_info["url"]
            # Handle other possible formats
            else:
                self.logger.warning(f"Unsupported open_access_pdf format: {type(open_access_info)}")
                return None
                
            if not pdf_url:
                return None
                
            self.logger.info(f"Downloading PDF for {paper_info.get('title', 'unknown paper')}")
            
            # Create PDF directory if it doesn't exist
            pdf_dir = os.path.join(self.cache_dir, document_id, "pdfs")
            os.makedirs(pdf_dir, exist_ok=True)
            
            # Generate filename from bibtex key
            bibtex_key = paper_info.get("bibtex_key", str(uuid.uuid4())[:8])
            pdf_path = os.path.join(pdf_dir, f"{bibtex_key}.pdf")
            
            # Check if already downloaded
            if os.path.exists(pdf_path):
                self.logger.info(f"PDF already exists at {pdf_path}")
            else:
                # Download PDF
                response = requests.get(pdf_url, timeout=30)
                if response.status_code == 200:
                    with open(pdf_path, 'wb') as f:
                        f.write(response.content)
                    self.logger.info(f"PDF downloaded to {pdf_path}")
                else:
                    self.logger.warning(f"Failed to download PDF: HTTP {response.status_code}")
                    return None
            
            # Extract text from PDF
            text = ""
            try:
                # Use BytesIO to handle potential file locking issues
                with open(pdf_path, "rb") as f:
                    pdf_content = f.read()
                
                pdf_file = BytesIO(pdf_content)
                reader = PdfReader(pdf_file)
                
                # Get number of pages
                num_pages = len(reader.pages)
                self.logger.info(f"Extracting text from {num_pages} pages")
                
                # Extract text (limit to first 50 pages for very large PDFs)
                max_pages = min(50, num_pages)
                for i in range(max_pages):
                    page = reader.pages[i]
                    text += page.extract_text() + "\n\n"
                
                self.logger.info(f"Extracted {len(text)} characters from PDF")
                return text
            except Exception as e:
                self.logger.error(f"Error extracting text from PDF: {str(e)}")
                return None
                
        except ImportError:
            self.logger.warning("pypdf not installed. Install with: pip install pypdf")
            return None
        except Exception as e:
            self.logger.error(f"Error downloading PDF: {str(e)}")
            return None
