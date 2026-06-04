"""
LangChain Client Module
Replaces bedrock_client.py with LangChain + OpenAI integration
LangChain provides document loaders, chains, and vector store integrations
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from qdrant_client import QdrantClient
from config import settings

logger = logging.getLogger(__name__)


class LangChainClient:
    """
    LangChain client for OpenAI integration
    Provides embeddings, LLM, and document loading capabilities
    """
    
    def __init__(self):
        """Initialize LangChain client with OpenAI"""
        self.embeddings = None
        self.llm = None
        self.qdrant_client = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize OpenAI embeddings and LLM"""
        try:
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key
            )
            logger.info(f"Initialized OpenAI embeddings: {settings.openai_embedding_model}")
            
            # Initialize OpenAI LLM
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0.0,
                openai_api_key=settings.openai_api_key
            )
            logger.info(f"Initialized OpenAI LLM: {settings.openai_model}")
            
            # Initialize Qdrant client for vector store
            self.qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )
            logger.info(f"Initialized Qdrant client: {settings.qdrant_host}:{settings.qdrant_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain components: {e}")
            raise
    
    def get_embeddings(self) -> OpenAIEmbeddings:
        """Get OpenAI embeddings instance"""
        return self.embeddings
    
    def get_llm(self) -> ChatOpenAI:
        """Get OpenAI LLM instance"""
        return self.llm
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF document using LangChain
        Support PDF file upload
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded PDF with {len(documents)} pages")
            return documents
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise
    
    def load_json(self, file_path: str, content_key: str = "content") -> List[Document]:
        """
        Load JSON document using LangChain
        Support JSON file upload
        """
        try:
            # Try using jq first (for complex JSON structures)
            try:
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=f".{content_key}",
                    text_content=False
                )
                documents = loader.load()
                logger.info(f"Loaded JSON with {len(documents)} documents using jq")
                return documents
            except Exception as jq_error:
                # Fallback: simple JSON loading without jq
                logger.warning(f"jq loading failed, trying simple JSON loading: {jq_error}")
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    # Check if it's a list of Q&A pairs
                    if len(data) > 0 and isinstance(data[0], dict):
                        if 'question' in data[0] and 'answer' in data[0]:
                            # Q&A pairs format - create readable text
                            text_content = "\n\n".join([
                                f"Question: {item.get('question', '')}\nAnswer: {item.get('answer', '')}"
                                for item in data
                            ])
                        else:
                            # Generic list of objects
                            text_content = "\n\n".join([
                                json.dumps(item, indent=2) for item in data
                            ])
                    else:
                        # Simple list
                        text_content = "\n\n".join([str(item) for item in data])
                elif isinstance(data, dict):
                    # Single object - extract content
                    if content_key in data:
                        text_content = str(data[content_key])
                    else:
                        # Join all values with keys for context
                        text_content = "\n\n".join([
                            f"{k}: {v}" for k, v in data.items()
                        ])
                else:
                    text_content = str(data)
                
                documents = [Document(page_content=text_content, metadata={"source": file_path})]
                logger.info(f"Loaded JSON with {len(documents)} documents using simple loader")
                return documents
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            raise
    
    def load_json_text(self, json_content: str) -> List[Document]:
        """
        Load JSON content directly from string
        Support JSON content upload
        """
        try:
            import json
            data = json.loads(json_content)
            
            if isinstance(data, dict):
                if "content" in data:
                    text = data["content"]
                elif "text" in data:
                    text = data["text"]
                else:
                    text = str(data)
            elif isinstance(data, list):
                text = "\n".join(str(item) for item in data)
            else:
                text = str(data)
            
            documents = [Document(page_content=text, metadata={"source": "json_upload"})]
            logger.info(f"Loaded JSON content with {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to load JSON content: {e}")
            raise
    
    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 512,
        chunk_overlap: int = 100
    ) -> List[Document]:
        """
        Split documents into chunks using LangChain text splitter
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise
    
    def create_vector_store(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None
    ) -> Qdrant:
        """
        Create Qdrant vector store from documents using LangChain
        """
        try:
            collection_name = collection_name or settings.qdrant_collection_name
            
            # Create vector store
            vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
                collection_name=collection_name
            )
            
            logger.info(f"Created vector store with {len(documents)} documents in collection '{collection_name}'")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def get_vector_store(self, collection_name: Optional[str] = None) -> Qdrant:
        """
        Get existing Qdrant vector store using LangChain
        """
        try:
            collection_name = collection_name or settings.qdrant_collection_name
            
            vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            logger.info(f"Connected to vector store collection '{collection_name}'")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to get vector store: {e}")
            raise
    
    def generate_response(
        self,
        query: str,
        context_docs: List[Document],
        temperature: float = 0.0
    ) -> str:
        """
        Generate response using LangChain LLM with context
        """
        try:
            # Combine context documents
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Create prompt
            prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.llm.invoke(prompt, temperature=temperature)
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise


# Global LangChain client instance
langchain_client = LangChainClient()
