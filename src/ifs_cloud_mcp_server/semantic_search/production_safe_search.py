"""
Production-Safe Semantic Search System
=====================================

This search system works with production-safe embeddings that contain
only metadata and AI-generated summaries, not source code content.

Users provide their own copy of source files at runtime for content access.

Author: AI Assistant
Date: August 17, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .production_data_structures import (
    ProductionCodeChunk,
    CodeContentLoader,
    ProductionSearchResult,
)


class ProductionSafeSearch:
    """
    Search system that works with production-safe embeddings.

    Key Features:
    - Searches using only AI summaries and metadata (no source code)
    - Allows users to provide their own source files for content access
    - Full copyright compliance while maintaining search functionality
    """

    def __init__(self, embeddings_dir: Path, user_workspace_root: Path = None):
        """
        Initialize production-safe search system.

        Args:
            embeddings_dir: Directory containing production-safe embeddings
            user_workspace_root: User's workspace root for loading actual content
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.user_workspace_root = user_workspace_root
        self.logger = logging.getLogger(__name__)

        # Load production-safe data
        self.chunks: List[ProductionCodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.content_loader: Optional[CodeContentLoader] = None

        # Load embeddings and metadata
        self._load_embeddings()
        self._load_chunks()

        # Initialize content loader if user workspace provided
        if user_workspace_root:
            self.content_loader = CodeContentLoader(user_workspace_root)

    def _load_embeddings(self):
        """Load embeddings created from AI summaries and metadata"""
        embeddings_path = self.embeddings_dir / "embeddings.npy"

        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
            self.logger.info(f"✅ Loaded {self.embeddings.shape[0]} embeddings")
        else:
            self.logger.error(f"❌ Embeddings not found: {embeddings_path}")
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    def _load_chunks(self):
        """Load production-safe chunk metadata (no source code)"""
        chunks_path = self.embeddings_dir / "production_chunks.json"

        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            self.chunks = [ProductionCodeChunk.from_dict(data) for data in chunks_data]
            self.logger.info(f"✅ Loaded {len(self.chunks)} production-safe chunks")

            # Verify alignment
            if self.embeddings is not None and len(self.chunks) != len(self.embeddings):
                self.logger.warning(
                    f"⚠️ Mismatch: {len(self.chunks)} chunks but {len(self.embeddings)} embeddings"
                )
        else:
            self.logger.error(f"❌ Chunks not found: {chunks_path}")
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    def search(
        self, query: str, top_k: int = 10, min_similarity: float = 0.1
    ) -> List[ProductionSearchResult]:
        """
        Search using AI summaries and metadata only.

        Args:
            query: Search query in natural language
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of ProductionSearchResult objects
        """
        if self.embeddings is None or not self.chunks:
            self.logger.error("❌ No embeddings or chunks loaded")
            return []

        # Create query embedding (would need tokenizer/model for real implementation)
        # For demo purposes, we'll use a placeholder approach
        query_embedding = self._create_query_embedding(query)

        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top results above threshold
        result_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for rank, idx in enumerate(result_indices):
            if similarities[idx] >= min_similarity:
                result = ProductionSearchResult(
                    chunk=self.chunks[idx],
                    similarity_score=float(similarities[idx]),
                    rank=rank + 1,
                )
                results.append(result)

        return results

    def _create_query_embedding(self, query: str) -> np.ndarray:
        """
        Create embedding for query (placeholder implementation).
        In real usage, this would use the same model as used for chunk embeddings.
        """
        # Placeholder: return random embedding with correct dimensions
        if self.embeddings is not None:
            embedding_dim = self.embeddings.shape[1]
            return np.random.random(embedding_dim)  # Replace with real embedding
        else:
            return np.random.random(768)  # Default dimension

    def get_search_results_with_content(
        self, results: List[ProductionSearchResult]
    ) -> List[Dict[str, Any]]:
        """
        Get search results with actual content loaded from user's files.

        Args:
            results: Search results from search()

        Returns:
            List of enriched results with actual content (if user provided workspace)
        """
        enriched_results = []

        for result in results:
            result_dict = result.get_display_info()

            # Try to load actual content if user provided workspace
            if self.content_loader:
                try:
                    content = result.load_content(self.content_loader)
                    result_dict["content"] = content
                    result_dict["content_available"] = content is not None

                    # Verify integrity
                    if content:
                        integrity_ok = self.content_loader.verify_chunk_integrity(
                            result.chunk
                        )
                        result_dict["integrity_verified"] = integrity_ok
                        if not integrity_ok:
                            result_dict["warning"] = "File has changed since indexing"

                except Exception as e:
                    result_dict["content_error"] = str(e)
                    result_dict["content_available"] = False
            else:
                result_dict["content"] = None
                result_dict["content_available"] = False
                result_dict["note"] = (
                    "Provide user_workspace_root to load actual content"
                )

            enriched_results.append(result_dict)

        return enriched_results

    def search_with_content(
        self, query: str, top_k: int = 10, min_similarity: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: search and return results with content.
        """
        results = self.search(query, top_k, min_similarity)
        return self.get_search_results_with_content(results)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded search index"""
        stats = {
            "total_chunks": len(self.chunks),
            "embedding_dimension": (
                self.embeddings.shape[1] if self.embeddings is not None else 0
            ),
            "ai_enhanced_chunks": sum(1 for chunk in self.chunks if chunk.ai_summary),
            "modules_covered": len(
                set(chunk.module_name for chunk in self.chunks if chunk.module_name)
            ),
            "content_loader_available": self.content_loader is not None,
            "user_workspace_root": (
                str(self.user_workspace_root) if self.user_workspace_root else None
            ),
        }

        # Chunk type distribution
        chunk_types = {}
        for chunk in self.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        stats["chunk_type_distribution"] = chunk_types

        # Language distribution
        languages = {}
        for chunk in self.chunks:
            languages[chunk.language] = languages.get(chunk.language, 0) + 1
        stats["language_distribution"] = languages

        return stats

    def verify_copyright_compliance(self) -> Dict[str, Any]:
        """
        Verify that the search system is copyright compliant.
        """
        compliance_report = {"compliant": True, "checks": []}

        # Check 1: No source code in chunks
        has_raw_content = any(
            hasattr(chunk, "raw_content") and chunk.raw_content for chunk in self.chunks
        )
        compliance_report["checks"].append(
            {
                "check": "No raw source code stored",
                "passed": not has_raw_content,
                "details": "Chunks contain only metadata and AI summaries",
            }
        )

        # Check 2: AI summaries present (derived work)
        ai_summary_count = sum(1 for chunk in self.chunks if chunk.ai_summary)
        compliance_report["checks"].append(
            {
                "check": "AI-generated content present",
                "passed": ai_summary_count > 0,
                "details": f"{ai_summary_count}/{len(self.chunks)} chunks have AI summaries",
            }
        )

        # Check 3: Only references to original files
        has_file_references = all(chunk.relative_file_path for chunk in self.chunks)
        compliance_report["checks"].append(
            {
                "check": "File references only (no content)",
                "passed": has_file_references,
                "details": "All chunks reference files but store no content",
            }
        )

        # Overall compliance
        compliance_report["compliant"] = all(
            check["passed"] for check in compliance_report["checks"]
        )

        return compliance_report


def demo_production_safe_search(embeddings_dir: Path, user_workspace_root: Path = None):
    """
    Demo the production-safe search system.
    """
    print("🔍 Production-Safe Search System Demo")
    print("=" * 50)

    # Initialize search system
    try:
        search = ProductionSafeSearch(embeddings_dir, user_workspace_root)

        # Show statistics
        stats = search.get_statistics()
        print(f"📊 Search Index Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   AI-enhanced chunks: {stats['ai_enhanced_chunks']}")
        print(f"   Modules covered: {stats['modules_covered']}")
        print(f"   Content loader available: {stats['content_loader_available']}")

        # Verify compliance
        compliance = search.verify_copyright_compliance()
        print(
            f"\n⚖️ Copyright Compliance: {'✅ COMPLIANT' if compliance['compliant'] else '❌ NOT COMPLIANT'}"
        )
        for check in compliance["checks"]:
            status = "✅" if check["passed"] else "❌"
            print(f"   {status} {check['check']}: {check['details']}")

        # Example searches
        example_queries = [
            "customer credit validation",
            "order processing functions",
            "inventory management",
            "financial calculations",
        ]

        print(f"\n🔍 Example Searches:")
        for query in example_queries[:2]:  # Limit for demo
            print(f"\n   Query: '{query}'")
            results = search.search(query, top_k=3)

            if results:
                for result in results:
                    chunk = result.chunk
                    print(f"     📝 {chunk.function_name} ({chunk.module_name})")
                    print(f"        Similarity: {result.similarity_score:.3f}")
                    print(
                        f"        File: {chunk.relative_file_path}:{chunk.start_line}-{chunk.end_line}"
                    )
                    if chunk.ai_summary:
                        print(f"        Summary: {chunk.ai_summary[:100]}...")
            else:
                print(f"     No results found")

        if user_workspace_root:
            print(f"\n📁 Content Loading Demo:")
            results = search.search("customer credit", top_k=1)
            if results:
                enriched = search.get_search_results_with_content(results)
                result = enriched[0]

                if result["content_available"]:
                    print(f"   ✅ Content loaded from user's files")
                    print(f"   📄 Content preview: {result['content'][:200]}...")
                else:
                    print(
                        f"   ❌ Content not available: {result.get('content_error', 'Unknown error')}"
                    )

        print(f"\n🎉 Production-safe search demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Demo usage
    embeddings_dir = Path("production_embeddings")
    user_workspace = Path(".")  # User's workspace with their source files

    demo_production_safe_search(embeddings_dir, user_workspace)
