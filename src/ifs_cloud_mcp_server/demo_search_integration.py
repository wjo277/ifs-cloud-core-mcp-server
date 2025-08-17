#!/usr/bin/env python3
"""
UniXcoder Demo Integration for Web UI
====================================

This module integrates the UniXcoder + FAISS demo search engine
with the existing web UI, providing a side-by-side comparison
between the current search and the new semantic search.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import HTTPException
from demo_unixcoder_search import UniXcoderSearchEngine
from src.ifs_cloud_mcp_server.semantic_search.data_structures import SearchResult

logger = logging.getLogger(__name__)


class DemoSearchIntegration:
    """
    Integration class for UniXcoder demo search in the web UI.

    FEATURES:
    --------
    - Side-by-side comparison with existing search
    - Real-time performance metrics
    - Module-specific filtering
    - Rich result formatting for web UI
    """

    def __init__(self):
        """Initialize the demo search integration."""
        self.unixcoder_engine: Optional[UniXcoderSearchEngine] = None
        self.is_initialized = False
        self.demo_modules = ["proj", "prjrep", "fndbas", "accrul"]
        self.initialization_error = None

        logger.info("🎯 Demo Search Integration initialized")

    async def initialize(self):
        """
        Initialize the UniXcoder search engine.
        This runs asynchronously to avoid blocking the web UI startup.
        """
        try:
            logger.info("🚀 Initializing UniXcoder demo search engine...")

            # Initialize in a thread pool to avoid blocking
            def init_engine():
                engine = UniXcoderSearchEngine()
                demo_index_path = Path("models/demo_search")

                if (demo_index_path / "metadata.json").exists():
                    logger.info("📂 Loading existing demo index...")
                    engine.load_demo_index(demo_index_path)
                else:
                    logger.info("🔧 Building demo index from source...")
                    work_dir = Path("_work")
                    chunks = engine.load_demo_modules(work_dir, self.demo_modules)
                    engine.build_search_index(chunks)
                    engine.save_demo_index(demo_index_path)

                return engine

            # Run initialization in thread pool
            loop = asyncio.get_event_loop()
            self.unixcoder_engine = await loop.run_in_executor(None, init_engine)

            self.is_initialized = True
            logger.info("✅ UniXcoder demo search engine ready!")

        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Failed to initialize UniXcoder engine: {e}")
            raise

    async def search(
        self,
        query: str,
        limit: int = 10,
        module_filter: Optional[str] = None,
        min_score: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Perform search using UniXcoder engine.

        Returns:
        --------
        Dictionary with search results and metadata for web UI display
        """
        if not self.is_initialized:
            if self.initialization_error:
                raise HTTPException(
                    status_code=503,
                    detail=f"UniXcoder search not available: {self.initialization_error}",
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="UniXcoder search is still initializing. Please try again in a moment.",
                )

        start_time = time.time()

        try:
            # Perform search
            results = self.unixcoder_engine.search(query, k=limit, min_score=min_score)

            # Apply module filter if specified
            if module_filter and module_filter != "all":
                results = [r for r in results if r.chunk.module == module_filter]

            search_time = (time.time() - start_time) * 1000

            # Format results for web UI
            formatted_results = []
            for result in results[:limit]:
                formatted_result = {
                    "id": result.chunk.chunk_id,
                    "title": self._generate_title(result),
                    "snippet": self._generate_snippet(result),
                    "file_path": str(result.chunk.file_path),
                    "file_name": Path(result.chunk.file_path).name,
                    "module": result.chunk.module,
                    "function_name": result.chunk.function_name,
                    "line_start": result.chunk.start_line,
                    "line_end": result.chunk.end_line,
                    "language": result.chunk.language,
                    "similarity_score": round(result.similarity_score, 3),
                    "rank": result.rank,
                    "relevance_explanation": result.relevance_explanation,
                    "has_error_handling": result.chunk.has_error_handling,
                    "has_transactions": result.chunk.has_transactions,
                    "complexity_score": round(result.chunk.complexity_score, 2),
                    "api_calls": result.chunk.api_calls[:5],  # Limit for UI
                    "database_tables": result.chunk.database_tables[:5],  # Limit for UI
                }
                formatted_results.append(formatted_result)

            # Return comprehensive response
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(results),
                "search_time_ms": round(search_time, 1),
                "search_method": "UniXcoder + FAISS",
                "model": "microsoft/unixcoder-base",
                "available_modules": self.demo_modules,
                "applied_filters": {"module": module_filter, "min_score": min_score},
                "metadata": {
                    "total_indexed_chunks": len(self.unixcoder_engine.chunks),
                    "embedding_dimension": self.unixcoder_engine.embeddings.shape[1],
                    "faiss_index_type": type(
                        self.unixcoder_engine.faiss_index
                    ).__name__,
                },
            }

        except Exception as e:
            logger.error(f"Error in UniXcoder search: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def _generate_title(self, result: SearchResult) -> str:
        """Generate a descriptive title for the search result."""
        parts = []

        # Add module prefix
        if result.chunk.module:
            parts.append(f"[{result.chunk.module.upper()}]")

        # Add function name if available
        if result.chunk.function_name:
            parts.append(result.chunk.function_name)
        else:
            # Use file name
            parts.append(Path(result.chunk.file_path).stem)

        # Add language indicator
        if result.chunk.language:
            lang_suffix = {
                "plsql": "(PL/SQL)",
                "javascript": "(JS)",
                "java": "(Java)",
                "sql": "(SQL)",
                "python": "(Python)",
            }.get(result.chunk.language.lower(), f"({result.chunk.language})")
            parts.append(lang_suffix)

        return " ".join(parts)

    def _generate_snippet(self, result: SearchResult) -> str:
        """Generate a code snippet for display."""
        content = result.chunk.raw_content.strip()

        # Try to find the most relevant lines
        lines = content.split("\n")

        # Prefer lines with function definitions, important keywords
        important_keywords = [
            "PROCEDURE",
            "FUNCTION",
            "BEGIN",
            "CREATE",
            "SELECT",
            "function",
            "const",
            "class",
        ]
        important_lines = []

        for i, line in enumerate(lines):
            for keyword in important_keywords:
                if keyword in line:
                    # Include surrounding context
                    start_idx = max(0, i - 1)
                    end_idx = min(len(lines), i + 3)
                    important_lines.extend(lines[start_idx:end_idx])
                    break

        if important_lines:
            snippet = "\n".join(important_lines[:8])  # Limit lines for UI
        else:
            # Fall back to first lines
            snippet = "\n".join(lines[:8])

        # Truncate if too long
        if len(snippet) > 500:
            snippet = snippet[:497] + "..."

        return snippet

    def get_available_modules(self) -> List[str]:
        """Get list of available modules for filtering."""
        return self.demo_modules.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the demo search integration."""
        if not self.is_initialized:
            return {
                "status": "initializing" if not self.initialization_error else "error",
                "error": self.initialization_error,
                "available": False,
            }

        return {
            "status": "ready",
            "available": True,
            "modules": self.demo_modules,
            "total_chunks": (
                len(self.unixcoder_engine.chunks) if self.unixcoder_engine else 0
            ),
            "model": "microsoft/unixcoder-base",
            "search_engine": "UniXcoder + FAISS",
        }


# Global instance for the web UI
demo_search = DemoSearchIntegration()


async def initialize_demo_search():
    """Initialize the demo search (call this on startup)."""
    await demo_search.initialize()


def get_demo_search() -> DemoSearchIntegration:
    """Get the demo search instance."""
    return demo_search
