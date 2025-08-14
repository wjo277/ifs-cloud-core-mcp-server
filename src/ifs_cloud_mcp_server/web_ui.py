#!/usr/bin/env python3
"""
FastAPI Web UI for IFS Cloud MCP Server with type-ahead search.
Provides a Meilisearch-like interface for exploring IFS Cloud codebases.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer, SearchResult

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    limit: int = 20
    file_type: Optional[str] = None
    module: Optional[str] = None
    logical_unit: Optional[str] = None
    min_complexity: Optional[float] = None
    max_complexity: Optional[float] = None


class SearchSuggestion(BaseModel):
    """Search suggestion model for type-ahead."""

    text: str
    type: str  # 'entity', 'function', 'module', 'file', 'frontend'
    score: float
    context: str  # Additional context like module or file


class WebUISearchResult(BaseModel):
    """Enhanced search result for web UI."""

    path: str
    name: str
    type: str
    content_preview: str
    score: float
    entities: List[str]
    functions: List[str] = []
    line_count: int
    complexity_score: float
    modified_time: datetime
    hash: str  # Unique content hash for React keys
    module: Optional[str] = None
    logical_unit: Optional[str] = None
    entity_name: Optional[str] = None
    component: Optional[str] = None
    # Frontend elements
    pages: List[str] = []
    lists: List[str] = []
    groups: List[str] = []
    entitysets: List[str] = []
    iconsets: List[str] = []
    trees: List[str] = []
    navigators: List[str] = []
    contexts: List[str] = []
    # UI-specific fields
    highlight: str = ""
    tags: List[str] = []

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class IFSCloudWebUI:
    """Web UI application for IFS Cloud search."""

    def __init__(self, index_path: str = "./index"):
        self.indexer = IFSCloudTantivyIndexer(index_path)
        self.app = FastAPI(
            title="IFS Cloud Explorer",
            description="Web UI for exploring IFS Cloud codebases with intelligent search",
            version="1.0.0",
        )
        self._setup_routes()
        self._setup_static_files()

        # Cache for suggestions
        self._suggestion_cache: Dict[str, List[SearchSuggestion]] = {}

    def _setup_static_files(self):
        """Setup static files and templates."""
        # Create static and templates directories if they don't exist
        static_dir = Path("static")
        templates_dir = Path("templates")
        static_dir.mkdir(exist_ok=True)
        templates_dir.mkdir(exist_ok=True)

        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

        # Setup templates
        self.templates = Jinja2Templates(directory="templates")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Serve the main search interface."""
            return self.templates.TemplateResponse("react.html", {"request": request})

        @self.app.get("/api/search")
        async def search(
            q: str = Query(..., description="Search query"),
            limit: int = Query(20, description="Maximum results"),
            file_type: Optional[str] = Query(None, description="Filter by file type"),
            module: Optional[str] = Query(None, description="Filter by module"),
            logical_unit: Optional[str] = Query(
                None, description="Filter by logical unit"
            ),
            min_complexity: Optional[float] = Query(
                None, description="Minimum complexity"
            ),
            max_complexity: Optional[float] = Query(
                None, description="Maximum complexity"
            ),
        ):
            """Search endpoint with filtering."""
            try:
                # Check if index is properly initialized
                if self.indexer._index is None:
                    logger.error("Index is not initialized")
                    raise HTTPException(
                        status_code=503,
                        detail="Search index is not initialized. Please rebuild the index.",
                    )

                logger.debug(
                    f"Performing search: query='{q}', limit={limit}, file_type={file_type}"
                )

                results = self.indexer.search_deduplicated(
                    query=q,
                    limit=limit,
                    file_type=file_type,
                    min_complexity=min_complexity,
                    max_complexity=max_complexity,
                )

                logger.debug(f"Search returned {len(results)} unique results")

                # Convert to web UI format with enhanced data
                web_results = []
                for result in results:
                    # Create tags based on content
                    tags = []
                    if result.module:
                        tags.append(f"module:{result.module}")
                    if result.logical_unit:
                        tags.append(f"lu:{result.logical_unit}")
                    if result.component:
                        tags.append(f"component:{result.component}")
                    if result.pages:
                        tags.append("has-pages")
                    if result.iconsets:
                        tags.append("has-iconsets")
                    if result.trees:
                        tags.append("has-trees")
                    if result.navigators:
                        tags.append("has-navigators")

                    # Create highlight snippet
                    highlight = self._create_highlight(result.content_preview, q)

                    web_result = WebUISearchResult(
                        path=result.path,
                        name=result.name,
                        type=result.type,
                        content_preview=result.content_preview,
                        score=result.score,
                        entities=result.entities,
                        line_count=result.line_count,
                        complexity_score=result.complexity_score,
                        modified_time=result.modified_time,
                        hash=result.hash,
                        module=result.module,
                        logical_unit=result.logical_unit,
                        entity_name=result.entity_name,
                        component=result.component,
                        pages=result.pages,
                        lists=result.lists,
                        groups=result.groups,
                        entitysets=result.entitysets,
                        iconsets=result.iconsets,
                        trees=result.trees,
                        navigators=result.navigators,
                        contexts=result.contexts,
                        highlight=highlight,
                        tags=tags,
                    )
                    web_results.append(web_result)

                # Use Pydantic's model_dump with mode='json' for proper serialization
                response_data = {
                    "results": [r.model_dump(mode="json") for r in web_results],
                    "total": len(web_results),
                    "query": q,
                    "filters": {
                        "file_type": file_type,
                        "module": module,
                        "logical_unit": logical_unit,
                        "complexity_range": [min_complexity, max_complexity],
                    },
                }
                return JSONResponse(response_data)

            except Exception as e:
                logger.error(f"Search error: {type(e).__name__}: {e}")
                logger.error(
                    f"Search parameters: q='{q}', limit={limit}, file_type={file_type}"
                )
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @self.app.post("/search")
        async def post_search(request: SearchRequest):
            """Search endpoint via POST."""
            try:
                # Check if index is properly initialized
                if self.indexer._index is None:
                    logger.error("Index is not initialized")
                    raise HTTPException(
                        status_code=503,
                        detail="Search index is not initialized. Please rebuild the index.",
                    )

                logger.debug(
                    f"Performing POST search: query='{request.query}', limit={request.limit}"
                )

                # Perform search with filters
                results = self.indexer.search_deduplicated(
                    request.query,
                    limit=request.limit,
                    file_type=getattr(request, "file_type", None),
                    min_complexity=getattr(request, "min_complexity", None),
                    max_complexity=getattr(request, "max_complexity", None),
                )

                # Convert results to web format
                web_results = []
                for result in results:
                    # Generate tags for better categorization
                    tags = []
                    if result.module:
                        tags.append(f"module:{result.module}")
                    if result.logical_unit:
                        tags.append(f"unit:{result.logical_unit}")
                    if result.component:
                        tags.append(f"component:{result.component}")
                    if result.pages:
                        tags.append("has-pages")
                    if result.iconsets:
                        tags.append("has-iconsets")
                    if result.trees:
                        tags.append("has-trees")
                    if result.navigators:
                        tags.append("has-navigators")

                    # Create highlight snippet
                    highlight = self._create_highlight(
                        result.content_preview, request.query
                    )

                    web_result = WebUISearchResult(
                        path=result.path,
                        name=result.name,
                        type=result.type,
                        content_preview=result.content_preview,
                        score=result.score,
                        entities=result.entities,
                        line_count=result.line_count,
                        complexity_score=result.complexity_score,
                        modified_time=result.modified_time,
                        hash=result.hash,
                        module=result.module,
                        logical_unit=result.logical_unit,
                        entity_name=result.entity_name,
                        component=result.component,
                        pages=result.pages,
                        lists=result.lists,
                        groups=result.groups,
                        iconsets=result.iconsets,
                        trees=result.trees,
                        navigators=result.navigators,
                        contexts=result.contexts,
                        tags=tags,
                        highlight=highlight,
                    )
                    web_results.append(web_result)

                # Use Pydantic's model_dump with mode='json' for proper serialization
                response_data = {
                    "results": [
                        result.model_dump(mode="json") for result in web_results
                    ],
                    "total": len(web_results),
                    "query": request.query,
                    "took_ms": 0,  # We could add timing if needed
                }
                return JSONResponse(response_data)

            except Exception as e:
                logger.error(f"POST Search error: {type(e).__name__}: {e}")
                logger.error(
                    f"Search parameters: query='{request.query}', limit={request.limit}"
                )
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @self.app.post("/search")
        async def post_search(request: SearchRequest):
            """Search endpoint via POST with JSON request."""
            try:
                # Check if index is properly initialized
                if self.indexer._index is None:
                    logger.error("Index is not initialized")
                    raise HTTPException(
                        status_code=503,
                        detail="Search index is not initialized. Please rebuild the index.",
                    )

                logger.debug(
                    f"Performing POST search: query='{request.query}', limit={request.limit}"
                )

                # Perform search
                results = self.indexer.search_deduplicated(
                    request.query,
                    limit=request.limit,
                    file_type=getattr(request, "file_type", None),
                    min_complexity=getattr(request, "min_complexity", None),
                    max_complexity=getattr(request, "max_complexity", None),
                )

                # Convert results to web UI format
                web_results = []
                for result in results:
                    # Create tags for classification
                    tags = []
                    if result.module:
                        tags.append(f"module:{result.module}")
                    if result.logical_unit:
                        tags.append(f"unit:{result.logical_unit}")
                    if result.component:
                        tags.append(f"component:{result.component}")
                    if result.pages:
                        tags.append("has-pages")
                    if result.iconsets:
                        tags.append("has-iconsets")
                    if result.trees:
                        tags.append("has-trees")
                    if result.navigators:
                        tags.append("has-navigators")

                    # Create highlight snippet
                    highlight = self._create_highlight(
                        result.content_preview, request.query
                    )

                    web_result = WebUISearchResult(
                        path=result.path,
                        name=result.name,
                        type=result.type,
                        content_preview=result.content_preview,
                        score=result.score,
                        entities=result.entities,
                        line_count=result.line_count,
                        complexity_score=result.complexity_score,
                        modified_time=result.modified_time,  # Keep as datetime, Pydantic will handle it
                        hash=result.hash,
                        module=result.module,
                        logical_unit=result.logical_unit,
                        entity_name=result.entity_name,
                        component=result.component,
                        pages=result.pages,
                        lists=result.lists,
                        groups=result.groups,
                        iconsets=result.iconsets,
                        trees=result.trees,
                        navigators=result.navigators,
                        contexts=result.contexts,
                        tags=tags,
                        highlight=highlight,
                    )
                    web_results.append(web_result)

                # Use Pydantic's model_dump with mode='json' for proper serialization
                response_data = {
                    "results": [
                        result.model_dump(mode="json") for result in web_results
                    ],
                    "total": len(web_results),
                    "query": request.query,
                }
                return JSONResponse(response_data)

            except Exception as e:
                logger.error(f"POST Search error: {type(e).__name__}: {e}")
                logger.error(
                    f"Search parameters: query='{request.query}', limit={request.limit}"
                )
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @self.app.get("/api/suggestions")
        async def get_suggestions(
            q: str = Query(..., description="Partial query for suggestions"),
            limit: int = Query(10, description="Maximum suggestions"),
        ):
            """Get type-ahead suggestions."""
            try:
                if len(q) < 2:  # Don't suggest for very short queries
                    return JSONResponse({"suggestions": []})

                # Check if index is properly initialized
                if self.indexer._index is None:
                    logger.error("Index is not initialized for suggestions")
                    return JSONResponse(
                        {"suggestions": [], "error": "Index not initialized"}
                    )

                # Check cache first
                cache_key = f"{q}:{limit}"
                if cache_key in self._suggestion_cache:
                    return JSONResponse(
                        {
                            "suggestions": [
                                s.model_dump(mode="json")
                                for s in self._suggestion_cache[cache_key]
                            ]
                        }
                    )

                suggestions = await self._generate_suggestions(q, limit)
                self._suggestion_cache[cache_key] = suggestions

                return JSONResponse(
                    {"suggestions": [s.model_dump(mode="json") for s in suggestions]}
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/suggestions")
        async def post_suggestions(request: SearchRequest):
            """Get type-ahead suggestions via POST."""
            try:
                if len(request.query) < 2:  # Don't suggest for very short queries
                    return JSONResponse({"suggestions": []})

                # Check if index is properly initialized
                if self.indexer._index is None:
                    logger.error("Index is not initialized for suggestions")
                    return JSONResponse(
                        {"suggestions": [], "error": "Index not initialized"}
                    )

                # Check cache first
                cache_key = f"{request.query}:{request.limit}"
                if cache_key in self._suggestion_cache:
                    return JSONResponse(
                        {
                            "suggestions": [
                                s.dict() for s in self._suggestion_cache[cache_key]
                            ]
                        }
                    )

                suggestions = await self._generate_suggestions(
                    request.query, request.limit
                )
                self._suggestion_cache[cache_key] = suggestions

                return JSONResponse(
                    {"suggestions": [s.model_dump(mode="json") for s in suggestions]}
                )

            except Exception as e:
                logger.error(f"Suggestions error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/stats")
        async def get_stats():
            """Get index statistics."""
            try:
                # Check if index is properly initialized
                if self.indexer._index is None:
                    return JSONResponse(
                        {
                            "total_files": 0,
                            "total_entities": 0,
                            "error": "Index not initialized. Please rebuild the index.",
                            "supported_extensions": list(
                                self.indexer.SUPPORTED_EXTENSIONS
                            ),
                            "index_path": str(self.indexer.index_path),
                        }
                    )

                stats = self.indexer.get_stats()
                return JSONResponse(stats)
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/index")
        async def rebuild_index(
            directory: str = Query(..., description="Directory to index")
        ):
            """Rebuild the index."""
            try:
                await self.indexer.index_directory(Path(directory))
                # Clear suggestion cache
                self._suggestion_cache.clear()
                return JSONResponse(
                    {"status": "success", "message": "Index rebuilt successfully"}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/file/{file_path:path}")
        async def get_file_content(file_path: str):
            """Get file content for preview."""
            try:
                path = Path(file_path)
                if not path.exists():
                    raise HTTPException(status_code=404, detail="File not found")

                content = path.read_text(encoding="utf-8", errors="ignore")
                return JSONResponse(
                    {
                        "path": str(path),
                        "content": content,
                        "size": len(content),
                        "lines": len(content.split("\n")),
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def _generate_suggestions(
        self, query: str, limit: int
    ) -> List[SearchSuggestion]:
        """Generate type-ahead suggestions."""
        suggestions = []

        # Use both exact and wildcard searches for better suggestions
        query_lower = query.lower()

        # Try multiple search strategies
        search_queries = [
            query,  # Exact query
            f"{query}*",  # Prefix search
            f"*{query}*",  # Contains search
        ]

        all_results = []
        for search_query in search_queries:
            try:
                results = self.indexer.search_deduplicated(
                    search_query, limit=limit * 3
                )
                all_results.extend(results)
                if len(all_results) >= limit * 2:  # Stop early if we have enough
                    break
            except:
                continue  # Try next search strategy

        # If no results from advanced search, try basic search
        if not all_results:
            try:
                all_results = self.indexer.search_deduplicated(query, limit=limit * 2)
            except:
                pass

        seen_suggestions = set()

        # Generate suggestions from results
        for result in all_results:
            if len(suggestions) >= limit:
                break

            # Add entity suggestions
            for entity in result.entities:
                if len(suggestions) >= limit:
                    break
                if (
                    query_lower in entity.lower()
                    and entity.lower() not in seen_suggestions
                ):
                    suggestions.append(
                        SearchSuggestion(
                            text=entity,
                            type="entity",
                            score=result.score,
                            context=f"in {result.name}",
                        )
                    )
                    seen_suggestions.add(entity.lower())

            # Add file name suggestions
            if (
                len(suggestions) < limit
                and query_lower in result.name.lower()
                and result.name.lower() not in seen_suggestions
            ):
                suggestions.append(
                    SearchSuggestion(
                        text=result.name,
                        type="file",
                        score=result.score,
                        context=f"{result.type} file",
                    )
                )
                seen_suggestions.add(result.name.lower())

            # Add module suggestions
            if (
                len(suggestions) < limit
                and result.module
                and query_lower in result.module.lower()
                and result.module.lower() not in seen_suggestions
            ):
                suggestions.append(
                    SearchSuggestion(
                        text=result.module,
                        type="module",
                        score=result.score,
                        context="IFS module",
                    )
                )
                seen_suggestions.add(result.module.lower())

            # Add frontend element suggestions with higher priority
            frontend_elements = [
                (result.pages, "page"),
                (result.lists, "list"),
                (result.groups, "group"),
                (result.iconsets, "iconset"),
                (result.trees, "tree"),
                (result.navigators, "navigator"),
                (result.contexts, "context"),
            ]

            for elements, element_type in frontend_elements:
                if len(suggestions) >= limit:
                    break
                for element in elements:
                    if len(suggestions) >= limit:
                        break
                    if (
                        query_lower in element.lower()
                        and element.lower() not in seen_suggestions
                    ):
                        suggestions.append(
                            SearchSuggestion(
                                text=element,
                                type="frontend",
                                score=result.score + 0.1,  # Boost frontend elements
                                context=f"{element_type} in {result.name}",
                            )
                        )
                        seen_suggestions.add(element.lower())

        # If still no suggestions, try common IFS terms
        if not suggestions and len(query) >= 2:
            common_terms = [
                "Activity",
                "Customer",
                "Order",
                "Project",
                "Delivery",
                "Structure",
                "Entity",
                "Projection",
                "Client",
                "Fragment",
                "Storage",
                "Views",
                "iconset",
                "tree",
                "navigator",
                "page",
                "list",
                "group",
            ]

            for term in common_terms:
                if query_lower in term.lower():
                    suggestions.append(
                        SearchSuggestion(
                            text=term,
                            type="common",
                            score=0.5,
                            context="common IFS term",
                        )
                    )
                if len(suggestions) >= limit:
                    break

        # Sort by score and limit
        suggestions.sort(key=lambda x: x.score, reverse=True)
        return suggestions[:limit]

    def _create_highlight(self, content: str, query: str) -> str:
        """Create highlighted content snippet."""
        if not query or not content:
            return content[:200] + "..." if len(content) > 200 else content

        # Simple highlighting - find query in content and add some context
        query_lower = query.lower()
        content_lower = content.lower()

        index = content_lower.find(query_lower)
        if index == -1:
            return content[:200] + "..." if len(content) > 200 else content

        # Get context around the match
        start = max(0, index - 50)
        end = min(len(content), index + len(query) + 50)

        snippet = content[start:end]

        # Add highlighting (will be handled by frontend)
        highlighted = snippet.replace(
            content[index : index + len(query)],
            f"<mark>{content[index:index + len(query)]}</mark>",
        )

        return highlighted


# Main application entry point
if __name__ == "__main__":
    import uvicorn

    # Create the web UI application
    web_ui = IFSCloudWebUI()

    print("🚀 Starting IFS Cloud Web UI...")
    print("📊 Interface will be available at: http://localhost:8000")
    print("🔍 Features:")
    print("  • Type-ahead search with intelligent suggestions")
    print("  • Frontend element discovery (pages, iconsets, trees, navigators)")
    print("  • Module-aware search and filtering")
    print("  • Real-time search with highlighting")
    print("  • Responsive design with modern UI")

    # Start the server
    uvicorn.run(web_ui.app, host="0.0.0.0", port=8000, reload=False)
