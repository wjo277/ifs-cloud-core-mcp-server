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
            return self.templates.TemplateResponse("index.html", {"request": request})

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

                results = self.indexer.search(
                    query=q,
                    limit=limit,
                    file_type=file_type,
                    min_complexity=min_complexity,
                    max_complexity=max_complexity,
                )

                logger.debug(f"Search returned {len(results)} results")

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

                return JSONResponse(
                    {
                        "results": [
                            {**r.dict(), "modified_time": r.modified_time.isoformat()}
                            for r in web_results
                        ],
                        "total": len(web_results),
                        "query": q,
                        "filters": {
                            "file_type": file_type,
                            "module": module,
                            "logical_unit": logical_unit,
                            "complexity_range": [min_complexity, max_complexity],
                        },
                    }
                )

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
                results = self.indexer.search(
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

                return JSONResponse(
                    {
                        "results": [result.dict() for result in web_results],
                        "total": len(web_results),
                        "query": request.query,
                        "took_ms": 0,  # We could add timing if needed
                    }
                )

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
                results = self.indexer.search(
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
                        modified_time=(
                            result.modified_time.isoformat()
                            if result.modified_time
                            else None
                        ),
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

                return JSONResponse(
                    {
                        "results": [result.dict() for result in web_results],
                        "total": len(web_results),
                        "query": request.query,
                    }
                )

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
                                s.dict() for s in self._suggestion_cache[cache_key]
                            ]
                        }
                    )

                suggestions = await self._generate_suggestions(q, limit)
                self._suggestion_cache[cache_key] = suggestions

                return JSONResponse({"suggestions": [s.dict() for s in suggestions]})

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

                return JSONResponse({"suggestions": [s.dict() for s in suggestions]})

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
                results = self.indexer.search(search_query, limit=limit * 3)
                all_results.extend(results)
                if len(all_results) >= limit * 2:  # Stop early if we have enough
                    break
            except:
                continue  # Try next search strategy

        # If no results from advanced search, try basic search
        if not all_results:
            try:
                all_results = self.indexer.search(query, limit=limit * 2)
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


def create_web_ui_files():
    """Create the HTML template and CSS files for the web UI."""

    # Create templates directory
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # Create main HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IFS Cloud Explorer</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
</head>
<body class="bg-gray-50">
    <div x-data="searchApp()" class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-4xl font-bold text-gray-900 mb-2">
                <i class="fas fa-search text-blue-600"></i>
                IFS Cloud Explorer
            </h1>
            <p class="text-gray-600">Intelligent search for IFS Cloud codebases with frontend element discovery</p>
        </div>

        <!-- Search Section -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <div class="relative">
                <!-- Search Input -->
                <div class="relative">
                    <input 
                        type="text" 
                        x-model="query" 
                        @input="handleInput"
                        @keydown.enter="search"
                        @keydown.arrow-down="selectSuggestion(1)"
                        @keydown.arrow-up="selectSuggestion(-1)"
                        @keydown.escape="hideSuggestions"
                        @focus="showSuggestions = true"
                        placeholder="Search entities, functions, pages, iconsets, trees, navigators..." 
                        class="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
                    >
                    <i class="fas fa-search absolute left-4 top-4 text-gray-400"></i>
                    <button 
                        @click="search" 
                        class="absolute right-2 top-2 bg-blue-600 text-white px-4 py-1 rounded hover:bg-blue-700"
                    >
                        Search
                    </button>
                </div>

                <!-- Type-ahead Suggestions -->
                <div 
                    x-show="showSuggestions && suggestions.length > 0" 
                    @click.away="hideSuggestions"
                    class="absolute z-10 w-full bg-white border border-gray-300 rounded-lg mt-1 max-h-64 overflow-y-auto shadow-lg"
                >
                    <template x-for="(suggestion, index) in suggestions" :key="index">
                        <div 
                            @click="selectSuggestionText(suggestion.text)"
                            :class="{'bg-blue-50': selectedSuggestion === index}"
                            class="px-4 py-2 hover:bg-gray-100 cursor-pointer border-b border-gray-100 last:border-b-0"
                        >
                            <div class="flex items-center justify-between">
                                <div>
                                    <span class="font-medium" x-text="suggestion.text"></span>
                                    <span class="text-sm text-gray-500 ml-2" x-text="suggestion.context"></span>
                                </div>
                                <span 
                                    :class="{
                                        'bg-blue-100 text-blue-800': suggestion.type === 'entity',
                                        'bg-green-100 text-green-800': suggestion.type === 'file',
                                        'bg-purple-100 text-purple-800': suggestion.type === 'module',
                                        'bg-orange-100 text-orange-800': suggestion.type === 'frontend'
                                    }"
                                    class="px-2 py-1 rounded text-xs font-medium"
                                    x-text="suggestion.type"
                                ></span>
                            </div>
                        </div>
                    </template>
                </div>
            </div>

            <!-- Filters -->
            <div class="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
                <select x-model="filters.file_type" class="border border-gray-300 rounded px-3 py-2">
                    <option value="">All File Types</option>
                    <option value=".entity">Entity</option>
                    <option value=".client">Client</option>
                    <option value=".projection">Projection</option>
                    <option value=".fragment">Fragment</option>
                    <option value=".plsql">PL/SQL</option>
                    <option value=".views">Views</option>
                    <option value=".storage">Storage</option>
                </select>
                
                <input x-model="filters.module" placeholder="Module filter..." class="border border-gray-300 rounded px-3 py-2">
                <input x-model="filters.logical_unit" placeholder="Logical Unit filter..." class="border border-gray-300 rounded px-3 py-2">
                <input x-model="filters.limit" type="number" placeholder="Result limit..." class="border border-gray-300 rounded px-3 py-2" value="20">
            </div>
        </div>

        <!-- Stats -->
        <div x-show="stats" class="bg-white rounded-lg shadow-md p-4 mb-6">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                <div>
                    <div class="text-2xl font-bold text-blue-600" x-text="stats.total_files"></div>
                    <div class="text-sm text-gray-600">Files Indexed</div>
                </div>
                <div>
                    <div class="text-2xl font-bold text-green-600" x-text="stats.total_entities"></div>
                    <div class="text-sm text-gray-600">Entities</div>
                </div>
                <div>
                    <div class="text-2xl font-bold text-purple-600" x-text="stats.supported_extensions?.length || 0"></div>
                    <div class="text-sm text-gray-600">File Types</div>
                </div>
                <div>
                    <div class="text-2xl font-bold text-orange-600" x-text="results.length"></div>
                    <div class="text-sm text-gray-600">Results</div>
                </div>
            </div>
        </div>

        <!-- Loading -->
        <div x-show="loading" class="text-center py-8">
            <i class="fas fa-spinner fa-spin text-3xl text-blue-600"></i>
            <p class="mt-2 text-gray-600">Searching...</p>
        </div>

        <!-- Results -->
        <div x-show="results.length > 0 && !loading" class="space-y-4">
            <template x-for="result in results" :key="result.path">
                <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                    <!-- File Header -->
                    <div class="flex items-start justify-between mb-3">
                        <div>
                            <h3 class="text-lg font-semibold text-gray-900" x-text="result.name"></h3>
                            <p class="text-sm text-gray-600" x-text="result.path"></p>
                        </div>
                        <div class="flex items-center space-x-2">
                            <span class="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs font-medium" x-text="result.type"></span>
                            <span class="text-sm text-gray-500">Score: <span x-text="result.score.toFixed(3)"></span></span>
                        </div>
                    </div>

                    <!-- Tags -->
                    <div x-show="result.tags && result.tags.length > 0" class="mb-3">
                        <template x-for="tag in result.tags" :key="tag">
                            <span class="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium mr-1 mb-1" x-text="tag"></span>
                        </template>
                    </div>

                    <!-- IFS Context -->
                    <div x-show="result.module || result.logical_unit || result.component" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3 text-sm">
                        <div x-show="result.module">
                            <span class="font-medium text-gray-700">Module:</span> 
                            <span x-text="result.module"></span>
                        </div>
                        <div x-show="result.logical_unit">
                            <span class="font-medium text-gray-700">Logical Unit:</span> 
                            <span x-text="result.logical_unit"></span>
                        </div>
                        <div x-show="result.component">
                            <span class="font-medium text-gray-700">Component:</span> 
                            <span x-text="result.component"></span>
                        </div>
                    </div>

                    <!-- Frontend Elements -->
                    <div x-show="result.pages?.length > 0 || result.iconsets?.length > 0 || result.trees?.length > 0 || result.navigators?.length > 0" class="mb-3">
                        <h4 class="font-medium text-gray-700 mb-2">Frontend Elements:</h4>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                            <div x-show="result.pages?.length > 0">
                                <span class="font-medium text-blue-600">Pages:</span> 
                                <span x-text="result.pages.join(', ')"></span>
                            </div>
                            <div x-show="result.lists?.length > 0">
                                <span class="font-medium text-green-600">Lists:</span> 
                                <span x-text="result.lists.join(', ')"></span>
                            </div>
                            <div x-show="result.iconsets?.length > 0">
                                <span class="font-medium text-purple-600">Iconsets:</span> 
                                <span x-text="result.iconsets.join(', ')"></span>
                            </div>
                            <div x-show="result.trees?.length > 0">
                                <span class="font-medium text-orange-600">Trees:</span> 
                                <span x-text="result.trees.join(', ')"></span>
                            </div>
                            <div x-show="result.navigators?.length > 0">
                                <span class="font-medium text-red-600">Navigators:</span> 
                                <span x-text="result.navigators.join(', ')"></span>
                            </div>
                        </div>
                    </div>

                    <!-- Entities -->
                    <div x-show="result.entities && result.entities.length > 0" class="mb-3">
                        <span class="font-medium text-gray-700">Entities:</span> 
                        <span class="text-sm" x-text="result.entities.join(', ')"></span>
                    </div>

                    <!-- Content Preview -->
                    <div class="bg-gray-50 rounded p-3">
                        <pre class="text-sm text-gray-700 whitespace-pre-wrap" x-html="result.highlight || result.content_preview"></pre>
                    </div>

                    <!-- Metadata -->
                    <div class="mt-3 flex justify-between text-xs text-gray-500">
                        <span>Lines: <span x-text="result.line_count"></span></span>
                        <span>Complexity: <span x-text="result.complexity_score.toFixed(2)"></span></span>
                        <span>Modified: <span x-text="new Date(result.modified_time).toLocaleDateString()"></span></span>
                    </div>
                </div>
            </template>
        </div>

        <!-- No Results -->
        <div x-show="results.length === 0 && searchPerformed && !loading" class="text-center py-8">
            <i class="fas fa-search text-4xl text-gray-400 mb-4"></i>
            <h3 class="text-lg font-medium text-gray-900 mb-2">No results found</h3>
            <p class="text-gray-600">Try adjusting your search terms or filters</p>
        </div>
    </div>

    <script>
        function searchApp() {
            return {
                query: '',
                results: [],
                suggestions: [],
                stats: null,
                loading: false,
                searchPerformed: false,
                showSuggestions: false,
                selectedSuggestion: -1,
                filters: {
                    file_type: '',
                    module: '',
                    logical_unit: '',
                    limit: 20
                },

                async init() {
                    await this.loadStats();
                },

                async handleInput() {
                    if (this.query.length >= 2) {
                        await this.getSuggestions();
                        this.showSuggestions = true;
                    } else {
                        this.suggestions = [];
                        this.showSuggestions = false;
                    }
                    this.selectedSuggestion = -1;
                },

                async getSuggestions() {
                    try {
                        const response = await fetch('/suggestions', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                query: this.query,
                                limit: 8
                            })
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            this.suggestions = data.suggestions || [];
                        } else {
                            console.warn('Suggestions request failed:', response.status);
                            this.suggestions = [];
                        }
                    } catch (error) {
                        console.error('Error getting suggestions:', error);
                        this.suggestions = [];
                    }
                },

                selectSuggestion(direction) {
                    if (this.suggestions.length === 0) return;
                    
                    this.selectedSuggestion += direction;
                    if (this.selectedSuggestion >= this.suggestions.length) {
                        this.selectedSuggestion = 0;
                    } else if (this.selectedSuggestion < 0) {
                        this.selectedSuggestion = this.suggestions.length - 1;
                    }
                },

                selectSuggestionText(text) {
                    this.query = text;
                    this.hideSuggestions();
                    this.search();
                },

                hideSuggestions() {
                    this.showSuggestions = false;
                    this.selectedSuggestion = -1;
                },

                async search() {
                    if (!this.query.trim()) return;

                    this.loading = true;
                    this.hideSuggestions();

                    try {
                        const searchData = {
                            query: this.query,
                            limit: this.filters.limit
                        };

                        // Add filters if specified
                        if (this.filters.file_type) searchData.file_type = this.filters.file_type;
                        if (this.filters.module) searchData.module = this.filters.module;
                        if (this.filters.logical_unit) searchData.logical_unit = this.filters.logical_unit;

                        const response = await fetch('/search', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(searchData)
                        });

                        if (response.ok) {
                            const data = await response.json();
                            this.results = data.results || [];
                            this.searchPerformed = true;
                        } else {
                            const errorText = await response.text();
                            console.error('Search failed:', response.status, errorText);
                            this.results = [];
                        }
                    } catch (error) {
                        console.error('Search error:', error);
                    } finally {
                        this.loading = false;
                    }
                },

                async loadStats() {
                    try {
                        const response = await fetch('/api/stats');
                        this.stats = await response.json();
                    } catch (error) {
                        console.error('Error loading stats:', error);
                    }
                }
            }
        }
    </script>
</body>
</html>"""

    with open(templates_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_template)


# Main application entry point
if __name__ == "__main__":
    import uvicorn

    # Create web UI files
    create_web_ui_files()

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
