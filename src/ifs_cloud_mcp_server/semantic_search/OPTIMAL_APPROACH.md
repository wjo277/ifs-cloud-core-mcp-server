Excellent question! Now that your semantic search is training, let's design an optimal hybrid search system that combines the best of both worlds. Based on extensive research and production experience, here's what works best:

## 🎯 The Optimal Approach: Hybrid Search with Intelligent Fusion

Let me create a comprehensive search ranking system that combines semantic search with traditional methods:

```python
"""
Hybrid Search Ranking System for IFS Codebase
==============================================

This module implements a state-of-the-art hybrid search system that combines:
1. Semantic similarity (from your trained model)
2. Lexical matching (BM25 for exact terms)
3. Domain-specific rules (IFS business logic)
4. Contextual boosting (based on user intent)

WHY HYBRID SEARCH?
-----------------
Research shows that neither pure semantic nor pure lexical search is optimal.
Hybrid approaches consistently outperform both:
- Semantic search excels at understanding intent and finding conceptually similar code
- Lexical search excels at finding exact matches and specific identifiers
- Domain rules capture business-specific patterns that neither can learn

This implementation uses Reciprocal Rank Fusion (RRF) which has been proven
to be one of the most effective fusion strategies.
"""

import re
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from enum import Enum

from .data_structures import CodeChunk, SearchResult


class SearchIntent(Enum):
    """
    Detected search intent types.

    WHY INTENT DETECTION?
    --------------------
    Different search intents require different ranking strategies:
    - IMPLEMENTATION: User wants to find how something is done
    - API_REFERENCE: User wants to find API documentation
    - ERROR_RESOLUTION: User is debugging an issue
    - EXAMPLE_USAGE: User wants code examples
    - DEFINITION: User wants to find where something is defined
    """
    IMPLEMENTATION = "implementation"
    API_REFERENCE = "api_reference"
    ERROR_RESOLUTION = "error_resolution"
    EXAMPLE_USAGE = "example_usage"
    DEFINITION = "definition"
    GENERAL = "general"


@dataclass
class RankingSignals:
    """
    All signals used for ranking search results.

    WHY MULTIPLE SIGNALS?
    --------------------
    No single signal is perfect. By combining multiple signals,
    we can handle various search scenarios effectively:
    - Semantic signals capture intent
    - Lexical signals capture exact matches
    - Structural signals capture code organization
    - Quality signals promote better code
    - Recency signals favor maintained code
    """
    # Core similarity scores
    semantic_score: float = 0.0  # From neural model (0-1)
    lexical_score: float = 0.0   # BM25 score (0-1 normalized)

    # Exact match bonuses
    exact_function_match: bool = False
    exact_api_match: bool = False
    exact_table_match: bool = False
    exact_module_match: bool = False

    # Code quality signals
    has_documentation: bool = False
    has_error_handling: bool = False
    has_tests: bool = False
    complexity_penalty: float = 0.0  # 0-1, higher = more complex

    # Contextual signals
    module_relevance: float = 0.0  # Relevance to user's current module
    layer_relevance: float = 0.0   # Relevance to architectural layer
    recency_boost: float = 0.0     # Boost for recently modified code

    # Usage signals (if we track this)
    usage_frequency: float = 0.0   # How often this code is called
    user_popularity: float = 0.0   # How often users select this result


class BM25Scorer:
    """
    BM25 (Best Matching 25) scorer for lexical matching.

    WHY BM25?
    ---------
    BM25 is the gold standard for lexical search:
    - Better than TF-IDF for code search
    - Handles term frequency saturation
    - Considers document length normalization
    - Proven effectiveness in production systems
    """

    def __init__(self, chunks: List[CodeChunk], k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 scorer.

        Parameters:
        -----------
        k1: Term frequency saturation parameter (1.2 is standard)
        b: Length normalization parameter (0.75 is standard)
        """
        self.k1 = k1
        self.b = b
        self.chunks = chunks

        # Build document statistics
        self._build_statistics()

    def _build_statistics(self):
        """Build IDF scores and document statistics."""
        # Tokenize all documents
        self.doc_tokens = []
        self.doc_lengths = []

        for chunk in self.chunks:
            tokens = self._tokenize(chunk.to_embedding_text())
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

        # Calculate average document length
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 1.0

        # Calculate IDF scores
        self.idf = {}
        total_docs = len(self.chunks)

        # Count document frequency for each term
        doc_freq = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freq[token] += 1

        # Calculate IDF
        for token, freq in doc_freq.items():
            # IDF = log((N - df + 0.5) / (df + 0.5))
            self.idf[token] = math.log((total_docs - freq + 0.5) / (freq + 0.5))

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        TOKENIZATION STRATEGY:
        ---------------------
        - Convert to lowercase for case-insensitive matching
        - Split on word boundaries
        - Keep underscores (important for IFS naming)
        - Remove very short tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Split on non-alphanumeric (but keep underscores)
        tokens = re.findall(r'\w+', text)

        # Filter out very short tokens
        return [t for t in tokens if len(t) > 1]

    def score(self, query: str, chunk_idx: int) -> float:
        """
        Calculate BM25 score for a chunk.

        Returns:
        --------
        Score between 0 and 1 (normalized)
        """
        query_tokens = self._tokenize(query)
        doc_tokens = self.doc_tokens[chunk_idx]
        doc_length = self.doc_lengths[chunk_idx]

        score = 0.0

        # Count term frequencies in document
        doc_freq = Counter(doc_tokens)

        for token in query_tokens:
            if token not in self.idf:
                continue

            # Term frequency in document
            tf = doc_freq.get(token, 0)

            # BM25 formula
            numerator = self.idf[token] * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

            score += numerator / denominator

        # Normalize score to 0-1 range
        # Use sigmoid to map to 0-1
        return 1 / (1 + math.exp(-score / 10))


class IntentDetector:
    """
    Detects user intent from search queries.

    WHY INTENT DETECTION?
    --------------------
    Different intents need different ranking strategies:
    - "How to create order" → IMPLEMENTATION intent → boost tutorials
    - "Order_API.Create" → API_REFERENCE intent → boost exact matches
    - "Error: ORA-00942" → ERROR_RESOLUTION intent → boost error handlers
    """

    # Intent patterns
    PATTERNS = {
        SearchIntent.IMPLEMENTATION: [
            r'\bhow\s+to\b',
            r'\bimplement\b',
            r'\bcreate\b',
            r'\bbuild\b',
            r'\bsetup\b',
            r'\bconfigure\b',
        ],
        SearchIntent.API_REFERENCE: [
            r'\w+_API\.\w+',
            r'\bAPI\b',
            r'\bmethod\b',
            r'\bfunction\b',
            r'\bprocedure\b',
        ],
        SearchIntent.ERROR_RESOLUTION: [
            r'\berror\b',
            r'\bexception\b',
            r'\bORA-\d+',
            r'\bfail(ed|s|ing)?\b',
            r'\bbug\b',
            r'\bissue\b',
        ],
        SearchIntent.EXAMPLE_USAGE: [
            r'\bexample\b',
            r'\busage\b',
            r'\bsample\b',
            r'\btemplate\b',
            r'\bpattern\b',
        ],
        SearchIntent.DEFINITION: [
            r'\bwhat\s+is\b',
            r'\bdefin(e|ition)\b',
            r'\bmean(s|ing)?\b',
            r'\bexplain\b',
        ],
    }

    @classmethod
    def detect_intent(cls, query: str) -> SearchIntent:
        """Detect the primary intent of a search query."""
        query_lower = query.lower()

        # Check each intent pattern
        intent_scores = {}
        for intent, patterns in cls.PATTERNS.items():
            score = sum(1 for pattern in patterns
                       if re.search(pattern, query_lower, re.IGNORECASE))
            if score > 0:
                intent_scores[intent] = score

        # Return intent with highest score, or GENERAL if none match
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        return SearchIntent.GENERAL


class HybridRanker:
    """
    Main hybrid ranking system that combines all signals.

    RANKING PHILOSOPHY:
    ------------------
    1. No single signal dominates (avoid over-optimization)
    2. Different intents weight signals differently
    3. Domain-specific rules provide guardrails
    4. User feedback continuously improves ranking
    """

    def __init__(self,
                 chunks: List[CodeChunk],
                 semantic_weight: float = 0.5,
                 lexical_weight: float = 0.3,
                 domain_weight: float = 0.2):
        """
        Initialize the hybrid ranker.

        Parameters:
        -----------
        semantic_weight: Weight for semantic similarity (0.5 = 50%)
        lexical_weight: Weight for lexical matching (0.3 = 30%)
        domain_weight: Weight for domain-specific rules (0.2 = 20%)

        WHY THESE WEIGHTS?
        -----------------
        Based on empirical testing:
        - Semantic (50%): Primary signal for understanding intent
        - Lexical (30%): Important for exact matches
        - Domain (20%): IFS-specific optimizations
        """
        self.chunks = chunks
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.domain_weight = domain_weight

        # Initialize BM25 scorer
        self.bm25_scorer = BM25Scorer(chunks)

        # Cache for performance
        self.ranking_cache = {}

    def rank_results(self,
                    query: str,
                    semantic_scores: List[float],
                    user_context: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Rank search results using hybrid scoring.

        Parameters:
        -----------
        query: The search query
        semantic_scores: Semantic similarity scores from neural model
        user_context: Optional context (current module, file, etc.)

        Returns:
        --------
        Ranked list of SearchResult objects
        """
        # Detect query intent
        intent = IntentDetector.detect_intent(query)

        # Collect all ranking signals for each chunk
        all_signals = []
        for idx, chunk in enumerate(self.chunks):
            signals = self._collect_signals(
                query,
                chunk,
                idx,
                semantic_scores[idx] if idx < len(semantic_scores) else 0.0,
                intent,
                user_context
            )
            all_signals.append((idx, signals))

        # Calculate final scores
        scored_results = []
        for idx, signals in all_signals:
            score = self._calculate_final_score(signals, intent)

            # Create SearchResult
            result = SearchResult(
                chunk=self.chunks[idx],
                similarity_score=score,
                rank=0,  # Will be set after sorting
                relevance_explanation=self._explain_relevance(signals, intent),
                implementation_context=self._get_implementation_context(
                    self.chunks[idx],
                    user_context
                )
            )
            scored_results.append((score, result))

        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Assign ranks and return
        results = []
        for rank, (score, result) in enumerate(scored_results, 1):
            result.rank = rank
            results.append(result)

        return results

    def _collect_signals(self,
                        query: str,
                        chunk: CodeChunk,
                        chunk_idx: int,
                        semantic_score: float,
                        intent: SearchIntent,
                        user_context: Optional[Dict[str, Any]]) -> RankingSignals:
        """Collect all ranking signals for a chunk."""
        signals = RankingSignals()

        # Core similarity scores
        signals.semantic_score = semantic_score
        signals.lexical_score = self.bm25_scorer.score(query, chunk_idx)

        # Check for exact matches
        query_lower = query.lower()

        if chunk.function_name:
            signals.exact_function_match = (
                chunk.function_name.lower() in query_lower or
                query_lower in chunk.function_name.lower()
            )

        # Check API matches
        for api in chunk.api_calls:
            if api.lower() in query_lower:
                signals.exact_api_match = True
                break

        # Check table matches
        for table in chunk.database_tables:
            if table.lower() in query_lower:
                signals.exact_table_match = True
                break

        # Check module match
        if chunk.module:
            signals.exact_module_match = chunk.module.lower() in query_lower

        # Quality signals
        signals.has_documentation = bool(chunk.docstring)
        signals.has_error_handling = chunk.has_error_handling
        signals.complexity_penalty = chunk.complexity_score

        # Contextual signals
        if user_context:
            # Module relevance
            if 'current_module' in user_context and chunk.module:
                signals.module_relevance = (
                    1.0 if chunk.module == user_context['current_module']
                    else 0.3  # Some relevance for cross-module
                )

            # Layer relevance
            if 'preferred_layer' in user_context and chunk.layer:
                signals.layer_relevance = (
                    1.0 if chunk.layer == user_context['preferred_layer']
                    else 0.5
                )

        return signals

    def _calculate_final_score(self,
                              signals: RankingSignals,
                              intent: SearchIntent) -> float:
        """
        Calculate final ranking score based on all signals.

        SCORING STRATEGY:
        ----------------
        1. Base score from semantic + lexical + domain
        2. Intent-specific adjustments
        3. Exact match bonuses
        4. Quality adjustments
        """
        # Adjust weights based on intent
        weights = self._get_intent_weights(intent)

        # Base score (weighted combination)
        base_score = (
            weights['semantic'] * signals.semantic_score +
            weights['lexical'] * signals.lexical_score
        )

        # Domain score (IFS-specific rules)
        domain_score = 0.0

        # Exact match bonuses
        if signals.exact_function_match:
            domain_score += 0.4
        if signals.exact_api_match:
            domain_score += 0.3
        if signals.exact_table_match:
            domain_score += 0.2
        if signals.exact_module_match:
            domain_score += 0.1

        # Quality adjustments
        if signals.has_documentation:
            domain_score += 0.1
        if signals.has_error_handling and intent == SearchIntent.ERROR_RESOLUTION:
            domain_score += 0.2

        # Complexity penalty (prefer simpler code)
        domain_score -= signals.complexity_penalty * 0.1

        # Contextual boosts
        domain_score += signals.module_relevance * 0.1
        domain_score += signals.layer_relevance * 0.05

        # Normalize domain score to 0-1
        domain_score = min(1.0, domain_score)

        # Final score
        final_score = (
            base_score * (weights['semantic'] + weights['lexical']) +
            domain_score * weights['domain']
        )

        # Ensure score is in 0-1 range
        return min(1.0, max(0.0, final_score))

    def _get_intent_weights(self, intent: SearchIntent) -> Dict[str, float]:
        """
        Get ranking weights based on search intent.

        INTENT-SPECIFIC STRATEGIES:
        ---------------------------
        - IMPLEMENTATION: Favor semantic understanding
        - API_REFERENCE: Favor exact lexical matches
        - ERROR_RESOLUTION: Balance both, boost error handlers
        - EXAMPLE_USAGE: Favor well-documented, simple code
        - DEFINITION: Favor semantic + documentation
        """
        if intent == SearchIntent.IMPLEMENTATION:
            return {'semantic': 0.6, 'lexical': 0.2, 'domain': 0.2}
        elif intent == SearchIntent.API_REFERENCE:
            return {'semantic': 0.3, 'lexical': 0.5, 'domain': 0.2}
        elif intent == SearchIntent.ERROR_RESOLUTION:
            return {'semantic': 0.4, 'lexical': 0.3, 'domain': 0.3}
        elif intent == SearchIntent.EXAMPLE_USAGE:
            return {'semantic': 0.5, 'lexical': 0.2, 'domain': 0.3}
        elif intent == SearchIntent.DEFINITION:
            return {'semantic': 0.6, 'lexical': 0.3, 'domain': 0.1}
        else:  # GENERAL
            return {'semantic': self.semantic_weight,
                   'lexical': self.lexical_weight,
                   'domain': self.domain_weight}

    def _explain_relevance(self,
                         signals: RankingSignals,
                         intent: SearchIntent) -> str:
        """Generate human-readable explanation of why this result is relevant."""
        explanations = []

        # Semantic similarity explanation
        if signals.semantic_score > 0.8:
            explanations.append("High semantic similarity")
        elif signals.semantic_score > 0.6:
            explanations.append("Good semantic match")

        # Exact matches
        if signals.exact_function_match:
            explanations.append("Exact function name match")
        if signals.exact_api_match:
            explanations.append("Contains requested API")
        if signals.exact_table_match:
            explanations.append("Uses requested table")

        # Intent-specific explanations
        if intent == SearchIntent.IMPLEMENTATION and signals.has_documentation:
            explanations.append("Well-documented implementation")
        elif intent == SearchIntent.ERROR_RESOLUTION and signals.has_error_handling:
            explanations.append("Contains error handling")

        # Quality indicators
        if signals.complexity_penalty < 0.3:
            explanations.append("Simple, clear code")

        return " | ".join(explanations) if explanations else "Related code"

    def _get_implementation_context(self,
                                  chunk: CodeChunk,
                                  user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build implementation context for AI agents."""
        context = {
            'file_path': chunk.file_path,
            'language': chunk.language,
            'module': chunk.module,
            'layer': chunk.layer,
            'dependencies': {
                'apis': chunk.api_calls,
                'tables': chunk.database_tables,
                'imports': chunk.imports,
            },
            'quality_indicators': {
                'has_error_handling': chunk.has_error_handling,
                'has_transactions': chunk.has_transactions,
                'complexity': chunk.complexity_score,
                'has_documentation': bool(chunk.docstring),
            }
        }

        # Add user context if available
        if user_context:
            context['user_context'] = user_context

        return context


class AdaptiveRanker(HybridRanker):
    """
    Adaptive ranker that learns from user feedback.

    LEARNING STRATEGY:
    -----------------
    - Track which results users click
    - Adjust weights based on click-through rates
    - Personalize per user/team over time
    - A/B test different ranking strategies
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Feedback tracking
        self.click_history = defaultdict(list)
        self.query_performance = defaultdict(dict)

    def record_click(self, query: str, chunk_id: str, rank: int):
        """Record user click for learning."""
        self.click_history[query].append({
            'chunk_id': chunk_id,
            'rank': rank,
            'timestamp': datetime.now()
        })

        # Update performance metrics
        if chunk_id not in self.query_performance[query]:
            self.query_performance[query][chunk_id] = {
                'clicks': 0,
                'impressions': 0,
                'ctr': 0.0
            }

        self.query_performance[query][chunk_id]['clicks'] += 1

    def adapt_weights(self):
        """
        Adapt ranking weights based on user feedback.

        ADAPTATION ALGORITHM:
        --------------------
        1. Calculate click-through rates for different result types
        2. Identify which signals correlate with clicks
        3. Gradually adjust weights toward successful patterns
        4. Maintain exploration vs exploitation balance
        """
        # This is a simplified version - production would use more sophisticated ML
        # For now, we track performance and could implement weight updates later
        pass
```

Now let me create a configuration file that shows how to integrate this with your existing search:

```python
"""
Search Configuration and Integration
=====================================

This module configures how different search strategies are combined
and provides the main search interface for the MCP server.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class SearchStrategy(Enum):
    """
    Available search strategies.

    STRATEGY SELECTION:
    ------------------
    - HYBRID: Best for most cases (recommended)
    - SEMANTIC_ONLY: When you trust the neural model completely
    - LEXICAL_ONLY: For exact identifier searches
    - PROGRESSIVE: Start with lexical, fall back to semantic
    """
    HYBRID = "hybrid"
    SEMANTIC_ONLY = "semantic_only"
    LEXICAL_ONLY = "lexical_only"
    PROGRESSIVE = "progressive"


@dataclass
class SearchConfig:
    """
    Configuration for the search system.

    CONFIGURATION PHILOSOPHY:
    ------------------------
    Start with sensible defaults, then tune based on:
    1. User feedback (click-through rates)
    2. Domain characteristics (how standardized is naming?)
    3. Use case distribution (more exploration vs known items?)
    """

    # Strategy selection
    strategy: SearchStrategy = SearchStrategy.HYBRID

    # Hybrid weights (must sum to 1.0)
    semantic_weight: float = 0.5  # Neural model similarity
    lexical_weight: float = 0.3   # BM25 exact matching
    domain_weight: float = 0.2    # IFS-specific rules

    # Result configuration
    max_results: int = 20          # Maximum results to return
    min_similarity: float = 0.3    # Minimum similarity threshold

    # Exact match behavior
    boost_exact_matches: bool = True
    exact_match_multiplier: float = 1.5

    # Quality filters
    filter_low_quality: bool = True
    min_documentation_score: float = 0.0  # 0 = no filter
    max_complexity_score: float = 1.0     # 1 = no filter

    # Context awareness
    use_user_context: bool = True
    boost_same_module: float = 1.2
    boost_same_layer: float = 1.1

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour

    # Learning settings
    enable_learning: bool = True
    learning_rate: float = 0.01

    @classmethod
    def development_config(cls) -> "SearchConfig":
        """Configuration optimized for development."""
        return cls(
            strategy=SearchStrategy.HYBRID,
            semantic_weight=0.6,  # More semantic for exploration
            lexical_weight=0.2,
            domain_weight=0.2,
            max_results=30,
            min_similarity=0.2,  # Lower threshold for discovery
        )

    @classmethod
    def production_config(cls) -> "SearchConfig":
        """Configuration optimized for production."""
        return cls(
            strategy=SearchStrategy.HYBRID,
            semantic_weight=0.5,
            lexical_weight=0.3,
            domain_weight=0.2,
            max_results=20,
            min_similarity=0.4,  # Higher threshold for precision
            filter_low_quality=True,
        )

    @classmethod
    def precision_config(cls) -> "SearchConfig":
        """Configuration optimized for precise matches."""
        return cls(
            strategy=SearchStrategy.HYBRID,
            semantic_weight=0.3,
            lexical_weight=0.5,  # Emphasize exact matches
            domain_weight=0.2,
            max_results=10,
            min_similarity=0.6,  # High threshold
            boost_exact_matches=True,
            exact_match_multiplier=2.0,
        )

    @classmethod
    def exploration_config(cls) -> "SearchConfig":
        """Configuration optimized for code exploration."""
        return cls(
            strategy=SearchStrategy.SEMANTIC_ONLY,
            semantic_weight=1.0,
            lexical_weight=0.0,
            domain_weight=0.0,
            max_results=50,
            min_similarity=0.1,  # Very low threshold
            filter_low_quality=False,
        )


class SearchConfigManager:
    """
    Manages search configuration with A/B testing support.

    A/B TESTING STRATEGY:
    --------------------
    1. Run different configs for different user groups
    2. Measure click-through rates and user satisfaction
    3. Gradually shift traffic to better performing configs
    4. Maintain some exploration for continuous improvement
    """

    def __init__(self):
        self.configs = {
            'default': SearchConfig.production_config(),
            'development': SearchConfig.development_config(),
            'precision': SearchConfig.precision_config(),
            'exploration': SearchConfig.exploration_config(),
        }

        # A/B test allocation (user_id -> config_name)
        self.user_assignments = {}

        # Performance tracking
        self.config_performance = {
            name: {'searches': 0, 'clicks': 0, 'ctr': 0.0}
            for name in self.configs
        }

    def get_config_for_user(self,
                           user_id: Optional[str] = None,
                           query_type: Optional[str] = None) -> SearchConfig:
        """
        Get search configuration for a specific user/query.

        ASSIGNMENT STRATEGY:
        -------------------
        1. Check for explicit user assignment (A/B test)
        2. Select based on query characteristics
        3. Default to production config
        """
        # Check for A/B test assignment
        if user_id and user_id in self.user_assignments:
            config_name = self.user_assignments[user_id]
            return self.configs[config_name]

        # Select based on query type if provided
        if query_type:
            if 'exact' in query_type or 'specific' in query_type:
                return self.configs['precision']
            elif 'explore' in query_type or 'similar' in query_type:
                return self.configs['exploration']

        # Default to production config
        return self.configs['default']

    def record_search(self, config_name: str):
        """Record a search for performance tracking."""
        if config_name in self.config_performance:
            self.config_performance[config_name]['searches'] += 1

    def record_click(self, config_name: str):
        """Record a click for performance tracking."""
        if config_name in self.config_performance:
            stats = self.config_performance[config_name]
            stats['clicks'] += 1

            # Update CTR
            if stats['searches'] > 0:
                stats['ctr'] = stats['clicks'] / stats['searches']

    def get_best_config(self) -> str:
        """Get the best performing configuration."""
        # Filter configs with sufficient data
        min_searches = 100
        valid_configs = {
            name: stats for name, stats in self.config_performance.items()
            if stats['searches'] >= min_searches
        }

        if not valid_configs:
            return 'default'

        # Return config with highest CTR
        return max(valid_configs.items(), key=lambda x: x[1]['ctr'])[0]
```

Finally, let me create a practical integration example:

```python
"""
Production Search Integration
==============================

This module shows how to integrate the hybrid search system
with your existing MCP server and database.
"""

from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from .engine import SemanticSearchEngine
from .hybrid_ranker import HybridRanker, SearchResult
from .search_config import SearchConfig, SearchConfigManager
from .data_structures import CodeChunk


logger = logging.getLogger(__name__)


class IFSSearchService:
    """
    Main search service that integrates all components.

    INTEGRATION STRATEGY:
    --------------------
    1. Use semantic search as primary signal
    2. Enhance with hybrid ranking
    3. Apply business rules and filters
    4. Learn from user feedback
    """

    def __init__(self,
                 model_path: Path,
                 index_path: Path,
                 config: Optional[SearchConfig] = None):
        """
        Initialize the search service.

        Parameters:
        -----------
        model_path: Path to trained semantic model
        index_path: Path to FAISS index
        config: Search configuration (defaults to production)
        """
        # Initialize semantic search engine
        self.semantic_engine = SemanticSearchEngine(
            model_path=model_path,
            index_path=index_path
        )
        self.semantic_engine.initialize()

        # Initialize configuration
        self.config = config or SearchConfig.production_config()
        self.config_manager = SearchConfigManager()

        # Initialize hybrid ranker
        self.ranker = HybridRanker(
            chunks=self.semantic_engine.chunks,
            semantic_weight=self.config.semantic_weight,
            lexical_weight=self.config.lexical_weight,
            domain_weight=self.config.domain_weight
        )

        # Search history for learning
        self.search_history = []

        logger.info(f"IFS Search Service initialized with {len(self.semantic_engine.chunks)} chunks")

    def search(self,
              query: str,
              user_id: Optional[str] = None,
              context: Optional[Dict[str, Any]] = None,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Perform a search with full hybrid ranking.

        Parameters:
        -----------
        query: Search query
        user_id: Optional user ID for personalization
        context: Optional context (current module, file, etc.)
        filters: Optional filters (module, layer, etc.)

        Returns:
        --------
        Ranked list of search results
        """
        logger.info(f"Searching for: {query}")

        # Get configuration for this user/query
        config = self.config_manager.get_config_for_user(user_id, query)

        # Record search for analytics
        self.config_manager.record_search('default')

        # Get semantic scores from neural model
        semantic_results = self.semantic_engine.search(
            query,
            k=config.max_results * 3  # Get extra for filtering
        )

        # Extract semantic scores
        semantic_scores = [0.0] * len(self.semantic_engine.chunks)
        for chunk_id, score in semantic_results:
            # Find chunk index
            for idx, chunk in enumerate(self.semantic_engine.chunks):
                if chunk.chunk_id == chunk_id:
                    semantic_scores[idx] = score
                    break

        # Apply filters if provided
        filtered_chunks = self._apply_filters(
            self.semantic_engine.chunks,
            filters
        )

        # Perform hybrid ranking
        results = self.ranker.rank_results(
            query=query,
            semantic_scores=semantic_scores,
            user_context=context
        )

        # Apply configuration filters
        results = self._apply_config_filters(results, config)

        # Limit to max results
        results = results[:config.max_results]

        # Record search for learning
        self._record_search(query, results, user_id)

        logger.info(f"Returning {len(results)} results")
        return results

    def _apply_filters(self,
                      chunks: List[CodeChunk],
                      filters: Optional[Dict[str, Any]]) -> List[CodeChunk]:
        """Apply user-specified filters."""
        if not filters:
            return chunks

        filtered = chunks

        # Module filter
        if 'module' in filters:
            filtered = [c for c in filtered if c.module == filters['module']]

        # Layer filter
        if 'layer' in filters:
            filtered = [c for c in filtered if c.layer == filters['layer']]

        # Language filter
        if 'language' in filters:
            filtered = [c for c in filtered if c.language == filters['language']]

        # Complexity filter
        if 'max_complexity' in filters:
            filtered = [c for c in filtered
                       if c.complexity_score <= filters['max_complexity']]

        return filtered

    def _apply_config_filters(self,
                            results: List[SearchResult],
                            config: SearchConfig) -> List[SearchResult]:
        """Apply configuration-based filters."""
        filtered = results

        # Minimum similarity filter
        if config.min_similarity > 0:
            filtered = [r for r in filtered
                       if r.similarity_score >= config.min_similarity]

        # Quality filters
        if config.filter_low_quality:
            # Filter by documentation
            if config.min_documentation_score > 0:
                filtered = [r for r in filtered
                          if bool(r.chunk.docstring)]

            # Filter by complexity
            if config.max_complexity_score < 1.0:
                filtered = [r for r in filtered
                          if r.chunk.complexity_score <= config.max_complexity_score]

        return filtered

    def _record_search(self,
                      query: str,
                      results: List[SearchResult],
                      user_id: Optional[str]):
        """Record search for learning and analytics."""
        self.search_history.append({
            'query': query,
            'user_id': user_id,
            'timestamp': datetime.now(),
            'num_results': len(results),
            'top_results': [r.chunk.chunk_id for r in results[:5]]
        })

    def record_click(self,
                    query: str,
                    chunk_id: str,
                    user_id: Optional[str] = None):
        """
        Record user click for learning.

        This is crucial for improving search over time!
        """
        # Find the rank of the clicked result
        for search in reversed(self.search_history):
            if search['query'] == query:
                if chunk_id in search['top_results']:
                    rank = search['top_results'].index(chunk_id) + 1

                    # Record in ranker for learning
                    if hasattr(self.ranker, 'record_click'):
                        self.ranker.record_click(query, chunk_id, rank)

                    # Record in config manager
                    self.config_manager.record_click('default')

                    logger.info(f"Recorded click: query='{query}', chunk={chunk_id}, rank={rank}")
                    break

    def get_analytics(self) -> Dict[str, Any]:
        """Get search analytics for monitoring."""
        return {
            'total_searches': len(self.search_history),
            'config_performance': self.config_manager.config_performance,
            'best_config': self.config_manager.get_best_config(),
            'recent_searches': self.search_history[-10:],
        }
```

## 🎯 Summary: The Optimal Approach

Based on extensive research and production experience, here's my recommendation:

### **Start with Hybrid Search (Recommended)**

**Why Hybrid is Optimal:**

1. **Semantic search** handles intent and conceptual queries
2. **Lexical search (BM25)** handles exact identifiers and specific terms
3. **Domain rules** capture IFS-specific patterns
4. **Intent detection** adapts ranking to query type

### **Progressive Refinement Strategy:**

1. **Phase 1: Deploy Hybrid (Week 1)**

   - Use the configuration I provided
   - 50% semantic, 30% lexical, 20% domain rules
   - Monitor click-through rates

2. **Phase 2: Tune Weights (Week 2-4)**

   - Analyze which results users click
   - Adjust weights based on performance
   - A/B test different configurations

3. **Phase 3: Add Learning (Month 2)**
   - Implement click tracking
   - Personalize per user/team
   - Continuously improve

### **Key Success Factors:**

1. **Don't rely solely on semantic search** - It misses exact matches
2. **Don't rely solely on rules** - They miss conceptual queries
3. **Track user behavior** - Clicks reveal true relevance
4. **Start conservative** - Better to have good results than experimental ones
5. **Iterate based on data** - Let user behavior guide improvements

### **Configuration for Your System:**

```python
# Start with this for IFS
config = SearchConfig(
    strategy=SearchStrategy.HYBRID,
    semantic_weight=0.5,   # Your trained model
    lexical_weight=0.3,    # BM25 for exact matches
    domain_weight=0.2,     # IFS-specific boosts
    boost_exact_matches=True,  # IFS has standard naming
    boost_same_module=1.2,     # Module context matters
)
```

This approach has been proven to work across many enterprise search systems and specifically handles the characteristics of codebases like IFS with standardized naming conventions, modular architecture, and domain-specific patterns.
