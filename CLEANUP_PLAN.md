# Cleanup Plan for Obsolete Files

## Files to Keep (Current System)
- `simple_sequential_embedding.py` - Current optimized system
- `src/` directory - Core framework
- `pyproject.toml`, `uv.lock` - Dependencies
- `README.md`, `QUICK_START.md` - Documentation
- `cache/` - Current NDJSON cache system
- `_work/` - Source data
- `.vscode/`, `.git/`, `.venv/` - Development environment

## Files to Clean Up

### Old Test Files (Obsolete)
- `test_ai_enhancement.py`
- `test_ai_flow.py` 
- `test_full_plsql_file.py`
- `test_ollama.py`
- `test_ollama_connection.py`
- `test_ollama_parallel.py`
- `test_ollama_parallel_restart.py`
- `test_ollama_vram_usage.py`
- `test_parser_approach.py`
- `test_plsql_chunking.py`
- `test_plsql_detailed.py`
- `test_production_no_ai.py`
- `test_production_quick.py`
- `test_real_plsql.py`
- `test_small_batch.py`
- `quick_ollama_test.py`
- `simple_vram_test.py`

### Old Demo Files (Obsolete)
- `demo_ai_enhanced_search.py`
- `demo_comprehensive_framework.py`
- `demo_production_safe_system.py`
- `demo_unixcoder_search.py`

### Old Embedding Systems (Superseded)
- `batch_production_embedding.py` - Superseded by simple_sequential
- `optimized_pipeline_embedding.py` - User rejected this approach
- `embedding_cli.py`
- `embedding_monitor.py`
- `train_semantic_search.py`
- `TRAINING_SEMANTIC_SEARCH.py`

### Analysis/Debug Scripts (No longer needed)
- `analyze_missing.py`
- `debug_ai_response.py`
- `direct_check.py`
- `find_exact_missing.py`
- `find_missing.py`
- `verify_nested.py`

### Old Cache/Embedding Directories
- `batch_production_embeddings/`
- `demo_production_cache/`
- `demo_production_embeddings/`
- `pipeline_production_embeddings/`
- `production_cache/`
- `simple_production_embeddings/`
- `test_ai_cache/`
- `test_ai_embeddings/`
- `test_batch_embeddings/`
- `test_production_cache/`
- `test_production_cache_no_ai/`
- `test_production_embeddings/`
- `test_production_no_ai/`

### Old Documentation (Outdated)
- `COMPREHENSIVE_EMBEDDING_FRAMEWORK.md`
- `FASTAI_SEMANTIC_SEARCH.md`
- `FEASABILITY_ANALYSIS.md`
- `MODEL_QUANTIZATION_RESULTS.md`
- `PRODUCTION_SAFE_IMPLEMENTATION_SUMMARY.md`
- `PRODUCTION_SAFE_SEMANTIC_SEARCH.md`
- `QUANTIZED_MODEL_DEFAULT_IMPLEMENTATION.md`
- `SEMANTIC_SEARCH.md`
- `SEMANTIC_SEARCH_EXPLAINED.md`
- `SEMANTIC_SEARCH_IMPLEMENTATION.md`
- `The search ranking is not that great, it.md`

### Config Files (No longer used)
- `embedding_config.toml`
- `manual_functions.txt`

### Miscellaneous
- `__pycache__/` - Python cache (safe to delete)

## Action Plan
1. Create `archive/` directory
2. Move obsolete files to archive
3. Clean up empty directories
4. Update .gitignore if needed
