#!/usr/bin/env python3
"""
AI Summarization Development Tool
=================================

This script demonstrates and tests the AI summarization feature for improving
semantic search quality. It can be used during development to:

1. Test AI summarization on sample code chunks
2. Generate summaries for existing chunks in the index
3. Compare semantic search results before/after AI enhancement
4. Benchmark the improvement in search quality

Usage:
------
# Install dev dependencies first
uv sync --dev

# Install and start Ollama (if not already done)
# Download from: https://ollama.ai
ollama pull qwen2.5:8b

# Run the summarization test
python -m src.ifs_cloud_mcp_server.semantic_search.test_ai_summarization
"""

import asyncio
import logging
from pathlib import Path
from typing import List
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ifs_cloud_mcp_server.semantic_search.ai_summarizer import (
    AISummarizer,
    enrich_chunks_with_ai_summaries,
)
from src.ifs_cloud_mcp_server.semantic_search.data_structures import CodeChunk
from src.ifs_cloud_mcp_server.semantic_search.data_loader import IFSDataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_chunk() -> CodeChunk:
    """Create a sample PL/SQL chunk for testing"""
    sample_code = """
    PROCEDURE Validate_Customer_Order___ (
       newrec_ IN OUT customer_order_tab%ROWTYPE
    ) IS
    BEGIN
       IF newrec_.order_date > SYSDATE THEN
          Error_SYS.Record_General(lu_name_, 'FUTUREORDER: Order date cannot be in the future');
       END IF;
       
       IF Customer_Info_API.Get_Customer_Category(newrec_.customer_no) = 'PROSPECT' THEN
          Error_SYS.Record_General(lu_name_, 'PROSPECTORDER: Cannot create order for prospect customer');
       END IF;
       
       -- Validate credit limit
       Customer_Order_API.Check_Credit_Limit__(newrec_.customer_no, newrec_.order_value);
    END Validate_Customer_Order___;
    """

    chunk = CodeChunk(
        chunk_id="test_validate_customer_order",
        file_path="test/CustomerOrder.plsql",
        start_line=100,
        end_line=115,
        raw_content=sample_code,
        processed_content=sample_code.strip(),
        chunk_type="plsql_procedure",
        function_name="Validate_Customer_Order___",
        module="ORDER",
        layer="business",
        language="plsql",
        business_terms=["customer", "order", "validation"],
        api_calls=[
            "Error_SYS.Record_General",
            "Customer_Info_API.Get_Customer_Category",
            "Customer_Order_API.Check_Credit_Limit__",
        ],
        database_tables=["customer_order_tab"],
        has_error_handling=True,
        complexity_score=0.3,
    )

    return chunk


async def test_ai_summarization():
    """Test the AI summarization feature"""

    logger.info("🧪 Testing AI Summarization Feature")
    logger.info("=" * 50)

    # Create test chunk
    chunk = create_sample_chunk()

    logger.info(f"📝 Original chunk:")
    logger.info(f"   Function: {chunk.function_name}")
    logger.info(f"   Module: {chunk.module}")
    logger.info(f"   Type: {chunk.chunk_type}")
    logger.info(f"   Lines: {chunk.start_line}-{chunk.end_line}")

    # Initialize AI summarizer
    cache_dir = Path("cache/test_summaries")
    summarizer = AISummarizer(cache_dir=cache_dir)

    # Test summarization
    logger.info("\n🤖 Generating AI summary...")

    summary, was_cached = await summarizer.summarize_chunk(chunk)

    logger.info("✅ AI Summary Generated:")
    logger.info(f"   Summary: {summary.get('summary', 'N/A')}")
    logger.info(f"   Purpose: {summary.get('purpose', 'N/A')}")
    logger.info(f"   Keywords: {', '.join(summary.get('keywords', []))}")
    logger.info(f"   Method: {summary.get('method', 'ai')}")
    logger.info(f"   From cache: {was_cached}")

    # Show before/after processed content
    logger.info("\n📊 Content Enhancement:")
    logger.info("BEFORE (original processed content):")
    logger.info(chunk.raw_content[:200] + "...")

    # Enrich the chunk
    enriched_chunk = summarizer.enrich_chunk_with_summary(chunk, summary)

    logger.info("\nAFTER (enhanced with AI summary):")
    logger.info(enriched_chunk.processed_content[:400] + "...")

    return enriched_chunk


async def test_batch_summarization():
    """Test batch summarization with multiple chunks"""

    logger.info("\n🔄 Testing Batch Summarization")
    logger.info("=" * 50)

    # Create multiple test chunks
    chunks = []

    # Chunk 1: Order validation
    chunk1 = create_sample_chunk()

    # Chunk 2: Price calculation
    chunk2 = CodeChunk(
        chunk_id="test_calculate_order_total",
        file_path="test/CustomerOrder.plsql",
        start_line=200,
        end_line=220,
        raw_content="""
        FUNCTION Calculate_Order_Total___ (
           order_no_ IN VARCHAR2
        ) RETURN NUMBER IS
           total_amount_ NUMBER := 0;
        BEGIN
           SELECT SUM(sale_unit_price * buy_qty_due)
           INTO total_amount_
           FROM customer_order_line_tab
           WHERE order_no = order_no_;
           
           RETURN NVL(total_amount_, 0);
        END Calculate_Order_Total___;
        """,
        processed_content="FUNCTION Calculate_Order_Total___ calculating order totals...",
        chunk_type="plsql_function",
        function_name="Calculate_Order_Total___",
        module="ORDER",
        layer="business",
        language="plsql",
        business_terms=["order", "price", "calculation"],
        database_tables=["customer_order_line_tab"],
        sql_queries=[
            "SELECT SUM(sale_unit_price * buy_qty_due) FROM customer_order_line_tab"
        ],
        complexity_score=0.2,
    )

    chunks = [chunk1, chunk2]

    # Test batch processing
    logger.info(f"📦 Processing {len(chunks)} chunks in batch...")

    enriched_chunks = await enrich_chunks_with_ai_summaries(chunks, batch_size=2)

    logger.info(f"✅ Batch processing complete!")

    for i, chunk in enumerate(enriched_chunks):
        logger.info(f"\n📝 Chunk {i+1}: {chunk.function_name}")
        if hasattr(chunk, "ai_summary") and chunk.ai_summary:
            logger.info(f"   Summary: {chunk.ai_summary.get('summary', 'N/A')}")
            logger.info(
                f"   Keywords: {', '.join(chunk.ai_summary.get('keywords', []))}"
            )
        else:
            logger.info("   No AI summary available")


async def benchmark_search_improvement():
    """
    Benchmark search quality improvement with AI summaries
    """
    logger.info("\n🏁 Benchmarking Search Quality Improvement")
    logger.info("=" * 50)

    # This would be implemented with actual search engine integration
    # For now, just demonstrate the concept

    test_queries = [
        "validate customer order",
        "calculate order total",
        "check credit limit",
        "order date validation",
        "customer prospect check",
    ]

    logger.info("🔍 Test queries:")
    for query in test_queries:
        logger.info(f"   • {query}")

    logger.info("\n💡 With AI summaries, these queries should match much better!")
    logger.info("   • Natural language descriptions improve semantic matching")
    logger.info("   • Business keywords enhance discoverability")
    logger.info("   • Purpose statements provide context for relevance")


async def main():
    """Run all AI summarization tests"""

    print("🚀 AI Summarization Development Tool")
    print("====================================")
    print()

    try:
        # Test individual summarization
        await test_ai_summarization()

        # Test batch processing
        await test_batch_summarization()

        # Benchmark concept
        await benchmark_search_improvement()

        print("\n🎉 All tests completed successfully!")
        print()
        print("💡 Next steps:")
        print("   1. Integrate with existing data loader")
        print("   2. Re-index chunks with AI summaries")
        print("   3. Test improved search quality")
        print("   4. Measure performance impact")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
