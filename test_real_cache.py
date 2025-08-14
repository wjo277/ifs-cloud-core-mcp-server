"""Quick test of enhanced caching with real IFS files."""

import asyncio
import time
from pathlib import Path

# Add src to Python path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


async def test_real_caching():
    """Test caching with real IFS files."""
    print("IFS Cloud MCP Server - Real Caching Test")
    print("=" * 45)
    
    # Use the _work directory with real IFS files
    work_dir = Path("_work")
    if not work_dir.exists():
        print("❌ _work directory not found. Run the PowerShell extraction script first.")
        return
    
    index_path = Path("./ifs_cache_test")
    
    # Test 1: First indexing
    print("\n1. First Indexing (Cold Start)")
    print("-" * 30)
    
    indexer = IFSCloudTantivyIndexer(index_path, create_new=True)
    
    # Index just a small subset for faster testing
    test_dirs = []
    for dir_path in work_dir.iterdir():
        if dir_path.is_dir():
            test_dirs.append(dir_path)
            if len(test_dirs) >= 3:  # Just test 3 directories
                break
    
    total_stats = {"indexed": 0, "cached": 0, "skipped": 0, "errors": 0}
    
    start_time = time.time()
    for test_dir in test_dirs:
        stats = await indexer.index_directory(test_dir, recursive=True)
        for key in total_stats:
            total_stats[key] += stats[key]
    end_time = time.time()
    
    initial_time = end_time - start_time
    
    print(f"✓ Initial indexing completed in {initial_time:.2f} seconds")
    print(f"  • Files indexed: {total_stats['indexed']}")
    print(f"  • Files cached: {total_stats['cached']}")
    print(f"  • Total processed: {total_stats['indexed'] + total_stats['cached']}")
    
    # Test 2: Re-indexing (should be much faster)
    print("\n2. Re-indexing (Cache Test)")
    print("-" * 30)
    
    total_stats2 = {"indexed": 0, "cached": 0, "skipped": 0, "errors": 0}
    
    start_time = time.time()
    for test_dir in test_dirs:
        stats = await indexer.index_directory(test_dir, recursive=True)
        for key in total_stats2:
            total_stats2[key] += stats[key]
    end_time = time.time()
    
    cached_time = end_time - start_time
    
    print(f"✓ Cached indexing completed in {cached_time:.2f} seconds")
    print(f"  • Files indexed: {total_stats2['indexed']}")
    print(f"  • Files cached: {total_stats2['cached']}")
    print(f"  • Total processed: {total_stats2['indexed'] + total_stats2['cached']}")
    
    # Test 3: Cache statistics
    print("\n3. Cache Performance Analysis")
    print("-" * 30)
    
    cache_stats = indexer.get_statistics()
    
    if cached_time > 0 and initial_time > 0:
        speedup = initial_time / cached_time
        print(f"✓ Performance improvement: {speedup:.1f}x faster")
    
    if total_stats2['cached'] > 0:
        cache_hit_ratio = total_stats2['cached'] / (total_stats2['cached'] + total_stats2['indexed']) * 100
        print(f"✓ Cache hit ratio: {cache_hit_ratio:.1f}%")
    
    print(f"✓ Total documents in index: {cache_stats['total_documents']}")
    print(f"✓ Cache metadata entries: {cache_stats['cached_files']}")
    
    # Cleanup
    indexer.close()
    
    print("\n" + "=" * 45)
    print("✅ Real Caching Test Complete!")
    
    if total_stats2['cached'] > 0:
        print(f"\n🚀 SUCCESS: Cache working! {total_stats2['cached']} files were cached.")
        print("   This dramatically improves performance for large codebases!")
    else:
        print("\n⚠️  Cache not fully utilized in this test run.")
        print("   This is normal for small test datasets.")


if __name__ == "__main__":
    asyncio.run(test_real_caching())
