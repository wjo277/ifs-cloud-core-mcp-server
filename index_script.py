#!/usr/bin/env python3
"""Simple script to index a directory from VS Code tasks."""

import sys
import asyncio
import logging
from pathlib import Path
from src.ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer

# Set up logging to see progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def main():
    if len(sys.argv) < 2:
        print("Usage: python index_script.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    print(f"🔍 Indexing directory: {directory_path}")
    
    # Check if directory exists
    if not Path(directory_path).exists():
        print(f"❌ Directory not found: {directory_path}")
        sys.exit(1)
    
    try:
        # Create a fresh index to avoid corruption issues
        indexer = IFSCloudTantivyIndexer("index", create_new=True)
        print("📝 Created fresh index to avoid corruption issues")
        
        # Count files first to show progress
        supported_extensions = indexer.SUPPORTED_EXTENSIONS
        pattern = "**/*"
        files = []
        for extension in supported_extensions:
            files.extend(Path(directory_path).glob(f"{pattern}{extension}"))
        
        print(f"📁 Found {len(files)} supported files to index")
        
        stats = await indexer.index_directory(directory_path, force_reindex=True)
        
        print(f"✅ Indexing complete!")
        print(f"📊 Final Stats:")
        print(f"   � Indexed: {stats['indexed']}")
        print(f"   💾 Cached: {stats['cached']}")
        print(f"   ⏭️  Skipped: {stats['skipped']}")
        print(f"   ❌ Errors: {stats['errors']}")
        
        if stats['errors'] > 0:
            print(f"⚠️  Some files had errors during indexing. Check the logs above for details.")
        
        # Clean up the indexer properly
        indexer.cleanup()
        print("🧹 Cleaned up indexer resources")
        
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
