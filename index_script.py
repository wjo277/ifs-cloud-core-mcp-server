#!/usr/bin/env python3
"""Simple script to index a directory from VS Code tasks."""

import sys
import asyncio
from pathlib import Path
from src.ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


async def main():
    if len(sys.argv) < 2:
        print("Usage: python index_script.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    print(f"🔍 Indexing directory: {directory_path}")
    
    try:
        indexer = IFSCloudTantivyIndexer("index", create_new=False)
        stats = await indexer.index_directory(directory_path)
        
        print(f"✅ Indexing complete!")
        print(f"📊 Stats: {stats['indexed']} indexed, {stats['cached']} cached, {stats['skipped']} skipped, {stats['errors']} errors")
        
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
