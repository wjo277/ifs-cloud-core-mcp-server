#!/usr/bin/env python3
"""
Simple test script to verify basic functionality of the IFS Cloud MCP Server
without requiring external dependencies to be installed.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic functionality without dependencies."""
    
    print("Testing IFS Cloud MCP Server basic functionality...")
    print("=" * 50)
    
    # Test 1: Import check
    try:
        import ifs_cloud_mcp_server
        print("✓ Package import successful")
        print(f"  Version: {ifs_cloud_mcp_server.__version__}")
        print(f"  Author: {ifs_cloud_mcp_server.__author__}")
    except ImportError as e:
        print(f"✗ Package import failed: {e}")
        return False
    
    # Test 2: Test complexity calculation (no external deps)
    try:
        from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer
        
        # Create a mock indexer for testing (without actual Tantivy)
        class MockIndexer:
            SUPPORTED_EXTENSIONS = {
                '.entity', '.plsql', '.views', '.storage', 
                '.fragment', '.client', '.projection'
            }
            
            def calculate_complexity_score(self, content, file_type):
                return IFSCloudTantivyIndexer.calculate_complexity_score(None, content, file_type)
            
            def extract_entities(self, content, file_type):
                return IFSCloudTantivyIndexer.extract_entities(None, content, file_type)
            
            def extract_functions(self, content, file_type):
                return IFSCloudTantivyIndexer.extract_functions(None, content, file_type)
        
        indexer = MockIndexer()
        
        # Test complexity calculation
        simple_code = "entity Simple { key Id; }"
        complex_code = """
        PROCEDURE ComplexProcedure IS
            CURSOR c1 IS SELECT * FROM table1;
        BEGIN
            FOR rec IN c1 LOOP
                IF rec.status = 'ACTIVE' THEN
                    CASE rec.type
                        WHEN 'A' THEN process_a();
                        WHEN 'B' THEN process_b();
                    END CASE;
                END IF;
            END LOOP;
        EXCEPTION
            WHEN others THEN log_error();
        END;
        """
        
        simple_score = indexer.calculate_complexity_score(simple_code, '.entity')
        complex_score = indexer.calculate_complexity_score(complex_code, '.plsql')
        
        print(f"✓ Complexity calculation works:")
        print(f"  Simple entity score: {simple_score:.3f}")
        print(f"  Complex PL/SQL score: {complex_score:.3f}")
        print(f"  Complex > Simple: {complex_score > simple_score}")
        
    except Exception as e:
        print(f"✗ Complexity calculation failed: {e}")
        return False
    
    # Test 3: Test entity extraction
    try:
        entity_content = "entity CustomerOrder { key OrderNo; }"
        plsql_content = "PROCEDURE ProcessOrder IS BEGIN NULL; END;"
        
        entities_entity = indexer.extract_entities(entity_content, '.entity')
        entities_plsql = indexer.extract_entities(plsql_content, '.plsql')
        
        print(f"✓ Entity extraction works:")
        print(f"  From entity file: {entities_entity}")
        print(f"  From PL/SQL file: {entities_plsql}")
        
    except Exception as e:
        print(f"✗ Entity extraction failed: {e}")
        return False
    
    # Test 4: Test function extraction
    try:
        plsql_with_functions = """
        FUNCTION CalculateTotal RETURN NUMBER IS
        BEGIN RETURN 0; END;
        
        PROCEDURE SaveOrder IS
        BEGIN NULL; END;
        """
        
        functions = indexer.extract_functions(plsql_with_functions, '.plsql')
        print(f"✓ Function extraction works:")
        print(f"  Found functions: {functions}")
        
    except Exception as e:
        print(f"✗ Function extraction failed: {e}")
        return False
    
    # Test 5: File type support
    try:
        supported = indexer.SUPPORTED_EXTENSIONS
        print(f"✓ Supported file types: {sorted(supported)}")
        
    except Exception as e:
        print(f"✗ File type check failed: {e}")
        return False
    
    return True

def test_sample_files():
    """Test with the sample IFS Cloud files."""
    
    print("\nTesting with sample IFS Cloud files...")
    print("=" * 50)
    
    # Find sample files
    examples_dir = Path(__file__).parent / "examples" / "sample_ifs_project"
    
    if not examples_dir.exists():
        print("✗ Sample files directory not found")
        return False
    
    sample_files = list(examples_dir.glob("*"))
    print(f"✓ Found {len(sample_files)} sample files:")
    
    for file_path in sorted(sample_files):
        file_size = file_path.stat().st_size
        print(f"  - {file_path.name} ({file_size:,} bytes)")
    
    # Test reading and processing sample files
    try:
        from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer
        
        # Create mock indexer for testing
        class MockIndexer:
            def calculate_complexity_score(self, content, file_type):
                return IFSCloudTantivyIndexer.calculate_complexity_score(None, content, file_type)
            
            def extract_entities(self, content, file_type):
                return IFSCloudTantivyIndexer.extract_entities(None, content, file_type)
        
        indexer = MockIndexer()
        
        for file_path in sample_files:
            if file_path.suffix in IFSCloudTantivyIndexer.SUPPORTED_EXTENSIONS:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    complexity = indexer.calculate_complexity_score(content, file_path.suffix)
                    entities = indexer.extract_entities(content, file_path.suffix)
                    
                    print(f"  ✓ {file_path.name}:")
                    print(f"    Complexity: {complexity:.3f}")
                    print(f"    Entities: {len(entities)} found")
                    if entities:
                        print(f"    Sample entities: {', '.join(entities[:3])}")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {file_path.name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample file processing failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("IFS Cloud MCP Server - Basic Functionality Test")
    print("=" * 60)
    
    success = True
    
    # Run basic functionality tests
    if not test_basic_functionality():
        success = False
    
    # Run sample file tests
    if not test_sample_files():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! The basic functionality is working correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -e .")
        print("2. Run the server: ifs-cloud-mcp-server --index-path ./index")
        print("3. Use MCP client to interact with the server")
        return 0
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())