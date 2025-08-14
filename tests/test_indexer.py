"""Tests for the IFS Cloud MCP Server."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from ifs_cloud_mcp_server.indexer import IFSCloudTantivyIndexer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_files(temp_dir):
    """Create sample IFS Cloud files for testing."""
    files = {}
    
    # Create sample entity file
    entity_content = """entity CustomerOrder {
    key Number;
    attribute CustomerNo varchar(20);
    attribute Description varchar(100);
    
    procedure CreateOrder();
    function GetOrderTotal() returns decimal;
}"""
    entity_file = temp_dir / "CustomerOrder.entity"
    entity_file.write_text(entity_content)
    files["entity"] = entity_file
    
    # Create sample PL/SQL file
    plsql_content = """PACKAGE CustomerOrderAPI IS
    PROCEDURE CreateOrder(
        customer_no_ IN VARCHAR2,
        description_ IN VARCHAR2
    );
    
    FUNCTION GetOrderTotal(
        order_no_ IN NUMBER
    ) RETURN NUMBER;
    
    CURSOR order_cursor IS
        SELECT * FROM customer_order
        WHERE status = 'OPEN';
END CustomerOrderAPI;"""
    plsql_file = temp_dir / "CustomerOrderAPI.plsql"
    plsql_file.write_text(plsql_content)
    files["plsql"] = plsql_file
    
    # Create sample view file
    view_content = """VIEW CustomerOrderView AS
SELECT 
    co.order_no,
    co.customer_no,
    c.customer_name,
    co.order_date,
    co.total_amount
FROM customer_order co
JOIN customer c ON co.customer_no = c.customer_no
WHERE co.status = 'CONFIRMED';"""
    view_file = temp_dir / "CustomerOrderView.views"
    view_file.write_text(view_content)
    files["view"] = view_file
    
    return files


@pytest.fixture
def indexer(temp_dir):
    """Create a test indexer."""
    index_path = temp_dir / "test_index"
    return IFSCloudTantivyIndexer(index_path, create_new=True)


class TestIFSCloudTantivyIndexer:
    """Test the Tantivy indexer."""
    
    def test_indexer_initialization(self, indexer):
        """Test indexer initialization."""
        assert indexer is not None
        assert indexer.index_path.exists()
        assert len(indexer.SUPPORTED_EXTENSIONS) == 7
    
    def test_complexity_calculation(self, indexer):
        """Test complexity score calculation."""
        simple_content = "entity Simple { key Id; }"
        complex_content = """
        PROCEDURE ComplexProcedure IS
            CURSOR c1 IS SELECT * FROM table1;
            CURSOR c2 IS SELECT * FROM table2;
        BEGIN
            FOR rec IN c1 LOOP
                IF rec.status = 'ACTIVE' THEN
                    WHILE rec.counter < 100 LOOP
                        CASE rec.type
                            WHEN 'A' THEN process_a();
                            WHEN 'B' THEN process_b();
                            ELSE process_default();
                        END CASE;
                    END LOOP;
                END IF;
            END LOOP;
        EXCEPTION
            WHEN others THEN
                log_error();
        END;
        """
        
        simple_score = indexer.calculate_complexity_score(simple_content, '.entity')
        complex_score = indexer.calculate_complexity_score(complex_content, '.plsql')
        
        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
        assert complex_score > simple_score
    
    def test_entity_extraction(self, indexer):
        """Test entity extraction from different file types."""
        # Test entity file
        entity_content = "entity CustomerOrder { key Number; }"
        entities = indexer.extract_entities(entity_content, '.entity')
        assert 'CustomerOrder' in entities
        
        # Test PL/SQL file
        plsql_content = "PROCEDURE ProcessOrder IS BEGIN NULL; END;"
        entities = indexer.extract_entities(plsql_content, '.plsql')
        assert 'PROCESSORDER' in entities
    
    def test_function_extraction(self, indexer):
        """Test function/procedure extraction."""
        plsql_content = """
        FUNCTION CalculateTotal RETURN NUMBER IS
        BEGIN
            RETURN 0;
        END;
        
        PROCEDURE SaveOrder IS
        BEGIN
            NULL;
        END;
        """
        functions = indexer.extract_functions(plsql_content, '.plsql')
        assert 'CALCULATETOTAL' in functions
        assert 'SAVEORDER' in functions
    
    @pytest.mark.asyncio
    async def test_file_indexing(self, indexer, sample_files):
        """Test indexing individual files."""
        # Test entity file indexing
        success = await indexer.index_file(sample_files["entity"])
        assert success
        
        # Test PL/SQL file indexing
        success = await indexer.index_file(sample_files["plsql"])
        assert success
        
        # Test view file indexing
        success = await indexer.index_file(sample_files["view"])
        assert success
        
        # Check statistics
        stats = indexer.get_statistics()
        assert stats["total_documents"] == 3
    
    @pytest.mark.asyncio
    async def test_directory_indexing(self, indexer, sample_files):
        """Test indexing a directory."""
        # Index the directory containing sample files
        directory = sample_files["entity"].parent
        stats = await indexer.index_directory(directory)
        
        assert stats["indexed"] == 3
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
    
    def test_search_functionality(self, indexer, sample_files):
        """Test search functionality."""
        # Need to commit after indexing for search to work
        indexer._writer.commit()
        
        # Test basic search
        results = indexer.search("CustomerOrder")
        assert len(results) > 0
        
        # Test type filtering
        results = indexer.search("*", file_type=".entity")
        entity_results = [r for r in results if r.type == ".entity"]
        assert len(entity_results) > 0
    
    def test_similarity_search(self, indexer, sample_files):
        """Test similarity search."""
        # Index files first
        indexer._writer.commit()
        
        # Find similar files to the entity file
        similar = indexer.find_similar_files(sample_files["entity"])
        # Should find files with similar entities/content
        assert isinstance(similar, list)
    
    def test_statistics(self, indexer):
        """Test getting index statistics."""
        stats = indexer.get_statistics()
        
        assert "total_documents" in stats
        assert "index_size" in stats
        assert "index_path" in stats
        assert "supported_extensions" in stats
        assert isinstance(stats["supported_extensions"], list)


class TestComplexityScoring:
    """Test complexity scoring for different file types."""
    
    def test_plsql_complexity(self, indexer):
        """Test PL/SQL complexity scoring."""
        simple_plsql = "PROCEDURE Simple IS BEGIN NULL; END;"
        complex_plsql = """
        PROCEDURE Complex IS
            CURSOR c1 IS SELECT * FROM orders;
            CURSOR c2 IS SELECT * FROM customers;
        BEGIN
            FOR order_rec IN c1 LOOP
                IF order_rec.status = 'PENDING' THEN
                    FOR customer_rec IN c2 LOOP
                        WHILE customer_rec.balance > 0 LOOP
                            CASE order_rec.priority
                                WHEN 1 THEN high_priority_process();
                                WHEN 2 THEN medium_priority_process();
                                ELSE low_priority_process();
                            END CASE;
                        END LOOP;
                    END LOOP;
                END IF;
            END LOOP;
        EXCEPTION
            WHEN NO_DATA_FOUND THEN
                handle_no_data();
            WHEN OTHERS THEN
                handle_error();
        END;
        """
        
        simple_score = indexer.calculate_complexity_score(simple_plsql, '.plsql')
        complex_score = indexer.calculate_complexity_score(complex_plsql, '.plsql')
        
        assert complex_score > simple_score
        assert complex_score > 0.5  # Should be high complexity
    
    def test_entity_complexity(self, indexer):
        """Test entity complexity scoring."""
        simple_entity = "entity Simple { key Id; }"
        complex_entity = """
        entity ComplexOrder {
            key OrderNo;
            attribute CustomerNo varchar(20);
            attribute OrderDate date;
            attribute Status varchar(10);
            
            procedure ValidateOrder();
            procedure ProcessPayment();
            function CalculateTotal() returns decimal;
            function GetDiscountAmount() returns decimal;
            
            reference CustomerRef(CustomerNo) to Customer(CustomerNo);
            reference ProductRef(ProductNo) to Product(ProductNo);
        }
        """
        
        simple_score = indexer.calculate_complexity_score(simple_entity, '.entity')
        complex_score = indexer.calculate_complexity_score(complex_entity, '.entity')
        
        assert complex_score > simple_score