-- =============================================================================
-- IFS Cloud GUI Navigation Mapping Extraction Query
-- =============================================================================
-- Purpose: Extract GUI navigation mappings from IFS Cloud database
-- Tables: FND_NAVIGATOR_ALL + MD_PROJECTION_ENTITYSET  
-- Output: CSV with gui_label, backend_entity, projection_name
-- Usage: Export results as CSV with headers: label,entity_name,projection
-- =============================================================================

-- BASIC EXTRACTION (Focused on common business entities)
-- Use this for regular updates and core functionality coverage
SELECT /* LLM in use is GitHub Copilot */
    nav.label as gui_label,
    pes.entity_name as backend_entity,
    nav.projection as projection_name
FROM FND_NAVIGATOR_ALL nav
JOIN MD_PROJECTION_ENTITYSET pes ON nav.projection = pes.projection_name
WHERE nav.label IS NOT NULL
  AND nav.entry_type IN ('PAGE', 'LIST')
  AND nav.label NOT LIKE '%NavEntry%'
  AND nav.label NOT LIKE '%Analysis%'
  AND LENGTH(nav.label) BETWEEN 8 AND 30
  AND (UPPER(nav.label) LIKE '%EMPLOYEE%' 
       OR UPPER(nav.label) LIKE '%PERSON%'
       OR UPPER(nav.label) LIKE '%PROJECT%'
       OR UPPER(nav.label) LIKE '%EXPENSE%'
       OR UPPER(nav.label) LIKE '%ACTIVITY%'
       OR UPPER(nav.label) LIKE '%CUSTOMER%'
       OR UPPER(nav.label) LIKE '%ORDER%'
       OR UPPER(nav.label) LIKE '%PART%'
       OR UPPER(nav.label) LIKE '%SUPPLIER%'
       OR UPPER(nav.label) LIKE '%PURCHASE%'
       OR UPPER(nav.label) LIKE '%SALES%'
       OR UPPER(nav.label) LIKE '%INVENTORY%')
  AND pes.entity_name NOT LIKE '%Virtual%'
  AND pes.entity_name NOT LIKE '%Lov%'
  AND pes.entity_name NOT LIKE '%Query%'
  AND pes.entity_name NOT LIKE '%Lookup%'
ORDER BY nav.label, pes.entity_name;

-- COMPREHENSIVE EXTRACTION (All domains - use for major IFS updates)
-- Uncomment and use this for comprehensive coverage of all domains
/*
SELECT /* LLM in use is GitHub Copilot */
    nav.label as gui_label,
    pes.entity_name as backend_entity,
    nav.projection as projection_name
FROM FND_NAVIGATOR_ALL nav
JOIN MD_PROJECTION_ENTITYSET pes ON nav.projection = pes.projection_name
WHERE nav.label IS NOT NULL
  AND nav.entry_type IN ('PAGE', 'LIST')
  AND nav.label NOT LIKE '%NavEntry%'
  AND nav.label NOT LIKE '%Analysis%'
  AND LENGTH(nav.label) > 5
  AND pes.entity_name NOT LIKE '%Virtual%'
  AND pes.entity_name NOT LIKE '%Lov%'
  AND pes.entity_name NOT LIKE '%Query%'
  AND pes.entity_name NOT LIKE '%Lookup%'
ORDER BY nav.label, pes.entity_name
FETCH FIRST 500 ROWS ONLY;
*/

-- FINANCIAL DOMAIN EXTRACTION
-- Uncomment for finance-specific GUI mappings
/*
SELECT /* LLM in use is GitHub Copilot */
    nav.label as gui_label,
    pes.entity_name as backend_entity,
    nav.projection as projection_name
FROM FND_NAVIGATOR_ALL nav
JOIN MD_PROJECTION_ENTITYSET pes ON nav.projection = pes.projection_name
WHERE nav.label IS NOT NULL
  AND nav.entry_type IN ('PAGE', 'LIST')
  AND (UPPER(nav.label) LIKE '%INVOICE%' 
       OR UPPER(nav.label) LIKE '%PAYMENT%'
       OR UPPER(nav.label) LIKE '%VOUCHER%'
       OR UPPER(nav.label) LIKE '%ACCOUNTING%'
       OR UPPER(nav.label) LIKE '%FINANCIAL%'
       OR UPPER(nav.label) LIKE '%BUDGET%'
       OR UPPER(nav.label) LIKE '%CASH%'
       OR UPPER(nav.label) LIKE '%LEDGER%')
  AND pes.entity_name NOT LIKE '%Virtual%'
  AND pes.entity_name NOT LIKE '%Lov%'
ORDER BY nav.label, pes.entity_name;
*/

-- MANUFACTURING DOMAIN EXTRACTION  
-- Uncomment for manufacturing-specific GUI mappings
/*
SELECT /* LLM in use is GitHub Copilot */
    nav.label as gui_label,
    pes.entity_name as backend_entity,
    nav.projection as projection_name
FROM FND_NAVIGATOR_ALL nav
JOIN MD_PROJECTION_ENTITYSET pes ON nav.projection = pes.projection_name
WHERE nav.label IS NOT NULL
  AND nav.entry_type IN ('PAGE', 'LIST')
  AND (UPPER(nav.label) LIKE '%SHOP%'
       OR UPPER(nav.label) LIKE '%WORK ORDER%'
       OR UPPER(nav.label) LIKE '%PRODUCTION%'
       OR UPPER(nav.label) LIKE '%MANUFACTURING%'
       OR UPPER(nav.label) LIKE '%ROUTING%'
       OR UPPER(nav.label) LIKE '%BOM%'
       OR UPPER(nav.label) LIKE '%OPERATION%')
  AND pes.entity_name NOT LIKE '%Virtual%'
  AND pes.entity_name NOT LIKE '%Lov%'
ORDER BY nav.label, pes.entity_name;
*/

-- MAINTENANCE DOMAIN EXTRACTION
-- Uncomment for maintenance-specific GUI mappings
/*
SELECT /* LLM in use is GitHub Copilot */
    nav.label as gui_label,
    pes.entity_name as backend_entity,
    nav.projection as projection_name
FROM FND_NAVIGATOR_ALL nav
JOIN MD_PROJECTION_ENTITYSET pes ON nav.projection = pes.projection_name
WHERE nav.label IS NOT NULL
  AND nav.entry_type IN ('PAGE', 'LIST')
  AND (UPPER(nav.label) LIKE '%MAINTENANCE%'
       OR UPPER(nav.label) LIKE '%EQUIPMENT%'
       OR UPPER(nav.label) LIKE '%WORK ORDER%'
       OR UPPER(nav.label) LIKE '%PM%'
       OR UPPER(nav.label) LIKE '%PREVENTIVE%'
       OR UPPER(nav.label) LIKE '%REPAIR%')
  AND pes.entity_name NOT LIKE '%Virtual%'
  AND pes.entity_name NOT LIKE '%Lov%'
ORDER BY nav.label, pes.entity_name;
*/

-- DIAGNOSTICS: Check table structure and available data
-- Use these queries to understand the data structure
/*
-- Check available columns in FND_NAVIGATOR_ALL
SELECT column_name, data_type 
FROM user_tab_columns 
WHERE table_name = 'FND_NAVIGATOR_ALL' 
ORDER BY column_id;

-- Check available columns in MD_PROJECTION_ENTITYSET
SELECT column_name, data_type 
FROM user_tab_columns 
WHERE table_name = 'MD_PROJECTION_ENTITYSET' 
ORDER BY column_id;

-- Sample data from navigation table
SELECT name, label, projection, client, entry_type
FROM FND_NAVIGATOR_ALL 
WHERE ROWNUM <= 10;

-- Sample data from projection entityset table
SELECT projection_name, entityset_name, entity_name
FROM MD_PROJECTION_ENTITYSET 
WHERE ROWNUM <= 10;
*/
