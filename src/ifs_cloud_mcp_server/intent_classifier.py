"""
Intent Classification for IFS Cloud Search Queries

This module provides ML-based query intent classification to improve search ranking
by understanding what users are actually looking for.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class QueryIntent(Enum):
    """Different types of query intents."""

    BUSINESS_LOGIC = "business_logic"  # authorization, validation, workflows
    ENTITY_DEFINITION = "entity_definition"  # data structure, schema
    UI_COMPONENTS = "ui_components"  # pages, forms, navigation
    API_INTEGRATION = "api_integration"  # projections, services
    DATA_ACCESS = "data_access"  # views, reports
    TROUBLESHOOTING = "troubleshooting"  # errors, debugging
    GENERAL = "general"  # broad topics


@dataclass
class IntentPrediction:
    """Result of intent classification."""

    intent: QueryIntent
    confidence: float
    probabilities: Dict[QueryIntent, float]


class IntentClassifier:
    """ML-based query intent classifier for IFS Cloud searches."""

    def __init__(self, model_path: Optional[Path] = None):
        self.vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), stop_words="english"
        )
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_path = (
            model_path or Path(__file__).parent / "models" / "intent_classifier.pkl"
        )
        self.is_trained = False

        # Try to load existing model
        self._load_model()

    def generate_training_data(self) -> List[Tuple[str, QueryIntent]]:
        """Generate synthetic training data for IFS Cloud queries."""
        training_data = [
            # Business Logic queries - Expanded with IFS-specific terminology
            ("expense authorization workflow", QueryIntent.BUSINESS_LOGIC),
            ("purchase order approval process", QueryIntent.BUSINESS_LOGIC),
            ("invoice validation rules", QueryIntent.BUSINESS_LOGIC),
            ("customer credit check procedure", QueryIntent.BUSINESS_LOGIC),
            ("payroll calculation logic", QueryIntent.BUSINESS_LOGIC),
            ("inventory reservation workflow", QueryIntent.BUSINESS_LOGIC),
            ("project cost calculation", QueryIntent.BUSINESS_LOGIC),
            ("employee status change process", QueryIntent.BUSINESS_LOGIC),
            ("supplier payment authorization", QueryIntent.BUSINESS_LOGIC),
            ("document approval workflow", QueryIntent.BUSINESS_LOGIC),
            ("budget control validation", QueryIntent.BUSINESS_LOGIC),
            ("time reporting business rules", QueryIntent.BUSINESS_LOGIC),
            ("quality control procedures", QueryIntent.BUSINESS_LOGIC),
            ("manufacturing order execution", QueryIntent.BUSINESS_LOGIC),
            ("financial posting logic", QueryIntent.BUSINESS_LOGIC),
            ("user access control", QueryIntent.BUSINESS_LOGIC),
            ("data validation rules", QueryIntent.BUSINESS_LOGIC),
            ("workflow automation", QueryIntent.BUSINESS_LOGIC),
            ("business rule engine", QueryIntent.BUSINESS_LOGIC),
            ("authorization check", QueryIntent.BUSINESS_LOGIC),
            # New comprehensive business logic queries
            ("voucher posting logic", QueryIntent.BUSINESS_LOGIC),
            ("general ledger validation", QueryIntent.BUSINESS_LOGIC),
            ("cost center authorization", QueryIntent.BUSINESS_LOGIC),
            ("tax calculation rules", QueryIntent.BUSINESS_LOGIC),
            ("depreciation calculation method", QueryIntent.BUSINESS_LOGIC),
            ("asset lifecycle management", QueryIntent.BUSINESS_LOGIC),
            ("procurement approval hierarchy", QueryIntent.BUSINESS_LOGIC),
            ("requisition to order workflow", QueryIntent.BUSINESS_LOGIC),
            ("three way matching logic", QueryIntent.BUSINESS_LOGIC),
            ("invoice matching rules", QueryIntent.BUSINESS_LOGIC),
            ("payment terms calculation", QueryIntent.BUSINESS_LOGIC),
            ("dunning process workflow", QueryIntent.BUSINESS_LOGIC),
            ("collection strategy rules", QueryIntent.BUSINESS_LOGIC),
            ("credit limit check", QueryIntent.BUSINESS_LOGIC),
            ("order promising logic", QueryIntent.BUSINESS_LOGIC),
            ("available to promise calculation", QueryIntent.BUSINESS_LOGIC),
            ("capable to promise rules", QueryIntent.BUSINESS_LOGIC),
            ("mrp calculation engine", QueryIntent.BUSINESS_LOGIC),
            ("planning horizon logic", QueryIntent.BUSINESS_LOGIC),
            ("safety stock calculation", QueryIntent.BUSINESS_LOGIC),
            ("reorder point formula", QueryIntent.BUSINESS_LOGIC),
            ("lot sizing rules", QueryIntent.BUSINESS_LOGIC),
            ("shop order release process", QueryIntent.BUSINESS_LOGIC),
            ("production scheduling algorithm", QueryIntent.BUSINESS_LOGIC),
            ("capacity planning calculation", QueryIntent.BUSINESS_LOGIC),
            ("work center loading rules", QueryIntent.BUSINESS_LOGIC),
            ("routing revision control", QueryIntent.BUSINESS_LOGIC),
            ("bom explosion logic", QueryIntent.BUSINESS_LOGIC),
            ("where used calculation", QueryIntent.BUSINESS_LOGIC),
            ("cost rollup procedure", QueryIntent.BUSINESS_LOGIC),
            ("standard cost calculation", QueryIntent.BUSINESS_LOGIC),
            ("variance analysis rules", QueryIntent.BUSINESS_LOGIC),
            ("overhead allocation method", QueryIntent.BUSINESS_LOGIC),
            ("activity based costing logic", QueryIntent.BUSINESS_LOGIC),
            ("transfer pricing rules", QueryIntent.BUSINESS_LOGIC),
            ("intercompany elimination", QueryIntent.BUSINESS_LOGIC),
            ("consolidation process", QueryIntent.BUSINESS_LOGIC),
            ("currency revaluation logic", QueryIntent.BUSINESS_LOGIC),
            ("translation adjustment rules", QueryIntent.BUSINESS_LOGIC),
            ("period closing procedures", QueryIntent.BUSINESS_LOGIC),
            ("year end processing", QueryIntent.BUSINESS_LOGIC),
            ("accrual reversal logic", QueryIntent.BUSINESS_LOGIC),
            ("revenue recognition rules", QueryIntent.BUSINESS_LOGIC),
            ("milestone billing logic", QueryIntent.BUSINESS_LOGIC),
            ("progress billing calculation", QueryIntent.BUSINESS_LOGIC),
            ("retention handling rules", QueryIntent.BUSINESS_LOGIC),
            ("warranty provision calculation", QueryIntent.BUSINESS_LOGIC),
            ("rma processing workflow", QueryIntent.BUSINESS_LOGIC),
            ("return authorization logic", QueryIntent.BUSINESS_LOGIC),
            ("quality inspection rules", QueryIntent.BUSINESS_LOGIC),
            ("non conformance workflow", QueryIntent.BUSINESS_LOGIC),
            ("corrective action process", QueryIntent.BUSINESS_LOGIC),
            ("preventive action logic", QueryIntent.BUSINESS_LOGIC),
            ("audit trail generation", QueryIntent.BUSINESS_LOGIC),
            ("segregation of duties rules", QueryIntent.BUSINESS_LOGIC),
            ("approval limits configuration", QueryIntent.BUSINESS_LOGIC),
            ("delegation of authority", QueryIntent.BUSINESS_LOGIC),
            ("password policy enforcement", QueryIntent.BUSINESS_LOGIC),
            ("session timeout logic", QueryIntent.BUSINESS_LOGIC),
            ("data retention rules", QueryIntent.BUSINESS_LOGIC),
            ("archiving strategy", QueryIntent.BUSINESS_LOGIC),
            ("purge process logic", QueryIntent.BUSINESS_LOGIC),
            ("backup scheduling rules", QueryIntent.BUSINESS_LOGIC),
            ("disaster recovery procedures", QueryIntent.BUSINESS_LOGIC),
            ("business continuity planning", QueryIntent.BUSINESS_LOGIC),
            ("change management workflow", QueryIntent.BUSINESS_LOGIC),
            ("release management process", QueryIntent.BUSINESS_LOGIC),
            ("configuration management rules", QueryIntent.BUSINESS_LOGIC),
            ("incident management workflow", QueryIntent.BUSINESS_LOGIC),
            ("problem resolution process", QueryIntent.BUSINESS_LOGIC),
            ("service level agreement logic", QueryIntent.BUSINESS_LOGIC),
            ("escalation procedures", QueryIntent.BUSINESS_LOGIC),
            ("notification rules engine", QueryIntent.BUSINESS_LOGIC),
            ("alert configuration logic", QueryIntent.BUSINESS_LOGIC),
            ("batch job scheduling", QueryIntent.BUSINESS_LOGIC),
            ("parallel processing logic", QueryIntent.BUSINESS_LOGIC),
            ("load balancing algorithm", QueryIntent.BUSINESS_LOGIC),
            ("resource optimization rules", QueryIntent.BUSINESS_LOGIC),
            ("capacity utilization calculation", QueryIntent.BUSINESS_LOGIC),
            ("performance tuning logic", QueryIntent.BUSINESS_LOGIC),
            
            # Entity Definition queries - Expanded
            ("customer entity structure", QueryIntent.ENTITY_DEFINITION),
            ("employee data model", QueryIntent.ENTITY_DEFINITION),
            ("purchase order fields", QueryIntent.ENTITY_DEFINITION),
            ("invoice entity attributes", QueryIntent.ENTITY_DEFINITION),
            ("project entity definition", QueryIntent.ENTITY_DEFINITION),
            ("supplier table schema", QueryIntent.ENTITY_DEFINITION),
            ("inventory item properties", QueryIntent.ENTITY_DEFINITION),
            ("activity entity", QueryIntent.ENTITY_DEFINITION),
            ("customer order structure", QueryIntent.ENTITY_DEFINITION),
            ("person entity fields", QueryIntent.ENTITY_DEFINITION),
            ("company data structure", QueryIntent.ENTITY_DEFINITION),
            ("address entity definition", QueryIntent.ENTITY_DEFINITION),
            ("contact information fields", QueryIntent.ENTITY_DEFINITION),
            ("product entity attributes", QueryIntent.ENTITY_DEFINITION),
            ("order line structure", QueryIntent.ENTITY_DEFINITION),
            # New entity definition queries
            ("voucher entity model", QueryIntent.ENTITY_DEFINITION),
            ("account entity structure", QueryIntent.ENTITY_DEFINITION),
            ("cost center entity", QueryIntent.ENTITY_DEFINITION),
            ("project activity entity", QueryIntent.ENTITY_DEFINITION),
            ("work order entity fields", QueryIntent.ENTITY_DEFINITION),
            ("shop order entity", QueryIntent.ENTITY_DEFINITION),
            ("routing entity structure", QueryIntent.ENTITY_DEFINITION),
            ("bom header entity", QueryIntent.ENTITY_DEFINITION),
            ("bom line entity", QueryIntent.ENTITY_DEFINITION),
            ("work center entity", QueryIntent.ENTITY_DEFINITION),
            ("operation entity model", QueryIntent.ENTITY_DEFINITION),
            ("inventory location entity", QueryIntent.ENTITY_DEFINITION),
            ("warehouse entity structure", QueryIntent.ENTITY_DEFINITION),
            ("inventory part entity", QueryIntent.ENTITY_DEFINITION),
            ("sales part entity", QueryIntent.ENTITY_DEFINITION),
            ("purchase part entity", QueryIntent.ENTITY_DEFINITION),
            ("engineering part entity", QueryIntent.ENTITY_DEFINITION),
            ("serial number entity", QueryIntent.ENTITY_DEFINITION),
            ("lot batch entity", QueryIntent.ENTITY_DEFINITION),
            ("handling unit entity", QueryIntent.ENTITY_DEFINITION),
            ("shipment entity structure", QueryIntent.ENTITY_DEFINITION),
            ("delivery entity model", QueryIntent.ENTITY_DEFINITION),
            ("transport task entity", QueryIntent.ENTITY_DEFINITION),
            ("pick list entity", QueryIntent.ENTITY_DEFINITION),
            ("counting report entity", QueryIntent.ENTITY_DEFINITION),
            ("adjustment entity structure", QueryIntent.ENTITY_DEFINITION),
            ("requisition entity", QueryIntent.ENTITY_DEFINITION),
            ("rfq entity model", QueryIntent.ENTITY_DEFINITION),
            ("quotation entity structure", QueryIntent.ENTITY_DEFINITION),
            ("agreement entity", QueryIntent.ENTITY_DEFINITION),
            ("contract entity fields", QueryIntent.ENTITY_DEFINITION),
            ("payment entity structure", QueryIntent.ENTITY_DEFINITION),
            ("payment plan entity", QueryIntent.ENTITY_DEFINITION),
            ("tax code entity", QueryIntent.ENTITY_DEFINITION),
            ("currency entity model", QueryIntent.ENTITY_DEFINITION),
            ("exchange rate entity", QueryIntent.ENTITY_DEFINITION),
            ("period entity structure", QueryIntent.ENTITY_DEFINITION),
            ("ledger entity model", QueryIntent.ENTITY_DEFINITION),
            ("budget entity structure", QueryIntent.ENTITY_DEFINITION),
            ("forecast entity", QueryIntent.ENTITY_DEFINITION),
            ("kpi entity model", QueryIntent.ENTITY_DEFINITION),
            ("document entity structure", QueryIntent.ENTITY_DEFINITION),
            ("attachment entity", QueryIntent.ENTITY_DEFINITION),
            ("note entity fields", QueryIntent.ENTITY_DEFINITION),
            ("message entity structure", QueryIntent.ENTITY_DEFINITION),
            ("notification entity", QueryIntent.ENTITY_DEFINITION),
            ("task entity model", QueryIntent.ENTITY_DEFINITION),
            ("appointment entity", QueryIntent.ENTITY_DEFINITION),
            ("calendar entity structure", QueryIntent.ENTITY_DEFINITION),
            ("resource entity", QueryIntent.ENTITY_DEFINITION),
            ("equipment entity model", QueryIntent.ENTITY_DEFINITION),
            ("tool entity structure", QueryIntent.ENTITY_DEFINITION),
            ("maintenance object entity", QueryIntent.ENTITY_DEFINITION),
            ("service request entity", QueryIntent.ENTITY_DEFINITION),
            ("case entity structure", QueryIntent.ENTITY_DEFINITION),
            ("solution entity model", QueryIntent.ENTITY_DEFINITION),
            
            # UI Components queries - Expanded
            ("customer order entry page", QueryIntent.UI_COMPONENTS),
            ("employee management form", QueryIntent.UI_COMPONENTS),
            ("purchase order navigator", QueryIntent.UI_COMPONENTS),
            ("invoice processing page", QueryIntent.UI_COMPONENTS),
            ("project management interface", QueryIntent.UI_COMPONENTS),
            ("inventory tracking screen", QueryIntent.UI_COMPONENTS),
            ("supplier registration form", QueryIntent.UI_COMPONENTS),
            ("report generation page", QueryIntent.UI_COMPONENTS),
            ("user profile settings", QueryIntent.UI_COMPONENTS),
            ("dashboard components", QueryIntent.UI_COMPONENTS),
            ("navigation menu", QueryIntent.UI_COMPONENTS),
            ("search interface", QueryIntent.UI_COMPONENTS),
            ("data entry form", QueryIntent.UI_COMPONENTS),
            ("list view", QueryIntent.UI_COMPONENTS),
            ("tree navigator", QueryIntent.UI_COMPONENTS),
            # New UI component queries
            ("aurena client page", QueryIntent.UI_COMPONENTS),
            ("lobby page configuration", QueryIntent.UI_COMPONENTS),
            ("quick report interface", QueryIntent.UI_COMPONENTS),
            ("crystal report viewer", QueryIntent.UI_COMPONENTS),
            ("operational report page", QueryIntent.UI_COMPONENTS),
            ("business analytics dashboard", QueryIntent.UI_COMPONENTS),
            ("kpi dashboard widget", QueryIntent.UI_COMPONENTS),
            ("chart component", QueryIntent.UI_COMPONENTS),
            ("graph visualization", QueryIntent.UI_COMPONENTS),
            ("gantt chart view", QueryIntent.UI_COMPONENTS),
            ("calendar widget", QueryIntent.UI_COMPONENTS),
            ("timeline component", QueryIntent.UI_COMPONENTS),
            ("kanban board interface", QueryIntent.UI_COMPONENTS),
            ("workflow designer", QueryIntent.UI_COMPONENTS),
            ("process modeler", QueryIntent.UI_COMPONENTS),
            ("form designer interface", QueryIntent.UI_COMPONENTS),
            ("report designer page", QueryIntent.UI_COMPONENTS),
            ("query builder interface", QueryIntent.UI_COMPONENTS),
            ("filter panel component", QueryIntent.UI_COMPONENTS),
            ("search criteria form", QueryIntent.UI_COMPONENTS),
            ("advanced search dialog", QueryIntent.UI_COMPONENTS),
            ("context menu component", QueryIntent.UI_COMPONENTS),
            ("toolbar configuration", QueryIntent.UI_COMPONENTS),
            ("ribbon interface", QueryIntent.UI_COMPONENTS),
            ("tab container", QueryIntent.UI_COMPONENTS),
            ("accordion panel", QueryIntent.UI_COMPONENTS),
            ("collapsible section", QueryIntent.UI_COMPONENTS),
            ("modal dialog", QueryIntent.UI_COMPONENTS),
            ("popup window", QueryIntent.UI_COMPONENTS),
            ("notification toast", QueryIntent.UI_COMPONENTS),
            ("alert message component", QueryIntent.UI_COMPONENTS),
            ("confirmation dialog", QueryIntent.UI_COMPONENTS),
            ("wizard interface", QueryIntent.UI_COMPONENTS),
            ("stepper component", QueryIntent.UI_COMPONENTS),
            ("progress indicator", QueryIntent.UI_COMPONENTS),
            ("loading spinner", QueryIntent.UI_COMPONENTS),
            ("data grid component", QueryIntent.UI_COMPONENTS),
            ("table view interface", QueryIntent.UI_COMPONENTS),
            ("card layout", QueryIntent.UI_COMPONENTS),
            ("tile view", QueryIntent.UI_COMPONENTS),
            ("master detail page", QueryIntent.UI_COMPONENTS),
            ("split view interface", QueryIntent.UI_COMPONENTS),
            ("sidebar navigation", QueryIntent.UI_COMPONENTS),
            ("breadcrumb component", QueryIntent.UI_COMPONENTS),
            ("pagination control", QueryIntent.UI_COMPONENTS),
            ("infinite scroll list", QueryIntent.UI_COMPONENTS),
            ("drag drop interface", QueryIntent.UI_COMPONENTS),
            ("file upload component", QueryIntent.UI_COMPONENTS),
            ("image gallery widget", QueryIntent.UI_COMPONENTS),
            ("video player component", QueryIntent.UI_COMPONENTS),
            ("document viewer", QueryIntent.UI_COMPONENTS),
            ("pdf preview component", QueryIntent.UI_COMPONENTS),
            ("print preview dialog", QueryIntent.UI_COMPONENTS),
            ("export dialog interface", QueryIntent.UI_COMPONENTS),
            ("import wizard page", QueryIntent.UI_COMPONENTS),
            ("batch processing interface", QueryIntent.UI_COMPONENTS),
            ("bulk update form", QueryIntent.UI_COMPONENTS),
            ("mass delete dialog", QueryIntent.UI_COMPONENTS),
            ("selection list component", QueryIntent.UI_COMPONENTS),
            ("multi select dropdown", QueryIntent.UI_COMPONENTS),
            ("autocomplete field", QueryIntent.UI_COMPONENTS),
            ("type ahead search", QueryIntent.UI_COMPONENTS),
            ("date picker component", QueryIntent.UI_COMPONENTS),
            ("time selector widget", QueryIntent.UI_COMPONENTS),
            ("datetime picker", QueryIntent.UI_COMPONENTS),
            ("range slider component", QueryIntent.UI_COMPONENTS),
            ("toggle switch", QueryIntent.UI_COMPONENTS),
            ("radio button group", QueryIntent.UI_COMPONENTS),
            ("checkbox list", QueryIntent.UI_COMPONENTS),
            ("rating component", QueryIntent.UI_COMPONENTS),
            ("color picker widget", QueryIntent.UI_COMPONENTS),
            ("rich text editor", QueryIntent.UI_COMPONENTS),
            ("markdown editor component", QueryIntent.UI_COMPONENTS),
            ("code editor interface", QueryIntent.UI_COMPONENTS),
            ("syntax highlighter", QueryIntent.UI_COMPONENTS),
            
            # API Integration queries - Expanded
            ("customer order api", QueryIntent.API_INTEGRATION),
            ("employee service endpoint", QueryIntent.API_INTEGRATION),
            ("purchase order projection", QueryIntent.API_INTEGRATION),
            ("invoice api integration", QueryIntent.API_INTEGRATION),
            ("project management service", QueryIntent.API_INTEGRATION),
            ("inventory api calls", QueryIntent.API_INTEGRATION),
            ("supplier data service", QueryIntent.API_INTEGRATION),
            ("rest api endpoints", QueryIntent.API_INTEGRATION),
            ("web service integration", QueryIntent.API_INTEGRATION),
            ("api documentation", QueryIntent.API_INTEGRATION),
            ("service contract", QueryIntent.API_INTEGRATION),
            ("projection mapping", QueryIntent.API_INTEGRATION),
            ("external api", QueryIntent.API_INTEGRATION),
            ("integration points", QueryIntent.API_INTEGRATION),
            # New API integration queries
            ("odata service endpoint", QueryIntent.API_INTEGRATION),
            ("soap web service", QueryIntent.API_INTEGRATION),
            ("graphql api schema", QueryIntent.API_INTEGRATION),
            ("webhook configuration", QueryIntent.API_INTEGRATION),
            ("event subscription api", QueryIntent.API_INTEGRATION),
            ("message queue integration", QueryIntent.API_INTEGRATION),
            ("kafka connector api", QueryIntent.API_INTEGRATION),
            ("service bus integration", QueryIntent.API_INTEGRATION),
            ("batch api endpoint", QueryIntent.API_INTEGRATION),
            ("bulk operation service", QueryIntent.API_INTEGRATION),
            ("streaming api interface", QueryIntent.API_INTEGRATION),
            ("real time data service", QueryIntent.API_INTEGRATION),
            ("push notification api", QueryIntent.API_INTEGRATION),
            ("websocket connection", QueryIntent.API_INTEGRATION),
            ("long polling service", QueryIntent.API_INTEGRATION),
            ("server sent events", QueryIntent.API_INTEGRATION),
            ("file transfer api", QueryIntent.API_INTEGRATION),
            ("ftp integration service", QueryIntent.API_INTEGRATION),
            ("sftp connector api", QueryIntent.API_INTEGRATION),
            ("email service integration", QueryIntent.API_INTEGRATION),
            ("smtp configuration api", QueryIntent.API_INTEGRATION),
            ("sms gateway integration", QueryIntent.API_INTEGRATION),
            ("payment gateway api", QueryIntent.API_INTEGRATION),
            ("credit card service", QueryIntent.API_INTEGRATION),
            ("bank integration api", QueryIntent.API_INTEGRATION),
            ("edi message service", QueryIntent.API_INTEGRATION),
            ("xml data exchange", QueryIntent.API_INTEGRATION),
            ("json api endpoint", QueryIntent.API_INTEGRATION),
            ("csv import service", QueryIntent.API_INTEGRATION),
            ("excel export api", QueryIntent.API_INTEGRATION),
            ("pdf generation service", QueryIntent.API_INTEGRATION),
            ("document conversion api", QueryIntent.API_INTEGRATION),
            ("ocr service integration", QueryIntent.API_INTEGRATION),
            ("barcode scanning api", QueryIntent.API_INTEGRATION),
            ("qr code service", QueryIntent.API_INTEGRATION),
            ("biometric integration", QueryIntent.API_INTEGRATION),
            ("single sign on api", QueryIntent.API_INTEGRATION),
            ("oauth2 integration", QueryIntent.API_INTEGRATION),
            ("saml service provider", QueryIntent.API_INTEGRATION),
            ("ldap connector api", QueryIntent.API_INTEGRATION),
            ("active directory service", QueryIntent.API_INTEGRATION),
            ("azure ad integration", QueryIntent.API_INTEGRATION),
            ("google api connector", QueryIntent.API_INTEGRATION),
            ("aws service integration", QueryIntent.API_INTEGRATION),
            ("salesforce connector api", QueryIntent.API_INTEGRATION),
            ("sap integration service", QueryIntent.API_INTEGRATION),
            ("oracle connector api", QueryIntent.API_INTEGRATION),
            ("microsoft dynamics integration", QueryIntent.API_INTEGRATION),
            ("quickbooks api connector", QueryIntent.API_INTEGRATION),
            ("blockchain integration api", QueryIntent.API_INTEGRATION),
            ("iot device service", QueryIntent.API_INTEGRATION),
            ("sensor data api", QueryIntent.API_INTEGRATION),
            ("telemetry service endpoint", QueryIntent.API_INTEGRATION),
            ("analytics api integration", QueryIntent.API_INTEGRATION),
            ("machine learning service", QueryIntent.API_INTEGRATION),
            ("ai model api endpoint", QueryIntent.API_INTEGRATION),
            ("chatbot integration service", QueryIntent.API_INTEGRATION),
            ("voice assistant api", QueryIntent.API_INTEGRATION),
            ("translation service api", QueryIntent.API_INTEGRATION),
            ("geocoding api integration", QueryIntent.API_INTEGRATION),
            ("mapping service endpoint", QueryIntent.API_INTEGRATION),
            ("weather api integration", QueryIntent.API_INTEGRATION),
            ("shipping carrier api", QueryIntent.API_INTEGRATION),
            ("tracking service integration", QueryIntent.API_INTEGRATION),
            ("customs clearance api", QueryIntent.API_INTEGRATION),
            
            # Data Access queries - Expanded
            ("customer order report", QueryIntent.DATA_ACCESS),
            ("employee status view", QueryIntent.DATA_ACCESS),
            ("purchase order history", QueryIntent.DATA_ACCESS),
            ("invoice summary report", QueryIntent.DATA_ACCESS),
            ("project cost view", QueryIntent.DATA_ACCESS),
            ("inventory levels report", QueryIntent.DATA_ACCESS),
            ("supplier performance view", QueryIntent.DATA_ACCESS),
            ("financial statements", QueryIntent.DATA_ACCESS),
            ("audit trail view", QueryIntent.DATA_ACCESS),
            ("management dashboard", QueryIntent.DATA_ACCESS),
            ("kpi reports", QueryIntent.DATA_ACCESS),
            ("data analytics view", QueryIntent.DATA_ACCESS),
            ("summary report", QueryIntent.DATA_ACCESS),
            ("detailed view", QueryIntent.DATA_ACCESS),
            # New data access queries
            ("general ledger report", QueryIntent.DATA_ACCESS),
            ("trial balance view", QueryIntent.DATA_ACCESS),
            ("balance sheet report", QueryIntent.DATA_ACCESS),
            ("income statement view", QueryIntent.DATA_ACCESS),
            ("cash flow report", QueryIntent.DATA_ACCESS),
            ("aged receivables report", QueryIntent.DATA_ACCESS),
            ("aged payables view", QueryIntent.DATA_ACCESS),
            ("inventory valuation report", QueryIntent.DATA_ACCESS),
            ("stock movement view", QueryIntent.DATA_ACCESS),
            ("sales analysis report", QueryIntent.DATA_ACCESS),
            ("revenue forecast view", QueryIntent.DATA_ACCESS),
            ("demand planning report", QueryIntent.DATA_ACCESS),
            ("supply chain view", QueryIntent.DATA_ACCESS),
            ("production report", QueryIntent.DATA_ACCESS),
            ("shop floor view", QueryIntent.DATA_ACCESS),
            ("quality metrics report", QueryIntent.DATA_ACCESS),
            ("defect analysis view", QueryIntent.DATA_ACCESS),
            ("maintenance history report", QueryIntent.DATA_ACCESS),
            ("equipment uptime view", QueryIntent.DATA_ACCESS),
            ("resource utilization report", QueryIntent.DATA_ACCESS),
            ("capacity planning view", QueryIntent.DATA_ACCESS),
            ("project status report", QueryIntent.DATA_ACCESS),
            ("milestone tracking view", QueryIntent.DATA_ACCESS),
            ("budget variance report", QueryIntent.DATA_ACCESS),
            ("cost analysis view", QueryIntent.DATA_ACCESS),
            ("profitability report", QueryIntent.DATA_ACCESS),
            ("margin analysis view", QueryIntent.DATA_ACCESS),
            ("customer profitability report", QueryIntent.DATA_ACCESS),
            ("product profitability view", QueryIntent.DATA_ACCESS),
            ("sales pipeline report", QueryIntent.DATA_ACCESS),
            ("opportunity tracking view", QueryIntent.DATA_ACCESS),
            ("lead conversion report", QueryIntent.DATA_ACCESS),
            ("customer satisfaction view", QueryIntent.DATA_ACCESS),
            ("service level report", QueryIntent.DATA_ACCESS),
            ("ticket resolution view", QueryIntent.DATA_ACCESS),
            ("employee productivity report", QueryIntent.DATA_ACCESS),
            ("attendance tracking view", QueryIntent.DATA_ACCESS),
            ("leave balance report", QueryIntent.DATA_ACCESS),
            ("payroll summary view", QueryIntent.DATA_ACCESS),
            ("tax compliance report", QueryIntent.DATA_ACCESS),
            ("regulatory reporting view", QueryIntent.DATA_ACCESS),
            ("customs declaration report", QueryIntent.DATA_ACCESS),
            ("shipping manifest view", QueryIntent.DATA_ACCESS),
            ("delivery performance report", QueryIntent.DATA_ACCESS),
            ("order fulfillment view", QueryIntent.DATA_ACCESS),
            ("backorder report", QueryIntent.DATA_ACCESS),
            ("stockout analysis view", QueryIntent.DATA_ACCESS),
            ("abc analysis report", QueryIntent.DATA_ACCESS),
            ("xyz classification view", QueryIntent.DATA_ACCESS),
            ("pareto analysis report", QueryIntent.DATA_ACCESS),
            ("trend analysis view", QueryIntent.DATA_ACCESS),
            ("seasonality report", QueryIntent.DATA_ACCESS),
            ("correlation analysis view", QueryIntent.DATA_ACCESS),
            ("regression analysis report", QueryIntent.DATA_ACCESS),
            ("predictive analytics view", QueryIntent.DATA_ACCESS),
            ("dashboard metrics report", QueryIntent.DATA_ACCESS),
            ("executive summary view", QueryIntent.DATA_ACCESS),
            ("board report package", QueryIntent.DATA_ACCESS),
            ("investor relations view", QueryIntent.DATA_ACCESS),
            ("sustainability report", QueryIntent.DATA_ACCESS),
            ("carbon footprint view", QueryIntent.DATA_ACCESS),
            ("energy consumption report", QueryIntent.DATA_ACCESS),
            ("waste management view", QueryIntent.DATA_ACCESS),
            
            # Troubleshooting queries - Expanded
            ("error handling", QueryIntent.TROUBLESHOOTING),
            ("debug logging", QueryIntent.TROUBLESHOOTING),
            ("exception management", QueryIntent.TROUBLESHOOTING),
            ("error messages", QueryIntent.TROUBLESHOOTING),
            ("system diagnostics", QueryIntent.TROUBLESHOOTING),
            ("troubleshooting guide", QueryIntent.TROUBLESHOOTING),
            ("performance issues", QueryIntent.TROUBLESHOOTING),
            ("bug fixes", QueryIntent.TROUBLESHOOTING),
            ("system errors", QueryIntent.TROUBLESHOOTING),
            # New troubleshooting queries
            ("ora error code", QueryIntent.TROUBLESHOOTING),
            ("sql exception handling", QueryIntent.TROUBLESHOOTING),
            ("database deadlock issue", QueryIntent.TROUBLESHOOTING),
            ("connection timeout error", QueryIntent.TROUBLESHOOTING),
            ("memory leak problem", QueryIntent.TROUBLESHOOTING),
            ("cpu usage spike", QueryIntent.TROUBLESHOOTING),
            ("disk space issue", QueryIntent.TROUBLESHOOTING),
            ("network latency problem", QueryIntent.TROUBLESHOOTING),
            ("authentication failure", QueryIntent.TROUBLESHOOTING),
            ("authorization denied error", QueryIntent.TROUBLESHOOTING),
            ("session expired issue", QueryIntent.TROUBLESHOOTING),
            ("token validation error", QueryIntent.TROUBLESHOOTING),
            ("certificate problem", QueryIntent.TROUBLESHOOTING),
            ("ssl handshake failure", QueryIntent.TROUBLESHOOTING),
            ("proxy connection issue", QueryIntent.TROUBLESHOOTING),
            ("firewall blocking error", QueryIntent.TROUBLESHOOTING),
            ("port already in use", QueryIntent.TROUBLESHOOTING),
            ("service unavailable error", QueryIntent.TROUBLESHOOTING),
            ("gateway timeout issue", QueryIntent.TROUBLESHOOTING),
            ("bad request error", QueryIntent.TROUBLESHOOTING),
            ("internal server error", QueryIntent.TROUBLESHOOTING),
            ("null pointer exception", QueryIntent.TROUBLESHOOTING),
            ("array index out of bounds", QueryIntent.TROUBLESHOOTING),
            ("stack overflow error", QueryIntent.TROUBLESHOOTING),
            ("heap space exhausted", QueryIntent.TROUBLESHOOTING),
            ("thread deadlock issue", QueryIntent.TROUBLESHOOTING),
            ("race condition problem", QueryIntent.TROUBLESHOOTING),
            ("concurrency issue", QueryIntent.TROUBLESHOOTING),
            ("transaction rollback error", QueryIntent.TROUBLESHOOTING),
            ("commit failed issue", QueryIntent.TROUBLESHOOTING),
            ("lock wait timeout", QueryIntent.TROUBLESHOOTING),
            ("duplicate key error", QueryIntent.TROUBLESHOOTING),
            ("foreign key violation", QueryIntent.TROUBLESHOOTING),
            ("constraint violation error", QueryIntent.TROUBLESHOOTING),
            ("data integrity issue", QueryIntent.TROUBLESHOOTING),
            ("corruption detected error", QueryIntent.TROUBLESHOOTING),
            ("checksum mismatch problem", QueryIntent.TROUBLESHOOTING),
            ("version conflict issue", QueryIntent.TROUBLESHOOTING),
            ("compatibility problem", QueryIntent.TROUBLESHOOTING),
            ("dependency missing error", QueryIntent.TROUBLESHOOTING),
            ("module not found issue", QueryIntent.TROUBLESHOOTING),
            ("class not found error", QueryIntent.TROUBLESHOOTING),
            ("method not implemented", QueryIntent.TROUBLESHOOTING),
            ("api deprecated warning", QueryIntent.TROUBLESHOOTING),
            ("configuration error", QueryIntent.TROUBLESHOOTING),
            ("invalid parameter issue", QueryIntent.TROUBLESHOOTING),
            ("parsing error problem", QueryIntent.TROUBLESHOOTING),
            ("serialization failure", QueryIntent.TROUBLESHOOTING),
            ("encoding issue error", QueryIntent.TROUBLESHOOTING),
            ("character set problem", QueryIntent.TROUBLESHOOTING),
            ("locale not supported", QueryIntent.TROUBLESHOOTING),
            ("timezone issue error", QueryIntent.TROUBLESHOOTING),
            ("date format problem", QueryIntent.TROUBLESHOOTING),
            ("number format exception", QueryIntent.TROUBLESHOOTING),
            ("overflow error issue", QueryIntent.TROUBLESHOOTING),
            ("underflow problem", QueryIntent.TROUBLESHOOTING),
            ("precision loss error", QueryIntent.TROUBLESHOOTING),
            ("rounding issue problem", QueryIntent.TROUBLESHOOTING),
            ("division by zero error", QueryIntent.TROUBLESHOOTING),
            ("infinite loop detected", QueryIntent.TROUBLESHOOTING),
            ("recursion limit exceeded", QueryIntent.TROUBLESHOOTING),
            ("timeout exceeded error", QueryIntent.TROUBLESHOOTING),
            ("rate limit exceeded", QueryIntent.TROUBLESHOOTING),
            ("quota exceeded issue", QueryIntent.TROUBLESHOOTING),
            ("license expired error", QueryIntent.TROUBLESHOOTING),
            ("subscription invalid", QueryIntent.TROUBLESHOOTING),
            
            # General queries - Expanded
            ("customer management", QueryIntent.GENERAL),
            ("employee information", QueryIntent.GENERAL),
            ("purchase orders", QueryIntent.GENERAL),
            ("invoicing", QueryIntent.GENERAL),
            ("project management", QueryIntent.GENERAL),
            ("inventory", QueryIntent.GENERAL),
            ("suppliers", QueryIntent.GENERAL),
            ("financial", QueryIntent.GENERAL),
            ("reporting", QueryIntent.GENERAL),
            ("user management", QueryIntent.GENERAL),
            ("system administration", QueryIntent.GENERAL),
            ("configuration", QueryIntent.GENERAL),
            # New general queries
            ("ifs cloud", QueryIntent.GENERAL),
            ("enterprise resource planning", QueryIntent.GENERAL),
            ("erp system", QueryIntent.GENERAL),
            ("business software", QueryIntent.GENERAL),
            ("cloud application", QueryIntent.GENERAL),
            ("saas platform", QueryIntent.GENERAL),
            ("multi tenant", QueryIntent.GENERAL),
            ("microservices", QueryIntent.GENERAL),
            ("containerization", QueryIntent.GENERAL),
            ("kubernetes deployment", QueryIntent.GENERAL),
            ("docker container", QueryIntent.GENERAL),
            ("cloud native", QueryIntent.GENERAL),
            ("serverless function", QueryIntent.GENERAL),
            ("event driven", QueryIntent.GENERAL),
            ("message broker", QueryIntent.GENERAL),
            ("service mesh", QueryIntent.GENERAL),
            ("api gateway", QueryIntent.GENERAL),
            ("load balancer", QueryIntent.GENERAL),
            ("reverse proxy", QueryIntent.GENERAL),
            ("caching layer", QueryIntent.GENERAL),
            ("database cluster", QueryIntent.GENERAL),
            ("data replication", QueryIntent.GENERAL),
            ("backup strategy", QueryIntent.GENERAL),
            ("disaster recovery", QueryIntent.GENERAL),
            ("high availability", QueryIntent.GENERAL),
            ("fault tolerance", QueryIntent.GENERAL),
            ("scalability", QueryIntent.GENERAL),
            ("performance optimization", QueryIntent.GENERAL),
            ("security compliance", QueryIntent.GENERAL),
            ("gdpr compliance", QueryIntent.GENERAL),
            ("sox compliance", QueryIntent.GENERAL),
            ("hipaa compliance", QueryIntent.GENERAL),
            ("pci dss", QueryIntent.GENERAL),
            ("iso certification", QueryIntent.GENERAL),
            ("audit compliance", QueryIntent.GENERAL),
            ("risk management", QueryIntent.GENERAL),
            ("change control", QueryIntent.GENERAL),
            ("version control", QueryIntent.GENERAL),
            ("release notes", QueryIntent.GENERAL),
            ("documentation", QueryIntent.GENERAL),
            ("user guide", QueryIntent.GENERAL),
            ("administrator manual", QueryIntent.GENERAL),
            ("developer documentation", QueryIntent.GENERAL),
            ("api reference", QueryIntent.GENERAL),
            ("technical specification", QueryIntent.GENERAL),
            ("functional specification", QueryIntent.GENERAL),
            ("design document", QueryIntent.GENERAL),
            ("architecture diagram", QueryIntent.GENERAL),
            ("data flow diagram", QueryIntent.GENERAL),
            ("entity relationship", QueryIntent.GENERAL),
            ("use case diagram", QueryIntent.GENERAL),
            ("sequence diagram", QueryIntent.GENERAL),
            ("class diagram", QueryIntent.GENERAL),
            ("deployment diagram", QueryIntent.GENERAL),
            ("component diagram", QueryIntent.GENERAL),
            ("state machine diagram", QueryIntent.GENERAL),
            ("activity diagram", QueryIntent.GENERAL),
            ("business process model", QueryIntent.GENERAL),
            ("workflow diagram", QueryIntent.GENERAL),
            ("system overview", QueryIntent.GENERAL),
            ("module description", QueryIntent.GENERAL),
            ("feature list", QueryIntent.GENERAL),
            ("product roadmap", QueryIntent.GENERAL),
            ("release schedule", QueryIntent.GENERAL),
            ("milestone plan", QueryIntent.GENERAL),
            ("project timeline", QueryIntent.GENERAL),
            ("resource allocation", QueryIntent.GENERAL),
            ("team structure", QueryIntent.GENERAL),
            ("organization chart", QueryIntent.GENERAL),
            ("responsibility matrix", QueryIntent.GENERAL),
            ("stakeholder analysis", QueryIntent.GENERAL),
            ("requirements gathering", QueryIntent.GENERAL),
            ("gap analysis", QueryIntent.GENERAL),
            ("feasibility study", QueryIntent.GENERAL),
            ("proof of concept", QueryIntent.GENERAL),
            ("pilot project", QueryIntent.GENERAL),
            ("rollout plan", QueryIntent.GENERAL),
            ("training material", QueryIntent.GENERAL),
            ("knowledge base", QueryIntent.GENERAL),
            ("faq section", QueryIntent.GENERAL),
            ("help documentation", QueryIntent.GENERAL),
            ("support ticket", QueryIntent.GENERAL),
            ("service request", QueryIntent.GENERAL),
            ("enhancement request", QueryIntent.GENERAL),
            ("feature request", QueryIntent.GENERAL),
            ("bug report", QueryIntent.GENERAL),
            ("incident report", QueryIntent.GENERAL),
            ("problem statement", QueryIntent.GENERAL),
            ("root cause analysis", QueryIntent.GENERAL),
            ("lessons learned", QueryIntent.GENERAL),
            ("best practices", QueryIntent.GENERAL),
            ("coding standards", QueryIntent.GENERAL),
            ("naming conventions", QueryIntent.GENERAL),
            ("style guide", QueryIntent.GENERAL),
            ("review checklist", QueryIntent.GENERAL),
            ("quality metrics", QueryIntent.GENERAL),
            ("performance metrics", QueryIntent.GENERAL),
            ("success criteria", QueryIntent.GENERAL),
            ("acceptance criteria", QueryIntent.GENERAL),
            ("test scenarios", QueryIntent.GENERAL),
            ("test cases", QueryIntent.GENERAL),
            ("test plan", QueryIntent.GENERAL),
            ("test results", QueryIntent.GENERAL),
            ("defect tracking", QueryIntent.GENERAL),
            ("issue management", QueryIntent.GENERAL),
            ("project status", QueryIntent.GENERAL),
            ("progress report", QueryIntent.GENERAL),
            ("status update", QueryIntent.GENERAL),
            ("meeting minutes", QueryIntent.GENERAL),
            ("action items", QueryIntent.GENERAL),
            ("decision log", QueryIntent.GENERAL),
            ("communication plan", QueryIntent.GENERAL),
            ("stakeholder communication", QueryIntent.GENERAL),
            ("executive briefing", QueryIntent.GENERAL),
            ("business case", QueryIntent.GENERAL),
            ("cost benefit analysis", QueryIntent.GENERAL),
            ("return on investment", QueryIntent.GENERAL),
            ("total cost ownership", QueryIntent.GENERAL),
            ("value proposition", QueryIntent.GENERAL),
            ("competitive advantage", QueryIntent.GENERAL),
            ("market analysis", QueryIntent.GENERAL),
            ("industry trends", QueryIntent.GENERAL),
            ("customer feedback", QueryIntent.GENERAL),
            ("user experience", QueryIntent.GENERAL),
            ("usability testing", QueryIntent.GENERAL),
            ("accessibility compliance", QueryIntent.GENERAL),
            ("localization support", QueryIntent.GENERAL),
            ("internationalization", QueryIntent.GENERAL),
            ("multi language", QueryIntent.GENERAL),
            ("multi currency", QueryIntent.GENERAL),
            ("multi company", QueryIntent.GENERAL),
            ("cross company", QueryIntent.GENERAL),
            ("intercompany", QueryIntent.GENERAL),
            ("consolidation", QueryIntent.GENERAL),
            ("group reporting", QueryIntent.GENERAL),
            ("corporate governance", QueryIntent.GENERAL),
            ("compliance management", QueryIntent.GENERAL),
            ("internal controls", QueryIntent.GENERAL),
            ("risk assessment", QueryIntent.GENERAL),
            ("control framework", QueryIntent.GENERAL),
            ("policy management", QueryIntent.GENERAL),
            ("procedure documentation", QueryIntent.GENERAL),
            ("work instructions", QueryIntent.GENERAL),
            ("standard operating procedure", QueryIntent.GENERAL),
            ("process improvement", QueryIntent.GENERAL),
            ("continuous improvement", QueryIntent.GENERAL),
            ("lean methodology", QueryIntent.GENERAL),
            ("six sigma", QueryIntent.GENERAL),
            ("agile methodology", QueryIntent.GENERAL),
            ("scrum framework", QueryIntent.GENERAL),
            ("kanban method", QueryIntent.GENERAL),
            ("devops practices", QueryIntent.GENERAL),
            ("continuous integration", QueryIntent.GENERAL),
            ("continuous deployment", QueryIntent.GENERAL),
            ("automated testing", QueryIntent.GENERAL),
            ("test automation", QueryIntent.GENERAL),
            ("regression testing", QueryIntent.GENERAL),
            ("integration testing", QueryIntent.GENERAL),
            ("unit testing", QueryIntent.GENERAL),
            ("system testing", QueryIntent.GENERAL),
            ("acceptance testing", QueryIntent.GENERAL),
            ("performance testing", QueryIntent.GENERAL),
            ("load testing", QueryIntent.GENERAL),
            ("stress testing", QueryIntent.GENERAL),
            ("security testing", QueryIntent.GENERAL),
            ("penetration testing", QueryIntent.GENERAL),
            ("vulnerability assessment", QueryIntent.GENERAL),
            ("code review", QueryIntent.GENERAL),
            ("peer review", QueryIntent.GENERAL),
            ("quality assurance", QueryIntent.GENERAL),
            ("quality control", QueryIntent.GENERAL),
            ("configuration management", QueryIntent.GENERAL),
            ("change management", QueryIntent.GENERAL),
            ("release management", QueryIntent.GENERAL),
            ("deployment management", QueryIntent.GENERAL),
            ("environment management", QueryIntent.GENERAL),
            ("infrastructure management", QueryIntent.GENERAL),
            ("capacity management", QueryIntent.GENERAL),
            ("availability management", QueryIntent.GENERAL),
            ("incident management", QueryIntent.GENERAL),
            ("problem management", QueryIntent.GENERAL),
            ("knowledge management", QueryIntent.GENERAL),
            ("service catalog", QueryIntent.GENERAL),
            ("service level agreement", QueryIntent.GENERAL),
            ("operational level agreement", QueryIntent.GENERAL),
            ("key performance indicator", QueryIntent.GENERAL),
            ("critical success factor", QueryIntent.GENERAL),
            ("balanced scorecard", QueryIntent.GENERAL),
            ("maturity model", QueryIntent.GENERAL),
            ("capability assessment", QueryIntent.GENERAL),
            ("readiness assessment", QueryIntent.GENERAL),
            ("impact analysis", QueryIntent.GENERAL),
            ("dependency analysis", QueryIntent.GENERAL),
            ("traceability matrix", QueryIntent.GENERAL),
            ("coverage analysis", QueryIntent.GENERAL),
            ("code coverage", QueryIntent.GENERAL),
            ("test coverage", QueryIntent.GENERAL),
            ("requirement coverage", QueryIntent.GENERAL),
        ]

        return training_data

    def train(self, training_data: Optional[List[Tuple[str, QueryIntent]]] = None):
        """Train the intent classifier."""
        if training_data is None:
            training_data = self.generate_training_data()

        # Prepare data
        texts = [item[0] for item in training_data]
        labels = [item[1].value for item in training_data]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train classifier
        self.classifier.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Intent Classifier Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.is_trained = True
        self._save_model()

    def predict(self, query: str) -> IntentPrediction:
        """Predict the intent of a query."""
        if not self.is_trained:
            # Return default for untrained model
            return IntentPrediction(
                intent=QueryIntent.GENERAL,
                confidence=0.5,
                probabilities={intent: 0.14 for intent in QueryIntent},
            )

        # Vectorize query
        query_vec = self.vectorizer.transform([query])

        # Get prediction and probabilities
        predicted_label = self.classifier.predict(query_vec)[0]
        probabilities = self.classifier.predict_proba(query_vec)[0]

        # Map probabilities to intents
        intent_probs = {}
        for i, intent in enumerate(QueryIntent):
            if intent.value in self.classifier.classes_:
                class_idx = list(self.classifier.classes_).index(intent.value)
                intent_probs[intent] = probabilities[class_idx]
            else:
                intent_probs[intent] = 0.0

        predicted_intent = QueryIntent(predicted_label)
        confidence = intent_probs[predicted_intent]

        return IntentPrediction(
            intent=predicted_intent, confidence=confidence, probabilities=intent_probs
        )

    def _save_model(self):
        """Save the trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "is_trained": self.is_trained,
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {self.model_path}")

    def _load_model(self):
        """Load a trained model from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    model_data = pickle.load(f)

                self.vectorizer = model_data["vectorizer"]
                self.classifier = model_data["classifier"]
                self.is_trained = model_data["is_trained"]

                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.is_trained = False


def train_intent_classifier():
    """Utility function to train and save the intent classifier."""
    classifier = IntentClassifier()
    classifier.train()
    return classifier


if __name__ == "__main__":
    # Train the classifier
    classifier = train_intent_classifier()

    # Test some queries
    test_queries = [
        "expense authorization workflow",
        "customer entity structure",
        "purchase order entry page",
        "invoice api integration",
        "employee status report",
        "error handling",
    ]

    print("\n" + "=" * 60)
    print("TESTING INTENT CLASSIFIER")
    print("=" * 60)

    for query in test_queries:
        prediction = classifier.predict(query)
        print(f"\nQuery: '{query}'")
        print(
            f"Intent: {prediction.intent.value} (confidence: {prediction.confidence:.3f})"
        )

        # Show top 3 probabilities
        top_intents = sorted(
            prediction.probabilities.items(), key=lambda x: x[1], reverse=True
        )[:3]
        print("Top predictions:")
        for intent, prob in top_intents:
            print(f"  {intent.value}: {prob:.3f}")
