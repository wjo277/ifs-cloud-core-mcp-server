"""
Example: Using Projection AST for Copilot Intelligence

This module demonstrates how the projection AST can be used to provide
intelligent code assistance, completion, and analysis for IFS Cloud development.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from .projection_analyzer import (
        ProjectionAnalyzer,
        ProjectionAST,
        ASTNode,
        NodeType,
    )

    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False


@dataclass
class CodeCompletion:
    """Represents a code completion suggestion"""

    text: str
    kind: str  # 'entity', 'attribute', 'action', 'reference', etc.
    detail: str
    documentation: str
    insert_text: Optional[str] = None


@dataclass
class CodeHover:
    """Represents hover information for code elements"""

    title: str
    content: str
    signature: Optional[str] = None


class ProjectionIntelliSense:
    """Provides IntelliSense capabilities using projection AST"""

    def __init__(self, ast: "ProjectionAST"):
        self.ast = ast
        self.analyzer = ProjectionAnalyzer()

    def get_completions_for_entity(
        self, entity_name: str, context: str = ""
    ) -> List[CodeCompletion]:
        """Get code completions for entity members"""
        completions = []

        # Find the entity in the AST
        entity = self._find_entity(entity_name)
        if not entity:
            return completions

        # Add attribute completions
        for child in entity.children:
            if child.node_type == NodeType.ATTRIBUTE:
                completions.append(
                    CodeCompletion(
                        text=child.name,
                        kind="attribute",
                        detail=child.attributes.get("data_type", "unknown"),
                        documentation=f"Attribute: {child.name}",
                        insert_text=child.name,
                    )
                )

            elif child.node_type == NodeType.REFERENCE:
                completions.append(
                    CodeCompletion(
                        text=child.name,
                        kind="reference",
                        detail="Reference",
                        documentation=f"Reference: {child.name}",
                        insert_text=child.name,
                    )
                )

            elif child.node_type == NodeType.ACTION:
                # Get parameters for the action
                params = []
                for param_child in child.children:
                    if param_child.node_type == NodeType.ATTRIBUTE:
                        param_type = param_child.attributes.get("data_type", "unknown")
                        params.append(f"{param_child.name}: {param_type}")

                param_signature = f"({', '.join(params)})"
                completions.append(
                    CodeCompletion(
                        text=child.name,
                        kind="action",
                        detail=f"Action{param_signature}",
                        documentation=f"Action: {child.name}",
                        insert_text=f"{child.name}({', '.join([p.split(':')[0] for p in params])})",
                    )
                )

        return completions

    def get_hover_info(
        self, element_name: str, context: str = ""
    ) -> Optional[CodeHover]:
        """Get hover information for a code element"""
        # Search all nodes for the element
        for node in self.ast.all_nodes:
            if node.name == element_name:
                return self._create_hover_for_node(node)

        return None

    def find_definition(self, element_name: str) -> Optional[Dict[str, Any]]:
        """Find the definition of an element"""
        for node in self.ast.all_nodes:
            if node.name == element_name:
                return {
                    "name": node.name,
                    "type": node.node_type.value,
                    "line": node.line_start,
                    "attributes": node.attributes,
                    "parent": node.parent.name if node.parent else None,
                }

        return None

    def get_references(self, element_name: str) -> List[Dict[str, Any]]:
        """Find all references to an element"""
        references = self.analyzer.find_references_to(self.ast, element_name)
        return [
            {
                "name": ref.name,
                "type": ref.node_type.value,
                "line": ref.line_start,
                "context": ref.raw_content,
            }
            for ref in references
        ]

    def validate_entity_usage(
        self, entity_name: str, attribute_name: str
    ) -> Dict[str, Any]:
        """Validate if an attribute exists on an entity"""
        entity = self._find_entity(entity_name)
        if not entity:
            return {
                "valid": False,
                "error": f"Entity '{entity_name}' not found",
                "suggestions": [e.name for e in self.ast.entities],
            }

        # Check if attribute exists
        for child in entity.children:
            if child.node_type == NodeType.ATTRIBUTE and child.name == attribute_name:
                return {
                    "valid": True,
                    "type": child.attributes.get("data_type", "unknown"),
                    "properties": child.attributes,
                }

        return {
            "valid": False,
            "error": f"Attribute '{attribute_name}' not found on entity '{entity_name}'",
            "suggestions": [
                child.name
                for child in entity.children
                if child.node_type == NodeType.ATTRIBUTE
            ],
        }

    def suggest_action_parameters(
        self, entity_name: str, action_name: str
    ) -> List[Dict[str, str]]:
        """Get parameter suggestions for an action"""
        entity = self._find_entity(entity_name)
        if not entity:
            return []

        # Find the action
        for child in entity.children:
            if child.node_type == NodeType.ACTION and child.name == action_name:
                parameters = []
                for param in child.children:
                    if param.node_type == NodeType.ATTRIBUTE:
                        parameters.append(
                            {
                                "name": param.name,
                                "type": param.attributes.get("data_type", "unknown"),
                                "required": param.attributes.get("required", False),
                            }
                        )
                return parameters

        return []

    def get_entity_schema(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get complete schema information for an entity"""
        entity = self._find_entity(entity_name)
        if not entity:
            return None

        schema = {
            "name": entity.name,
            "type": "entity",
            "attributes": {},
            "references": {},
            "arrays": {},
            "actions": {},
            "functions": {},
        }

        for child in entity.children:
            if child.node_type == NodeType.ATTRIBUTE:
                schema["attributes"][child.name] = {
                    "type": child.attributes.get("data_type", "unknown"),
                    "properties": child.attributes,
                }
            elif child.node_type == NodeType.REFERENCE:
                schema["references"][child.name] = {
                    "target": child.attributes.get("target_entity", "unknown"),
                    "properties": child.attributes,
                }
            elif child.node_type == NodeType.ARRAY:
                schema["arrays"][child.name] = {
                    "target": child.attributes.get("target_entity", "unknown"),
                    "properties": child.attributes,
                }
            elif child.node_type == NodeType.ACTION:
                params = [
                    {"name": p.name, "type": p.attributes.get("data_type", "unknown")}
                    for p in child.children
                    if p.node_type == NodeType.ATTRIBUTE
                ]
                schema["actions"][child.name] = {
                    "parameters": params,
                    "properties": child.attributes,
                }
            elif child.node_type == NodeType.FUNCTION:
                params = [
                    {"name": p.name, "type": p.attributes.get("data_type", "unknown")}
                    for p in child.children
                    if p.node_type == NodeType.ATTRIBUTE
                ]
                schema["functions"][child.name] = {
                    "parameters": params,
                    "return_type": child.attributes.get("return_type", "unknown"),
                    "properties": child.attributes,
                }

        return schema

    def _find_entity(self, entity_name: str) -> Optional["ASTNode"]:
        """Find an entity node by name"""
        for entity in self.ast.entities:
            if entity.name == entity_name:
                return entity
        return None

    def _create_hover_for_node(self, node: "ASTNode") -> CodeHover:
        """Create hover information for a node"""
        content_parts = [f"**{node.node_type.value.title()}**: {node.name}"]

        if node.attributes:
            content_parts.append("\n**Properties:**")
            for key, value in node.attributes.items():
                content_parts.append(f"- {key}: {value}")

        if node.children:
            content_parts.append(f"\n**Children**: {len(node.children)} items")

        signature = None
        if node.node_type in [NodeType.ACTION, NodeType.FUNCTION]:
            params = [
                f"{child.name}: {child.attributes.get('data_type', 'unknown')}"
                for child in node.children
                if child.node_type == NodeType.ATTRIBUTE
            ]
            signature = f"{node.name}({', '.join(params)})"

        return CodeHover(
            title=f"{node.node_type.value.title()}: {node.name}",
            content="\n".join(content_parts),
            signature=signature,
        )


def demonstrate_copilot_usage():
    """Demonstrate how Copilot can use the AST for intelligent assistance"""
    if not AST_AVAILABLE:
        print("Projection analyzer not available")
        return

    # Sample projection for demonstration
    sample_projection = """
    projection CustomerOrderHandling;
    
    entity CustomerOrderHeader {
        attribute OrderNo Text;
        attribute CustomerNo Text;
        attribute OrderDate Date;
        
        reference CustomerInfo(CustomerNo) to Customer(CustomerNo);
        array OrderLines(OrderNo) to CustomerOrderLine(OrderNo);
        
        action ReleaseOrder {
            parameter ReleaseDate Date;
        }
    }
    """

    # Analyze the projection
    analyzer = ProjectionAnalyzer()
    ast = analyzer.analyze(sample_projection)

    # Create IntelliSense provider
    intellisense = ProjectionIntelliSense(ast)

    # Example 1: Get completions for entity
    print("🎯 Code Completions for CustomerOrderHeader:")
    completions = intellisense.get_completions_for_entity("CustomerOrderHeader")
    for comp in completions:
        print(f"  • {comp.text} ({comp.kind}) - {comp.detail}")

    # Example 2: Get hover info
    print("\n🔍 Hover Info for 'ReleaseOrder':")
    hover = intellisense.get_hover_info("ReleaseOrder")
    if hover:
        print(f"  Title: {hover.title}")
        print(f"  Signature: {hover.signature}")
        print(f"  Content: {hover.content}")

    # Example 3: Validate usage
    print("\n✅ Validate Entity Usage:")
    validation = intellisense.validate_entity_usage("CustomerOrderHeader", "OrderNo")
    print(f"  Valid: {validation['valid']}")
    print(f"  Type: {validation.get('type', 'N/A')}")

    # Example 4: Get entity schema
    print("\n📋 Entity Schema:")
    schema = intellisense.get_entity_schema("CustomerOrderHeader")
    if schema:
        print(f"  Attributes: {list(schema['attributes'].keys())}")
        print(f"  References: {list(schema['references'].keys())}")
        print(f"  Actions: {list(schema['actions'].keys())}")


if __name__ == "__main__":
    demonstrate_copilot_usage()
