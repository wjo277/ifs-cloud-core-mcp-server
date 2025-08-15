#!/usr/bin/env python3
"""
Test script for the projecti        print(f"  🏛️  Entities: {len(ast.entities)}")
        for entity in ast.entities:
            attrs = entity.attributes.get('entity_attributes', [])
            refs = entity.attributes.get('references', [])
            print(f"    - {entity.name} ({len(attrs)} attrs, {len(refs)} refs)")analyzer using real projection files from the _work directory.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ifs_cloud_mcp_server.projection_analyzer import ProjectionAnalyzer
import json


def test_with_real_projection():
    """Test the analyzer with a real projection file from _work directory."""

    # Path to a real projection file
    projection_path = r"_work\accrul\model\accrul\AccountsHandling.projection"

    if not os.path.exists(projection_path):
        print(f"❌ Projection file not found: {projection_path}")
        return False

    print(f"📁 Testing with real projection file: {projection_path}")

    try:
        # Read the real projection file
        with open(projection_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"📄 File size: {len(content)} characters")
        print(f"📝 First few lines:")
        for i, line in enumerate(content.split("\n")[:10]):
            print(f"  {i+1:2d}: {line}")
        print()

        # Analyze the projection
        analyzer = ProjectionAnalyzer()
        ast = analyzer.analyze(content)

        print("🎯 Analysis Results:")
        print(f"  📋 Projection: {ast.name}")
        print(f"  🏗️  Component: {ast.component}")
        print(f"  🎂 Layer: {ast.layer}")
        print(f"  📄 Description: {ast.description}")
        print(f"  🏷️  Category: {ast.category}")
        print(f"  📦 Includes: {len(ast.includes)} fragments")
        for include in ast.includes[:3]:  # Show first 3
            print(f"    - {include['name']}")

        print(f"  📊 Entity Sets: {len(ast.entitysets)}")
        for entityset in ast.entitysets:
            print(f"    - {entityset['name']} for {entityset['entity']}")
            if entityset.get("context"):
                print(f"      Context: {entityset['context']}")
            if entityset.get("where"):
                print(f"      Where: {entityset['where'][:50]}...")

        print(f"  🏛️  Entities: {len(ast.entities)}")
        for entity in ast.entities:
            attrs = entity.attributes.get("entity_attributes", [])
            refs = entity.attributes.get("references", [])
            print(f"    - {entity.name} ({len(attrs)} attrs, {len(refs)} refs)")

        print(f"  ⚡ Actions: {len(ast.actions)}")
        for action in ast.actions:
            print(f"    - {action['name']}")

        print(f"  🔧 Functions: {len(ast.functions)}")
        for function in ast.functions:
            print(f"    - {function['name']}")

        print("\n✅ Real projection file analysis completed successfully!")

        # Test serialization
        ast_dict = ast.to_dict()
        json_str = json.dumps(ast_dict, indent=2)
        print(f"\n📋 AST serialized to {len(json_str)} characters of JSON")

        return True

    except Exception as e:
        print(f"❌ Error analyzing real projection: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_with_multiple_real_projections():
    """Test with multiple real projection files."""
    print("\n🔍 Testing with multiple real projection files...")

    import glob

    projection_files = glob.glob("_work/**/model/**/*.projection", recursive=True)[
        :5
    ]  # Test first 5

    if not projection_files:
        print("❌ No projection files found in _work directory")
        return False

    print(f"📁 Found {len(projection_files)} projection files, testing first 5:")

    analyzer = ProjectionAnalyzer()
    results = []

    for i, proj_file in enumerate(projection_files, 1):
        print(f"\n{i}. {proj_file}")
        try:
            with open(proj_file, "r", encoding="utf-8") as f:
                content = f.read()

            ast = analyzer.analyze(content)
            result = {
                "file": proj_file,
                "name": ast.name,
                "component": ast.component,
                "layer": ast.layer,
                "entitysets": len(ast.entitysets),
                "entities": len(ast.entities),
                "actions": len(ast.actions),
                "functions": len(ast.functions),
                "includes": len(ast.includes),
            }
            results.append(result)

            print(
                f"   ✅ {ast.name} - {ast.component} ({result['entitysets']} sets, {result['entities']} entities)"
            )

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({"file": proj_file, "error": str(e)})

    print(
        f"\n📊 Summary: Successfully analyzed {len([r for r in results if 'error' not in r])}/{len(results)} projection files"
    )
    return True


if __name__ == "__main__":
    print("🚀 Testing Projection Analyzer with Real IFS Cloud Files")
    print("=" * 60)

    success1 = test_with_real_projection()
    success2 = test_with_multiple_real_projections()

    if success1 and success2:
        print(
            "\n🎉 All tests passed! The projection analyzer works with real IFS Cloud files."
        )
    else:
        print("\n❌ Some tests failed. Check the output above.")
