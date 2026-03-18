#!/usr/bin/env python3
"""
Import manually-defined world graph data into Neo4j.

JSON schema:
{
  "nodes": [
    {
      "id": "char_jokebear",
      "name": "JokeBear",
      "labels": ["Character"],
      "props": {"alias": "自嘲熊"}
    }
  ],
  "relations": [
    {
      "from": "char_jokebear",
      "to": "char_pug",
      "type": "BEST_FRIEND_WITH",
      "props": {"confidence": 0.95}
    }
  ]
}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Set, Tuple

from neo4j import GraphDatabase


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_safe_identifier(value: str, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} must be a non-empty string")
    if not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"Invalid {kind}: {value!r}. Only [A-Za-z_][A-Za-z0-9_]* is allowed."
        )
    return value


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Root JSON must be an object")
    return data


def _validate_graph(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes = data.get("nodes", [])
    relations = data.get("relations", [])

    if not isinstance(nodes, list):
        raise ValueError("'nodes' must be a list")
    if not isinstance(relations, list):
        raise ValueError("'relations' must be a list")

    node_ids: Set[str] = set()

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{i}] must be an object")

        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(f"nodes[{i}].id must be a non-empty string")
        if node_id in node_ids:
            raise ValueError(f"Duplicate node id: {node_id}")
        node_ids.add(node_id)

        name = node.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"nodes[{i}].name must be a non-empty string")

        labels = node.get("labels", [])
        if labels is None:
            labels = []
        if not isinstance(labels, list):
            raise ValueError(f"nodes[{i}].labels must be a list")
        for label in labels:
            _require_safe_identifier(label, f"nodes[{i}].labels[]")

        props = node.get("props", {})
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise ValueError(f"nodes[{i}].props must be an object")

    for i, rel in enumerate(relations):
        if not isinstance(rel, dict):
            raise ValueError(f"relations[{i}] must be an object")

        from_id = rel.get("from")
        to_id = rel.get("to")
        rel_type = rel.get("type")

        if not isinstance(from_id, str) or from_id not in node_ids:
            raise ValueError(f"relations[{i}].from must reference an existing node id")
        if not isinstance(to_id, str) or to_id not in node_ids:
            raise ValueError(f"relations[{i}].to must reference an existing node id")
        _require_safe_identifier(rel_type, f"relations[{i}].type")

        props = rel.get("props", {})
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise ValueError(f"relations[{i}].props must be an object")

    return nodes, relations


def _merge_node(session: Any, node: Dict[str, Any]) -> None:
    node_id = node["id"]
    name = node["name"]
    labels = ["Entity", "WorldEntity"] + list(node.get("labels", []))
    # Preserve order while removing duplicates.
    labels = list(dict.fromkeys(labels))
    safe_labels = [_require_safe_identifier(label, "label") for label in labels]
    labels_clause = ":" + ":".join(safe_labels)

    props = dict(node.get("props", {}))
    # Guard reserved keys.
    props.pop("entity_id", None)
    props.pop("name", None)

    query = f"""
        MERGE (n{labels_clause} {{entity_id: $entity_id}})
        SET n.name = $name,
            n.source = 'manual',
            n.updated_at = datetime()
        SET n += $props
    """
    session.run(query, entity_id=node_id, name=name, props=props)


def _merge_relation(session: Any, rel: Dict[str, Any]) -> None:
    rel_type = _require_safe_identifier(rel["type"], "relation type")
    props = dict(rel.get("props", {}))
    props.pop("source", None)
    props.pop("updated_at", None)

    query = f"""
        MATCH (a:Entity {{entity_id: $from_id}})
        MATCH (b:Entity {{entity_id: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.source = 'manual',
            r.updated_at = datetime()
        SET r += $props
    """
    session.run(
        query,
        from_id=rel["from"],
        to_id=rel["to"],
        props=props,
    )


def _print_preview(nodes: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> None:
    print(f"[Preview] nodes={len(nodes)} relations={len(relations)}")
    print("[Preview] first 5 nodes:")
    for node in nodes[:5]:
        labels = ["Entity", "WorldEntity"] + list(node.get("labels", []))
        labels = list(dict.fromkeys(labels))
        print(
            f"  - id={node['id']} name={node['name']} labels={labels} props={node.get('props', {})}"
        )

    print("[Preview] first 5 relations:")
    for rel in relations[:5]:
        print(
            f"  - ({rel['from']})-[:{rel['type']}]->({rel['to']}) props={rel.get('props', {})}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Import manual world graph into Neo4j")
    parser.add_argument(
        "--file",
        default="data_get/world_graph_manual.json",
        help="Path to manual graph JSON file",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (default: env NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: env NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", "jokebear2024"),
        help="Neo4j password (default: env NEO4J_PASSWORD or jokebear2024)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview data without writing to Neo4j",
    )
    args = parser.parse_args()

    try:
        data = _load_json(args.file)
        nodes, relations = _validate_graph(data)
    except Exception as exc:
        print(f"[Error] invalid input file: {exc}")
        return 1

    _print_preview(nodes, relations)
    if args.dry_run:
        print("[Dry-run] no data written.")
        return 0

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            # Keep this index/constraint safe for repeated imports.
            session.run(
                """
                CREATE CONSTRAINT entity_id_constraint IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE
                """
            )

            for node in nodes:
                _merge_node(session, node)
            for rel in relations:
                _merge_relation(session, rel)
    finally:
        driver.close()

    print("[Done] manual world graph imported successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
