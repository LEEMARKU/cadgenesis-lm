import sys
sys.path.insert(0, r'D:\Gen-AI CAD_LLM\src')

with open('src/cadgenesis/reasoning/knowledge_graph.py', 'r') as f:
    content = f.read()

# Find the remove_node method and add requirement features after it
insert_marker = '    def remove_node'
insert_pos = content.find(insert_marker)
if insert_pos >= 0:
    # Find the end of remove_node method (next double newline or end of class)
    section = content[insert_pos:]
    # Find two consecutive newlines after insert_pos
    double_newline = section.find('\n\n')
    if double_newline >= 0:
        end_pos = insert_pos + double_newline + 2
        new_methods = '''

    def add_requirement(
        self,
        req_id: str,
        label: str,
        description: str = "",
        related_features: list[str] | None = None,
        related_ops: list[str] | None = None,
    ) -> GraphNode:
        """Add a requirement node to the graph.

        Parameters
        ----------
        req_id : str
            Unique requirement identifier (e.g., "REQ-001").
        label : str
            Human-readable requirement name.
        description : str
            Detailed requirement description.
        related_features : list of str
            Feature IDs related to this requirement (e.g., "FEAT_HOLE", "PRIM_CYLINDER").
        related_ops : list of str
            Operation IDs related to this requirement (e.g., "OP-001", "OP-002").
        """
        node = self.add_node(
            node_id=req_id,
            label=label,
            node_type="requirement",
            attributes={
                "description": description,
                "related_features": related_features or [],
                "related_ops": related_ops or [],
            },
        )
        return node

    def find_requirements_by_feature(self, feature_id: str) -> list[GraphNode]:
        """Find all requirement nodes related to a given feature ID."""
        matching = []
        for node in self._nodes.values():
            attrs = node.attributes
            if "related_features" in attrs and feature_id in attrs["related_features"]:
                matching.append(node)
        return matching

    def find_requirements_by_op(self, op_id: str) -> list[GraphNode]:
        """Find all requirement nodes related to a given operation ID."""
        matching = []
        for node in self._nodes.values():
            attrs = node.attributes
            if "related_ops" in attrs and op_id in attrs["related_ops"]:
                matching.append(node)
        return matching

    def requirement_traceability_path(
        self,
        from_req: str,
        to_feature: str | None = None,
        to_op: str | None = None,
    ) -> list[dict[str, Any]]:
        """Trace from a requirement to related features or operations.

        Returns a list of dicts with 'node', 'edge', 'path' information.
        """
        path = []
        if from_req not in self._nodes:
            return path

        visited = {from_req}
        current = from_req
        while True:
            node = self._nodes.get(current)
            if node is None:
                break

            # Check if this node connects to the target
            if to_feature and "related_features" in node.attributes:
                if to_feature in node.attributes["related_features"]:
                    path.append(
                        {
                            "node": current,
                            "label": node.label,
                            "attributes": dict(node.attributes),
                        }
                    )
                    break
            if to_op and "related_ops" in node.attributes:
                if to_op in node.attributes["related_ops"]:
                    path.append(
                        {
                            "node": current,
                            "label": node.label,
                            "attributes": dict(node.attributes),
                        }
                    )
                    break

            # Follow outgoing edges
            next_nodes = []
            for edge in self._out.get(current, []):
                if edge.target not in visited:
                    next_nodes.append(edge.target)
                    visited.add(edge.target)

            if not next_nodes:
                break

            # Pick the first unvisited node (simple traversal)
            current = next_nodes[0]
            path.append(
                {
                    "node": current,
                    "label": self._nodes[current].label,
                    "attributes": dict(self._nodes[current].attributes),
                    "edge": edge.relation,
                }
            )

        return path


__all__ = [
    "GraphNode",
    "GraphEdge",
    "KnowledgeGraph",
    "add_requirement",
    "find_requirements_by_feature",
    "find_requirements_by_op",
    "requirement_traceability_path",
]
'''

    new_content = content[:insert_pos] + new_methods + content[insert_pos:]
    with open('src/cadgenesis/reasoning/knowledge_graph.py', 'w') as f:
        f.write(new_content)
    print('Added requirement traceability to KG')
else:
    print('Marker not found')
PYEOF