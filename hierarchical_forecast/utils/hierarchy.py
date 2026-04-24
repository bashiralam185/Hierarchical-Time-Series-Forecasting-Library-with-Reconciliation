"""
HierarchyTree: Core data structure for defining and managing hierarchical time series.

A hierarchy is expressed as a summing matrix S (n x m) where:
  - n = total number of series (all levels)
  - m = number of bottom-level series
  - S[i, j] = 1 if bottom series j contributes to aggregate series i

Example retail hierarchy:
    Total Sales
    ├── Region North
    │   ├── Store A
    │   └── Store B
    └── Region South
        ├── Store C
        └── Store D
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class HierarchyNode:
    """A single node in the hierarchy tree."""
    name: str
    level: int
    children: List["HierarchyNode"] = field(default_factory=list)
    parent: Optional["HierarchyNode"] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        return f"HierarchyNode(name={self.name!r}, level={self.level}, children={len(self.children)})"


class HierarchyTree:
    """
    Defines and manages a hierarchical time series structure.

    Parameters
    ----------
    spec : dict
        Nested dict defining the hierarchy. Keys are node names, values are
        either nested dicts (internal nodes) or None (leaf nodes).

    Example
    -------
    >>> spec = {
    ...     "Total": {
    ...         "North": {"StoreA": None, "StoreB": None},
    ...         "South": {"StoreC": None, "StoreD": None},
    ...     }
    ... }
    >>> tree = HierarchyTree(spec)
    >>> print(tree.n_bottom)  # 4
    >>> print(tree.n_total)   # 7
    """

    def __init__(self, spec: dict):
        self.spec = spec
        self._nodes: Dict[str, HierarchyNode] = {}
        self._bottom_nodes: List[str] = []
        self._all_nodes: List[str] = []
        self._root: Optional[HierarchyNode] = None
        self._build_tree(spec)

    def _build_tree(self, spec: dict, parent: Optional[HierarchyNode] = None, level: int = 0):
        """Recursively build the tree from the spec dict."""
        for name, children in spec.items():
            node = HierarchyNode(name=name, level=level, parent=parent)
            self._nodes[name] = node
            self._all_nodes.append(name)

            if parent is None:
                self._root = node
            else:
                parent.children.append(node)

            if children is None:
                self._bottom_nodes.append(name)
            else:
                self._build_tree(children, parent=node, level=level + 1)

    @property
    def root(self) -> HierarchyNode:
        assert self._root is not None, "Tree not initialized."
        return self._root

    @property
    def n_bottom(self) -> int:
        """Number of bottom-level (leaf) series."""
        return len(self._bottom_nodes)

    @property
    def n_total(self) -> int:
        """Total number of series across all levels."""
        return len(self._all_nodes)

    @property
    def bottom_series(self) -> List[str]:
        return self._bottom_nodes.copy()

    @property
    def all_series(self) -> List[str]:
        return self._all_nodes.copy()

    def get_levels(self) -> Dict[int, List[str]]:
        """Return series grouped by their level in the hierarchy."""
        levels: Dict[int, List[str]] = {}
        for name, node in self._nodes.items():
            levels.setdefault(node.level, []).append(name)
        return dict(sorted(levels.items()))

    def get_summing_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Construct the summing matrix S (n x m).

        Returns
        -------
        S : np.ndarray of shape (n_total, n_bottom)
        all_series : list of all series names (row order)
        bottom_series : list of bottom series names (column order)
        """
        n = self.n_total
        m = self.n_bottom
        bottom_idx = {name: i for i, name in enumerate(self._bottom_nodes)}

        # Collect bottom-level descendants for each node
        S = np.zeros((n, m), dtype=np.float32)

        for i, name in enumerate(self._all_nodes):
            descendants = self._get_leaf_descendants(self._nodes[name])
            for d in descendants:
                S[i, bottom_idx[d]] = 1.0

        return S, self._all_nodes.copy(), self._bottom_nodes.copy()

    def _get_leaf_descendants(self, node: HierarchyNode) -> List[str]:
        """Recursively collect all leaf-node names under a given node."""
        if node.is_leaf():
            return [node.name]
        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaf_descendants(child))
        return leaves

    def validate_dataframe(self, df: pd.DataFrame, time_col: str = "ds", value_col: str = "y"):
        """
        Validate that a long-format DataFrame contains all expected series.

        Parameters
        ----------
        df : pd.DataFrame with columns [unique_id, ds, y]
        """
        if "unique_id" not in df.columns:
            raise ValueError("DataFrame must have a 'unique_id' column.")
        present = set(df["unique_id"].unique())
        expected = set(self._all_nodes)
        missing = expected - present
        if missing:
            raise ValueError(
                f"DataFrame is missing {len(missing)} series: {sorted(missing)[:5]}..."
            )

    def print_tree(self, node: Optional[HierarchyNode] = None, indent: int = 0):
        """Pretty-print the hierarchy tree."""
        if node is None:
            node = self.root
        prefix = "  " * indent + ("└── " if indent > 0 else "")
        print(f"{prefix}{node.name}")
        for child in node.children:
            self.print_tree(child, indent + 1)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, level_cols: List[str]) -> "HierarchyTree":
        """
        Build a HierarchyTree from a DataFrame with level columns.

        Parameters
        ----------
        df : pd.DataFrame
            Each row is a bottom-level series; level_cols define the path.
        level_cols : list of str
            Columns from outermost to innermost level, e.g. ['region', 'store'].

        Example
        -------
        >>> df = pd.DataFrame({
        ...     'region': ['North', 'North', 'South'],
        ...     'store':  ['A',     'B',     'C'],
        ... })
        >>> tree = HierarchyTree.from_dataframe(df, level_cols=['region', 'store'])
        """
        spec: dict = {"Total": {}}

        def insert(d: dict, path: List[str]):
            if not path:
                return
            key = path[0]
            if key not in d:
                d[key] = {} if len(path) > 1 else None
            if len(path) > 1 and d[key] is None:
                d[key] = {}
            if len(path) > 1:
                insert(d[key], path[1:])

        for _, row in df[level_cols].drop_duplicates().iterrows():
            path = list(row.values)
            insert(spec["Total"], path)

        return cls(spec)

    def __repr__(self) -> str:
        return (
            f"HierarchyTree(n_total={self.n_total}, n_bottom={self.n_bottom}, "
            f"levels={len(self.get_levels())})"
        )
