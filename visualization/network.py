"""
네트워크 그래프 시각화 (키워드 ↔ 종목)
"""
from collections import defaultdict

import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from mrn_project.config import GRAPH_OUTPUT_DIR


# 네트워크 그래프 구축

def build_network_graph(df, row1: str, row2: str) -> nx.Graph:
    """두 리스트형 열의 공동 출현 관계로 네트워크 그래프를 구축한다."""
    G = nx.Graph()
    for _, row in df.iterrows():
        node1_list = row[row1]
        node2_list = row[row2]
        for n1 in node1_list:
            for n2 in node2_list:
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)
    return G


# 유틸 함수

def get_top_n_edges(G: nx.Graph, center_node: str, n: int = 10) -> list:
    """중심 노드의 이웃 중 degree 상위 n 개를 반환한다."""
    edges = [(v, G.degree(v)) for _, v in G.edges(center_node)]
    sorted_edges = sorted(edges, key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_edges[:n]]


def normalize_sizes(values: dict, center_node: str,
                    min_size: int = 1000, max_size: int = 4000, center_size: int = 4000) -> dict:
    """노드 크기를 정규화한다."""
    others = {k: v for k, v in values.items() if k != center_node}
    if not others:
        return {center_node: center_size}

    min_val = min(others.values())
    max_val = max(others.values())

    if min_val == max_val:
        normalized = {node: (max_size + min_size) // 2 for node in others}
    else:
        normalized = {
            node: min_size + (max_size - min_size) * (val - min_val) / (max_val - min_val)
            for node, val in others.items()
        }

    normalized[center_node] = center_size
    return normalized


# 단일 네트워크 시각화

def visualize_single_network(G: nx.Graph, center_node: str, sentiment_scores: dict):
    """중심 키워드와 연결된 상위 종목 네트워크를 시각화한다."""
    if center_node not in G.nodes():
        print(f"'{center_node}' not found in the network")
        return

    top_edges = get_top_n_edges(G, center_node)
    subgraph_nodes = set(top_edges)
    subgraph_nodes.add(center_node)
    subgraph = G.subgraph(subgraph_nodes)

    degree_dict = dict(subgraph.degree(weight="weight"))
    node_sizes = normalize_sizes(degree_dict, center_node)

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    node_colors = {}
    for node in subgraph.nodes():
        score = sentiment_scores.get(node, 0)
        normalized_score = (score - (-1)) / (1 - (-1))
        node_colors[node] = cmap(normalized_score)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subgraph, seed=42, k=0.8)

    nx.draw(
        subgraph, pos,
        with_labels=True,
        node_size=[node_sizes[n] for n in subgraph.nodes()],
        font_size=8,
        edge_color="gray",
        width=1.0,
        node_color=[node_colors[n] for n in subgraph.nodes()],
        alpha=0.8,
    )

    nx.draw_networkx_nodes(
        subgraph, pos,
        nodelist=[center_node],
        node_size=node_sizes[center_node],
        node_color=[node_colors[center_node]],
    )

    plt.title(f"'{center_node}' network graph")
    plt.savefig(f"{GRAPH_OUTPUT_DIR}/{center_node}_network_graph.png")
    plt.show()
