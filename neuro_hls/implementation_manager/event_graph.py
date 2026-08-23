from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class EventEdge:
    source: str
    target: str
    feedback: bool = False


@dataclass
class EventGraph:
    nodes: tuple
    edges: tuple
    schedule: tuple
    recurrent_components: tuple

    def incoming(self, node):
        edges = tuple(edge for edge in self.edges if edge.target == node)
        return tuple(sorted(edges, key=lambda edge: edge.feedback))

    def outgoing(self, node):
        return tuple(edge for edge in self.edges if edge.source == node)

    @property
    def feedback_edges(self):
        return tuple(edge for edge in self.edges if edge.feedback)


def _reachable_order(nodes, edges, start):
    adjacency = defaultdict(list)
    for source, target in edges:
        adjacency[source].append(target)

    order = []
    discovered = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for target in adjacency[node]:
            if target not in discovered:
                discovered.add(target)
                queue.append(target)

    unreachable = tuple(node for node in nodes if node not in discovered)
    if unreachable:
        raise ValueError(f"event-driven graph contains nodes unreachable from 'input': {unreachable}")
    return tuple(order)


def _strongly_connected_components(nodes, edges):
    adjacency = defaultdict(list)
    for source, target in edges:
        adjacency[source].append(target)

    index = 0
    indices = {}
    lowlinks = {}
    stack = []
    on_stack = set()
    components = []

    def visit(node):
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in adjacency[node]:
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])

        if lowlinks[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(tuple(component))

    for node in nodes:
        if node not in indices:
            visit(node)
    return tuple(components)


def _find_feedback_edges(nodes, edges, traversal_order):
    adjacency = defaultdict(list)
    edge_set = set(edges)
    order_index = {node: index for index, node in enumerate(traversal_order)}
    for source, target in edges:
        adjacency[source].append(target)
    for source in adjacency:
        adjacency[source].sort(key=lambda node: order_index[node])

    colors = {node: 0 for node in nodes}
    feedback = set()

    def visit(node):
        colors[node] = 1
        for target in adjacency[node]:
            if colors[target] == 0:
                visit(target)
            elif colors[target] == 1:
                feedback.add((node, target))
        colors[node] = 2

    visit("input")
    if not feedback.issubset(edge_set):
        raise AssertionError("internal feedback-edge detection error")
    return feedback


def _topological_schedule(nodes, edges, feedback_edges, traversal_order):
    adjacency = defaultdict(list)
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        if (source, target) in feedback_edges:
            continue
        adjacency[source].append(target)
        indegree[target] += 1

    order_index = {node: index for index, node in enumerate(traversal_order)}
    ready = [node for node in nodes if indegree[node] == 0]
    ready.sort(key=lambda node: order_index[node])
    schedule = []
    while ready:
        node = ready.pop(0)
        schedule.append(node)
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=lambda item: order_index[item])

    if len(schedule) != len(nodes):
        remaining = tuple(node for node in nodes if node not in schedule)
        raise ValueError(f"unable to break all recurrent cycles: {remaining}")
    return tuple(schedule)


def build_event_graph(model):
    graph_layers = getattr(model, "graph_layers", None)
    graph_edges = getattr(model, "graph_edges", None)
    if not graph_layers:
        graph_layers = {
            layer.name: layer for layer in model.layers
            if type(layer).__name__ != "Merge"
        }
        graph_edges = []
        for layer in graph_layers.values():
            graph_edges.extend((dependency, layer.name) for dependency, _ in layer.dependencies)
        model.graph_layers = graph_layers
        model.graph_edges = graph_edges
    if graph_edges is None:
        raise ValueError("model does not contain a graph representation")

    nodes = tuple(graph_layers)
    edges = tuple((str(source), str(target)) for source, target in graph_edges)
    if "input" not in graph_layers or "output" not in graph_layers:
        raise ValueError("event-driven graph requires 'input' and 'output' nodes")

    traversal_order = _reachable_order(nodes, edges, "input")
    components = _strongly_connected_components(nodes, edges)
    recurrent_components = tuple(
        tuple(sorted(component, key=lambda node: traversal_order.index(node)))
        for component in components
        if len(component) > 1 or (component[0], component[0]) in edges
    )
    feedback_edges = _find_feedback_edges(nodes, edges, traversal_order)
    schedule = _topological_schedule(nodes, edges, feedback_edges, traversal_order)

    event_edges = tuple(
        EventEdge(source, target, (source, target) in feedback_edges)
        for source, target in edges
    )
    return EventGraph(nodes, event_edges, schedule, recurrent_components)
