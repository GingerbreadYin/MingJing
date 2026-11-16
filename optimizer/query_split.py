"""Conservative clause splitting and bounded, deterministic evidence merging."""
from itertools import zip_longest
from operator import add
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


QUESTION_MARKERS = re.compile(r'什么|哪些|何时|怎么|怎样|如何|能否|是否|哪里|多久|怎么办|多少')
TIMING_QUESTION = re.compile(r'(.+?)(?:何时|什么时候)([\u4e00-\u9fff]{2,4})$')


def split_query(question: str) -> list[str]:
    clauses = list(dict.fromkeys(part.strip() for part in
                                re.split(r'[，,；;？?]', question) if part.strip()))
    if not 2 <= len(clauses) <= 3 or not all(QUESTION_MARKERS.search(part) for part in clauses):
        return [question]
    queries = [clauses[0]]
    for previous, current in zip(clauses, clauses[1:]):
        timing = TIMING_QUESTION.fullmatch(previous)
        if timing:
            topic, action = timing.groups()
            if current.startswith(tuple(action + suffix for suffix in ('后', '前', '时', '之后', '之前'))):
                current = f'{topic} {current}'
        queries.append(current)
    return queries


class BranchInput(TypedDict):
    position: int
    query: str


class Branch(BranchInput):
    hits: list[dict]


class QueryState(TypedDict, total=False):
    queries: list[str]
    collected: Annotated[list[Branch], add]
    branches: list[Branch]
    hits: list[dict]


def build_query_graph(retrieve, *, top_k: int, context_chars: int):
    def dispatch(state: QueryState) -> list[Send]:
        return [Send('retrieve', {'position': position, 'query': query})
                for position, query in enumerate(state['queries'])]

    def search(branch: BranchInput):
        return {'collected': [branch | {'hits': retrieve(branch['query'])}]}

    def merge(state: QueryState):
        branches = sorted(state['collected'], key=lambda branch: branch['position'])
        hits, seen = [], set()
        remaining = context_chars
        for row in zip_longest(*(branch['hits'] for branch in branches)):
            for hit in row:
                if hit is None or hit['id'] in seen or len(hit['text']) > remaining:
                    continue
                hits.append(hit)
                seen.add(hit['id'])
                remaining -= len(hit['text'])
                if len(hits) == top_k:
                    return {'branches': branches, 'hits': hits}
        return {'branches': branches, 'hits': hits}

    graph = StateGraph(QueryState)
    graph.add_node('retrieve', search)
    graph.add_node('merge', merge)
    graph.add_conditional_edges(START, dispatch, ['retrieve'])
    graph.add_edge('retrieve', 'merge')
    graph.add_edge('merge', END)
    return graph.compile()
