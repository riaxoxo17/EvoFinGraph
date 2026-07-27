"""
EvoFinGraph API - Integration version
Investigation Platform Engineer (ECE) deliverable.

Replaces Day 3's mock_data lookups with real Cypher queries against
the populated Neo4j graph (Transaction, Community, MEMBER_OF, FLOWS_TO).
"""

from fastapi import FastAPI, HTTPException

from db import get_session

app = FastAPI(title="EvoFinGraph API", version="0.2.0")


@app.get("/")
def root():
    return {"message": "EvoFinGraph API is running"}


@app.get("/community/{pcid}")
def get_community(pcid: int):
    """
    Returns the most recent snapshot of a given community (highest timeStep).
    """
    query = """
    MATCH (c:Community {pcid: $pcid})
    RETURN c ORDER BY c.timeStep DESC LIMIT 1
    """
    with get_session() as session:
        result = session.run(query, pcid=pcid)
        record = result.single()

    if record is None:
        raise HTTPException(status_code=404, detail=f"Community {pcid} not found")

    return dict(record["c"])


@app.get("/timeline/{pcid}")
def get_timeline(pcid: int):
    """
    Returns every snapshot of this community, ordered chronologically.
    """
    query = """
    MATCH (c:Community {pcid: $pcid})
    RETURN c ORDER BY c.timeStep ASC
    """
    with get_session() as session:
        result = session.run(query, pcid=pcid)
        records = [dict(r["c"]) for r in result]

    if not records:
        raise HTTPException(status_code=404, detail=f"Timeline for {pcid} not found")

    return {"pcid": pcid, "timeline": records}


@app.get("/alerts")
def get_alerts(limit: int = 10):
    """
    Returns the highest-risk (ground-truth fraud) communities, ranked by AFEI.
    """
    query = """
    MATCH (c:Community)
    WHERE c.isFraudCommunity = 1
    RETURN c ORDER BY c.afei DESC LIMIT $limit
    """
    with get_session() as session:
        result = session.run(query, limit=limit)
        alerts = [dict(r["c"]) for r in result]

    return {"alerts": alerts}


@app.get("/graph/{pcid}")
def get_graph(pcid: int):
    """
    Returns members and internal edges of a community's latest snapshot.
    """
    latest_query = """
    MATCH (c:Community {pcid: $pcid})
    RETURN c.timeStep AS timeStep ORDER BY c.timeStep DESC LIMIT 1
    """
    with get_session() as session:
        latest = session.run(latest_query, pcid=pcid).single()

        if latest is None:
            raise HTTPException(status_code=404, detail=f"Graph for {pcid} not found")

        time_step = latest["timeStep"]

        members_query = """
        MATCH (t:Transaction)-[:MEMBER_OF]->(c:Community {pcid: $pcid, timeStep: $timeStep})
        RETURN t.txId AS txId, t.label AS label
        """
        members = session.run(members_query, pcid=pcid, timeStep=time_step)
        nodes = [{"txId": r["txId"], "label": r["label"]} for r in members]

        node_ids = [n["txId"] for n in nodes]

        edges_query = """
        MATCH (a:Transaction)-[:FLOWS_TO]->(b:Transaction)
        WHERE a.txId IN $node_ids AND b.txId IN $node_ids
        RETURN a.txId AS source, b.txId AS target
        """
        edges_result = session.run(edges_query, node_ids=node_ids)
        edges = [{"source": r["source"], "target": r["target"]} for r in edges_result]

    return {"pcid": pcid, "nodes": nodes, "edges": edges}