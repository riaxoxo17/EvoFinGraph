"""
Neo4j connection module.

Centralizes driver setup so main.py doesn't manage connections directly.
Replace the password below with your actual Neo4j password.
"""

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "evofin123"  # <-- replace with your real password

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def get_session():
    return driver.session()