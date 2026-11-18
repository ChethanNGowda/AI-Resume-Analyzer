# SQL & Database Mastery Guide

## Core Concepts & Patterns
1. **Relational Database Design**: First, second, third normal form (1NF, 2NF, 3NF), primary keys, foreign keys, constraints (CHECK, UNIQUE), and indexing strategies (B-Tree, Hash, GIN/GiST).
2. **Advanced SQL Techniques**:
   - Window Functions: `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`, `LEAD()`, `LAG()`, `SUM() OVER ()`.
   - Common Table Expressions (CTEs) and Recursive CTEs for hierarchical and tree data.
   - Aggregate Queries: `GROUP BY`, `HAVING`, `ROLLUP`, `CUBE`.
   - Transactions: ACID properties (Atomicity, Consistency, Isolation, Durability) and isolation levels (Read Uncommitted, Read Committed, Repeatable Read, Serializable).
3. **Performance Optimization**:
   - Query Execution Plans: Reading `EXPLAIN ANALYZE`, recognizing sequential scans vs index scans vs index-only scans.
   - Preventing N+1 queries in ORMs.
   - Partitioning large tables (Range, Hash, List) and connection pooling.

## Resume Proof Points
- "Tuned slow PostgreSQL reporting queries by redesigning composite indexes and refactoring subqueries into CTEs, slashing runtime from 42s to 1.8s."
- "Designed relational schema for e-commerce checkout supporting 50k daily transactions with strict foreign key integrity and zero data loss."

## Common Technical Interview Questions
- What causes a database deadlock and how do you resolve or prevent it?
  - Deadlocks occur when two or more transactions hold locks on resources and wait for each other's locks. Prevention: acquire locks in consistent order, keep transactions short, and use proper isolation levels.
- Compare B-Tree indexes and Hash indexes.
  - B-Trees support range queries (`>`, `<`, `BETWEEN`) and ordering (`ORDER BY`), while Hash indexes only support exact equality checks (`=`).
