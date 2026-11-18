# System Design & Scalability Guide

## Core Architecture Concepts
1. **Scalability**: Horizontal scaling (adding machines) vs Vertical scaling (adding hardware capacity). Stateless architecture vs stateful services.
2. **Load Balancing**: Round robin, least connections, IP hash, Layer 4 (TCP) vs Layer 7 (HTTP) reverse proxies (NGINX, Envoy).
3. **Caching Strategies**: Cache-aside, read-through, write-through, write-back. Cache eviction policies (LRU, LFU). Redis and Memcached.
4. **Data Partitioning & Sharding**: Consistent hashing, horizontal sharding, database replication (leader-follower, multi-leader), and failover strategies.
5. **Asynchronous Processing**: Message brokers (Kafka, RabbitMQ, Celery), decoupling services, event-driven architectures, handling backpressure.
6. **CAP Theorem**: Consistency, Availability, Partition Tolerance. Pacelc theorem trade-offs.

## System Design Interview Framework
1. **Clarify Requirements**: Functional requirements, non-functional requirements (throughput, latency, availability, durability).
2. **Back-of-the-envelope Estimations**: QPS, storage per year, network bandwidth.
3. **High-Level Design**: Client -> CDN/Load Balancer -> API Gateway -> Application Servers -> Cache/DB.
4. **Deep Dive**: Concurrency, bottlenecks, single points of failure (SPOF), monitoring, and disaster recovery.
