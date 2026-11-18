# Python Software Engineering Guide

## Key Concepts
1. **Language Fundamentals**: Memory management, GIL (Global Interpreter Lock), garbage collection, reference counting, mutable vs immutable types, list/dict/set comprehensions, generators, and iterators.
2. **Object-Oriented & Functional Design**: Magic methods (`__enter__`, `__exit__`, `__call__`, `__repr__`), class decorators, properties, dataclasses, closures, and functools.
3. **Web Frameworks**: Flask, FastAPI, Django. RESTful API design, request life cycle, middleware, dependency injection, and asynchronous request handling with asyncio/httpx.
4. **Data Handling & Testing**: pytest, unittest.mock, fixtures, parameterized testing, type annotations (typing, pydantic), and database ORM (SQLAlchemy).
5. **Concurrency & Performance**: threading vs multiprocessing vs asyncio. Profiling with cProfile, memory profiling, and Cython/C-extensions where necessary.

## Resume Highlights for Python Engineers
- Emphasize architecture: "Architected asynchronous REST microservices using Python, FastAPI, and PostgreSQL handling 5,000 requests/sec with p99 latency < 65ms."
- Demonstrate quality: "Integrated pytest test suites with 92% code coverage into GitHub Actions CI/CD pipeline, catching regressions pre-deployment."
- Show scaling: "Optimized database queries with SQLAlchemy connection pooling and Redis caching, cutting average endpoint response time by 45%."

## Interview Questions & Focus
- How does Python's GIL impact multithreaded CPU-bound vs I/O-bound programs?
  - GIL allows only one native thread to execute Python bytecode at once. For I/O-bound tasks, threading or asyncio yields the GIL during I/O waits, making it efficient. For CPU-bound tasks, multiprocessing is required to leverage multiple CPU cores.
- Explain Python decorators and write an example of an execution timing decorator.
- What is the difference between a generator and a regular function returning a list?
  - A generator yields items lazily on demand using the iterator protocol, saving memory when processing massive sequences ($O(1)$ memory vs $O(N)$).
