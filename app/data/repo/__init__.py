"""The repository, in parts.

``SqliteRepo`` is one class and one import for the rest of the application
- ``from app.data.sqlite_repo import SqliteRepo``, as it always was. What
changed is where its methods live: a 3 300-line module is not something
anyone reads, and the seams between its subjects were already marked by
section comments (todo.txt N-5).

Each part here is a mixin with no state of its own and ``__slots__ = ()``,
so the concrete class keeps the slots its dataclass declares - a typo in
an attribute name still fails loudly instead of quietly creating one. The
fields, the connection, the caches and the transactions stay on
``SqliteRepo`` itself: they are what the parts share.
"""
