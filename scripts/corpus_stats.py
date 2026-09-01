"""
DocsQuery - Corpus Statistics CLI

Prints basic statistics about the persisted BM25 corpus.
"""

from collections import Counter

from app.config.settings import get_settings
from app.retrieval.bm25_storage import BM25Storage


def main() -> None:
    """
    Print corpus statistics.
    """

    settings = get_settings()

    storage = BM25Storage(settings.bm25_index_path)

    if not storage.exists():
        print("BM25 corpus does not exist.")
        return

    chunks = storage.load()

    documents = {chunk.document_id for chunk in chunks}

    sources = Counter(chunk.source for chunk in chunks)

    print()
    print("DocsQuery Corpus Statistics")
    print("=" * 60)
    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")
    print()
    print("Chunks by source:")

    for source, count in sources.items():
        print(f"  {source}: {count}")

    print("=" * 60)


if __name__ == "__main__":
    main()
