"""
DocsQuery - Document Text Cleaner

This module cleans text extracted from documents.

The cleaner does NOT:
- split text into chunks
- create embeddings
- perform retrieval

Its only responsibility is to normalize extracted text while
preserving the meaning of the original document.
"""

import re

from app.ingestion.models import DocumentPage


def clean_text(text: str) -> str:
    """
    Clean raw text extracted from a document.

    The cleaning process:
    1. Normalizes Windows-style line endings.
    2. Removes excessive spaces/tabs.
    3. Removes excessive blank lines.
    4. Removes whitespace at the beginning/end.

    Args:
        text:
            Raw extracted document text.

    Returns:
        Cleaned text.
    """

    # --------------------------------------------------------
    # Normalize line endings.
    #
    # Different operating systems may represent a new line
    # differently:
    #
    # Windows: \r\n
    # macOS/Linux: \n
    #
    # We normalize everything to \n.
    # --------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # --------------------------------------------------------
    # Replace tabs with a normal space.
    # --------------------------------------------------------

    text = text.replace("\t", " ")

    # --------------------------------------------------------
    # Collapse multiple spaces.
    #
    # Example:
    #
    # "Hello     world"
    #
    # becomes:
    #
    # "Hello world"
    #
    # We intentionally do NOT remove newlines here.
    # Newlines can represent paragraph boundaries.
    # --------------------------------------------------------

    text = re.sub(r"[ ]{2,}", " ", text)

    # --------------------------------------------------------
    # Remove spaces at the beginning/end of each line.
    # --------------------------------------------------------

    lines = [line.strip() for line in text.split("\n")]

    # --------------------------------------------------------
    # Remove excessive blank lines.
    #
    # Multiple blank lines are reduced to one.
    # --------------------------------------------------------

    cleaned_lines: list[str] = []

    previous_line_was_empty = False

    for line in lines:
        if line == "":
            if not previous_line_was_empty:
                cleaned_lines.append("")

            previous_line_was_empty = True

        else:
            cleaned_lines.append(line)
            previous_line_was_empty = False

    # --------------------------------------------------------
    # Rebuild the text.
    # --------------------------------------------------------

    cleaned_text = "\n".join(cleaned_lines)

    # Remove whitespace at the beginning/end of the document.
    return cleaned_text.strip()


def clean_page(page: DocumentPage) -> DocumentPage:
    """
    Clean a DocumentPage while preserving its metadata.

    Args:
        page:
            Original DocumentPage.

    Returns:
        A new DocumentPage containing cleaned text.
    """

    return DocumentPage(
        page_number=page.page_number,
        text=clean_text(page.text),
        source=page.source,
    )
