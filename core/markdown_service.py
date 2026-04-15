from __future__ import annotations

import html

import markdown as md


MARKDOWN_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "toc"]


def render_markdown_to_html(markdown_text: str) -> str:
    if not markdown_text.strip():
        markdown_text = "_No markdown content found in this package._"

    body = md.markdown(markdown_text, extensions=MARKDOWN_EXTENSIONS)
    return (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body {font-family: Segoe UI, Arial, sans-serif; margin: 12px; line-height: 1.45;}"
        "h1, h2, h3 {margin-top: 1.2em;}"
        "pre {background: #f4f4f4; padding: 8px; border-radius: 4px; overflow-x: auto;}"
        "code {background: #f4f4f4; padding: 2px 4px; border-radius: 3px;}"
        "table {border-collapse: collapse; width: 100%; margin: 8px 0;}"
        "th, td {border: 1px solid #d8d8d8; padding: 6px; text-align: left;}"
        "th {background: #efefef;}"
        "blockquote {border-left: 3px solid #ccc; margin: 8px 0; padding-left: 10px; color: #555;}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )


def plain_text_to_html(text: str) -> str:
    escaped = html.escape(text)
    return (
        "<html><head><meta charset='utf-8'>"
        "<style>body {font-family: Consolas, monospace; white-space: pre-wrap; margin: 12px;}</style>"
        "</head><body>"
        f"{escaped}"
        "</body></html>"
    )
