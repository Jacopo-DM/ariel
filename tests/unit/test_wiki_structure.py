"""Structure checks for the knowledge wiki in ``.claude/wiki``.

The rules come from ``.claude/SCHEMA.md``. Obsidian resolves ``[[links]]``
by page name, so names must stay unique when pages move between folders.
"""

# Standard library
import re
from pathlib import Path

# Third-party libraries
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / ".claude" / "wiki"

TOPIC_FOLDERS = {
    "mujoco",
    "mjx",
    "cma-es",
    "nevergrad",
    "ariel",
    "astropy",
    "pydantic",
}
ROOT_FILES = {"index.md", "log.md"}
PAGE_TYPES = {
    "api_reference",
    "algorithm_reference",
    "concept_reference",
    "ariel_reference",
    "source_summary",
}
REQUIRED_KEYS = ("type", "tags", "source", "date_ingested")

FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
WIKILINK = re.compile(r"\[\[([^\]|#]+)")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
H1 = re.compile(r"^# (.+)$", re.MULTILINE)


def _pages() -> list[Path]:
    """Return every wiki page except the logs and the index."""
    return sorted(
        page
        for page in WIKI.rglob("*.md")
        if page.name not in ROOT_FILES and ".obsidian" not in page.parts
    )


def _frontmatter(page: Path) -> dict[str, str]:
    match = FRONTMATTER.match(page.read_text(encoding="utf-8"))
    if match is None:
        return {}
    pairs = re.findall(r"^(\w+):\s*(.*)$", match.group(1), re.MULTILINE)
    return dict(pairs)


def _links(page: Path) -> list[str]:
    text = page.read_text(encoding="utf-8")
    text = FENCED_CODE.sub("", text)
    text = INLINE_CODE.sub("", text)
    return [link.strip() for link in WIKILINK.findall(text)]


def _ids(pages: list[Path]) -> list[str]:
    return [str(page.relative_to(WIKI)) for page in pages]


def test_there_are_pages() -> None:
    assert _pages()


def test_one_wiki_and_one_log() -> None:
    assert not (REPO_ROOT / "wiki").exists()
    assert not (REPO_ROOT / "log.md").exists()
    assert (WIKI / "log.md").is_file()


def test_no_stray_page_at_the_wiki_root() -> None:
    stray = [
        page.name for page in WIKI.glob("*.md") if page.name not in ROOT_FILES
    ]
    assert stray == []


def test_pages_sit_in_known_topic_folders() -> None:
    folders = {
        page.relative_to(WIKI).parts[0]
        for page in _pages()
        if len(page.relative_to(WIKI).parts) > 1
    }
    assert folders <= TOPIC_FOLDERS


def test_page_names_are_unique() -> None:
    names = [page.stem for page in _pages()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == []


@pytest.mark.parametrize("page", _pages(), ids=_ids(_pages()))
def test_frontmatter_is_complete(page: Path) -> None:
    frontmatter = _frontmatter(page)
    missing = [key for key in REQUIRED_KEYS if key not in frontmatter]
    assert missing == []
    assert frontmatter["type"] in PAGE_TYPES


@pytest.mark.parametrize("page", _pages(), ids=_ids(_pages()))
def test_h1_matches_the_filename(page: Path) -> None:
    match = H1.search(page.read_text(encoding="utf-8"))
    assert match is not None
    assert match.group(1).strip() == page.stem


def test_wikilinks_resolve() -> None:
    names = {page.stem for page in _pages()} | {"index", "log"}
    broken = sorted(
        f"{page.relative_to(WIKI)} -> {link}"
        for page in _pages()
        for link in _links(page)
        if link not in names
    )
    assert broken == []


def test_index_lists_every_page() -> None:
    index = (WIKI / "index.md").read_text(encoding="utf-8")
    listed = set(WIKILINK.findall(index))
    missing = sorted(page.stem for page in _pages() if page.stem not in listed)
    assert missing == []
