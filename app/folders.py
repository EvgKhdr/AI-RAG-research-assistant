"""Folder and document storage: files on disk (DOCS_DIR/<folder id>/<name>) + metadata in SQLite.

All changes go through this module so disk and DB stay in sync. Folders are keyed by id on disk,
so folder names never touch the filesystem.
"""
import hashlib
import shutil
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

import ingest
import vectorstore
from db import DOCS_DIR, Document, Folder, engine

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class FolderError(ValueError):
    """User-facing validation problem (bad name, duplicate folder, unsupported file)."""


def _folder_dir(folder_id: int) -> Path:
    return DOCS_DIR / str(folder_id)


def _clean_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name.strip()  # drop any path components
    if not name or name in {".", ".."}:
        raise FolderError("Invalid file name.")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise FolderError(f"Unsupported file type: {name}")
    return name


def _unique_name(taken: set[str], name: str) -> str:
    """'a.pdf' -> 'a (1).pdf', 'a (2).pdf', ... when the name is already used."""
    if name.lower() not in taken:
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while f"{stem} ({i}){suffix}".lower() in taken:
        i += 1
    return f"{stem} ({i}){suffix}"


def list_folders() -> list[dict]:
    with Session(engine) as s:
        return [
            {
                "id": f.id,
                "name": f.name,
                "documents": [
                    {"id": d.id, "filename": d.filename, "size": d.size, "status": d.status}
                    for d in f.documents
                ],
            }
            for f in s.scalars(select(Folder).order_by(Folder.name))
        ]


def create_folder(name: str) -> int:
    name = name.strip()
    if not name:
        raise FolderError("Enter a folder name.")
    if len(name) > 100:
        raise FolderError("Folder name is too long (max 100 characters).")
    with Session(engine) as s:
        if s.scalar(select(Folder.id).where(func.lower(Folder.name) == name.lower())):
            raise FolderError("Folder already exists.")
        folder = Folder(name=name)
        s.add(folder)
        s.commit()
        return folder.id


def delete_folder(folder_id: int) -> None:
    vectorstore.delete_folder(folder_id)
    with Session(engine) as s:
        folder = s.get(Folder, folder_id)
        if folder is not None:
            s.delete(folder)
            s.commit()
    shutil.rmtree(_folder_dir(folder_id), ignore_errors=True)


def add_document(folder_id: int, filename: str, data: bytes) -> tuple[str, str | None]:
    """Store and index a file. Returns (stored name, indexing error or None).

    A file that fails to index stays stored with status 'failed'.
    """
    filename = _clean_filename(filename)
    with Session(engine) as s:
        folder = s.get(Folder, folder_id)
        if folder is None:
            raise FolderError("Folder no longer exists.")
        taken = {d.filename.lower() for d in folder.documents}
        stored = _unique_name(taken, filename)
        directory = _folder_dir(folder_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / stored
        path.write_bytes(data)
        try:
            doc = Document(
                folder_id=folder_id,
                filename=stored,
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            s.add(doc)
            s.commit()
            doc_id = doc.id
        except Exception:
            path.unlink(missing_ok=True)  # don't leave an untracked file behind
            raise
    try:
        ingest.index_document(doc_id)
        return stored, None
    except Exception as e:
        return stored, str(e)


def reindex_document(doc_id: int) -> str | None:
    """Retry indexing a stored document. Returns an error message or None."""
    try:
        ingest.index_document(doc_id)
        return None
    except Exception as e:
        return str(e)


def delete_document(doc_id: int) -> None:
    with Session(engine) as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return
        path = _folder_dir(doc.folder_id) / doc.filename
        vectorstore.delete_document(doc_id)
        s.delete(doc)
        s.commit()
    path.unlink(missing_ok=True)


def document_path(doc_id: int) -> Path | None:
    with Session(engine) as s:
        doc = s.get(Document, doc_id)
        return _folder_dir(doc.folder_id) / doc.filename if doc else None
