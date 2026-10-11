"""SQLite chat storage (SQLAlchemy). The database file is created on first launch."""
import json
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import ForeignKey, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DB_PATH = DATA_DIR / "chats.db"
DOCS_DIR = DATA_DIR / "docs"


class Base(DeclarativeBase):
    pass


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(100))  # set on assistant messages
    sources: Mapped[str | None] = mapped_column(Text)  # JSON list of retrieved chunks
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    chat: Mapped[Chat] = relationship(back_populates="messages")


class Folder(Base):
    __tablename__ = "folders"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    documents: Mapped[list["Document"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan", order_by="Document.id"
    )


class Document(Base):
    """Metadata for an uploaded file. The file itself lives under DOCS_DIR/<folder id>/."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))  # stored name, unique within the folder
    size: Mapped[int]
    sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="stored")  # stored | indexed | failed
    uploaded_at: Mapped[datetime] = mapped_column(default=datetime.now)
    folder: Mapped[Folder] = relationship(back_populates="documents")


DATA_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}")


def init_db() -> None:
    """Create tables if missing. Safe to call on every launch."""
    first_run = "folders" not in inspect(engine).get_table_names()
    Base.metadata.create_all(engine)
    if "sources" not in {c["name"] for c in inspect(engine).get_columns("messages")}:
        with engine.begin() as conn:  # databases created before RAG
            conn.execute(text("ALTER TABLE messages ADD COLUMN sources TEXT"))
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if first_run:  # seed once; a deleted "General" must stay deleted
        with Session(engine) as s:
            s.add(Folder(name="General"))
            s.commit()


def create_chat(model: str, title: str = "New chat") -> int:
    with Session(engine) as s:
        chat = Chat(model=model, title=title)
        s.add(chat)
        s.commit()
        return chat.id


def list_chats() -> list[dict]:
    with Session(engine) as s:
        rows = s.scalars(select(Chat).order_by(Chat.created_at.desc(), Chat.id.desc()))
        return [{"id": c.id, "title": c.title, "model": c.model} for c in rows]


def get_chat(chat_id: int) -> dict | None:
    with Session(engine) as s:
        c = s.get(Chat, chat_id)
        if c is None:
            return None
        return {
            "id": c.id,
            "title": c.title,
            "model": c.model,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "model": m.model,
                    "sources": json.loads(m.sources) if m.sources else [],
                }
                for m in c.messages
            ],
        }


def update_chat(chat_id: int, *, title: str | None = None, model: str | None = None) -> None:
    with Session(engine) as s:
        c = s.get(Chat, chat_id)
        if c is None:
            return
        if title is not None:
            c.title = title
        if model is not None:
            c.model = model
        s.commit()


def delete_chat(chat_id: int) -> None:
    with Session(engine) as s:
        c = s.get(Chat, chat_id)
        if c is not None:
            s.delete(c)
            s.commit()


def add_message(
    chat_id: int, role: str, content: str, model: str | None = None, sources: list[dict] | None = None
) -> None:
    with Session(engine) as s:
        s.add(Message(
            chat_id=chat_id, role=role, content=content, model=model,
            sources=json.dumps(sources) if sources else None,
        ))
        s.commit()
