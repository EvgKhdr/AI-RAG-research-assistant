import html

import streamlit as st

import db
import folders
import rag
from config import (
    DEFAULT_MODEL, MAX_HISTORY_MESSAGES, MODELS, PROVIDERS, api_key, is_available, model_label, context_chars_for,
)
from llm import stream_chat

st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")

CSS = """
<style>
[data-testid="stSidebar"] { border-right: 1px solid #222a35; }
/* fixed sidebar width: no drag-to-resize */
[data-testid="stSidebar"][aria-expanded="true"] {
    width: 300px !important; min-width: 300px !important; max-width: 300px !important;
}
[data-testid="stSidebarResizeHandle"] { display: none !important; }
[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start; text-align: left; width: 100%;
    border: none; background: transparent; color: #c9d1d9;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #1f2630; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #232c3a; color: #fff; }
.sidebar-title { font-size: 1.2rem; font-weight: 600; margin-bottom: .25rem; }
.sidebar-caption { color: #7d8590; font-size: .75rem; text-transform: uppercase;
    letter-spacing: .06em; margin: 1rem 0 .25rem; }
.folder-meta { color: #7d8590; font-size: .8rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- state (chats, folders and document metadata live in SQLite) ----------
@st.cache_resource
def _init() -> bool:
    db.init_db()  # creates data/chats.db on first launch
    return True


_init()


def new_chat() -> None:
    """Switch to an empty draft; the DB row is created on the first message."""
    st.session_state.active_chat = None
    st.session_state.view = "chat"


def delete_chat(cid: int) -> None:
    db.delete_chat(cid)
    if st.session_state.active_chat == cid:
        new_chat()


@st.dialog("Delete chat?")
def confirm_delete(cid: int, title: str) -> None:
    st.write(f"“{title}” will be permanently deleted.")
    yes, no = st.columns(2)
    if yes.button("Delete", type="primary", use_container_width=True):
        delete_chat(cid)
        st.rerun()
    if no.button("Cancel", use_container_width=True):
        st.rerun()


def init_state():
    ss = st.session_state
    ss.setdefault("view", "chat")
    ss.setdefault("draft_model", DEFAULT_MODEL)
    if "active_chat" not in ss:
        chats = db.list_chats()
        ss.active_chat = chats[0]["id"] if chats else None  # reopen the latest chat


init_state()
ss = st.session_state


# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="sidebar-title">📚 Research Assistant</div>', unsafe_allow_html=True)
    if st.button("＋  New chat", key="new_chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown('<div class="sidebar-caption">History</div>', unsafe_allow_html=True)
    with st.container(height=420, border=False):
        for chat in db.list_chats():
            cid = chat["id"]
            active = cid == ss.active_chat and ss.view == "chat"
            open_col, del_col = st.columns([4, 1], gap="small", vertical_alignment="center")
            if open_col.button(
                f"💬  {chat['title']}",
                key=f"chat_{cid}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                ss.active_chat, ss.view = cid, "chat"
                st.rerun()
            if del_col.button("🗑", key=f"delchat_{cid}", help="Delete chat"):
                confirm_delete(cid, chat["title"])

    st.divider()
    if st.button(
        "📁  Documents",
        key="open_docs",
        type="primary" if ss.view == "docs" else "secondary",
        use_container_width=True,
    ):
        ss.view = "docs"
        st.rerun()

    st.markdown('<div class="sidebar-caption">Providers</div>', unsafe_allow_html=True)
    for name, p in PROVIDERS.items():
        if api_key(name):
            st.markdown(f"🟢 {name}")
        else:
            st.markdown(f"⚪ {name} — set `{p['key_env']}` in `.env` ([get key]({p['key_url']}))")


# ---------- views ----------
def show_sources(sources: list[dict] | None) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, h in enumerate(sources, 1):
            where = f"{h['filename']}, p. {h['page']}" if h.get("page") else h["filename"]
            via = " · 🔤 keyword match" if h.get("via") in ("keyword", "both") else ""
            st.markdown(f"**[{i}] {where}** · match {h['score']:.0%}{via}")
            st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))


def chat_view():
    cid = ss.active_chat
    chat = db.get_chat(cid) if cid is not None else None
    if chat is None:  # draft, or the chat was removed
        cid = ss.active_chat = None
        chat = {"title": "New chat", "model": ss.draft_model, "messages": []}

    st.subheader(chat["title"])

    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                st.caption(model_label(msg["model"]))
                show_sources(msg.get("sources"))

    if not chat["messages"]:
        st.info("Ask a question about your documents. Pick a model below the chat.")

    # st.bottom pins widgets to the bottom bar, next to the chat input
    with st.bottom:
        ids = [m["id"] for m in MODELS]
        picker, scope_col, strict_col = st.columns([2, 3, 1.2], vertical_alignment="center")
        all_folders = {f["id"]: f["name"] for f in folders.list_folders()}
        scope = scope_col.multiselect(
            "Folders",
            list(all_folders),
            format_func=all_folders.get,
            placeholder="Searching all folders",
            label_visibility="collapsed",
            key="scope",
        )
        model = picker.selectbox(
            "Model",
            ids,
            index=ids.index(chat["model"]) if chat["model"] in ids else 0,
            format_func=lambda i: model_label(i) + ("" if is_available(i) else "  (no API key)"),
            label_visibility="collapsed",
            key=f"model_{cid}",
        )
        docs_only = strict_col.toggle(
            "Docs only", value=True, key="docs_only",
            help="On: answer strictly from your documents. Off: the model may add general knowledge, labelled as such.",
        )
        prompt = st.chat_input("Ask something about your documents…")

    if model != chat["model"]:
        chat["model"] = model
        if cid is None:
            ss.draft_model = model
        else:
            db.update_chat(cid, model=model)

    if prompt:
        if not is_available(model):
            st.error(f"{model_label(model)} needs an API key. See Providers in the sidebar.")
            return
        history = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
        history = history[-MAX_HISTORY_MESSAGES:]
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents…"):
                    docs = folders.scope_documents(scope or None)
                    hits = rag.retrieve(prompt, scope or None, context_chars_for(model), docs)
                messages = rag.build_messages(history, prompt, hits, docs, docs_only)
                reply = st.write_stream(stream_chat(model, messages))
            except Exception as e:  # retrieval, network, auth or quota errors
                st.error(f"Request failed: {e}")  # nothing is saved, so history stays consistent
                return
            show_sources(hits)
        if cid is None:  # first message of a draft: the chat row is created only now
            title = prompt[:40] + ("…" if len(prompt) > 40 else "")
            cid = ss.active_chat = db.create_chat(model, title)
        db.add_message(cid, "user", prompt)
        db.add_message(cid, "assistant", reply, model, hits)
        st.rerun()


@st.dialog("Delete folder?")
def confirm_delete_folder(folder_id: int, name: str, n_files: int) -> None:
    st.write(f"“{name}” and its {n_files} file(s) will be permanently deleted.")
    yes, no = st.columns(2)
    if yes.button("Delete", type="primary", use_container_width=True):
        folders.delete_folder(folder_id)
        st.rerun()
    if no.button("Cancel", use_container_width=True):
        st.rerun()


STATUS_ICON = {"indexed": "✅", "failed": "⚠️", "stored": "⏳"}


def _fmt_size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def docs_view():
    st.subheader("📁 Documents")
    st.caption("Organise your documents into folders. Folders can be used to scope chat answers.")

    with st.form("new_folder", clear_on_submit=True, border=False):
        c1, c2 = st.columns([4, 1])
        name = c1.text_input("New folder", placeholder="New folder name", label_visibility="collapsed")
        if c2.form_submit_button("Create folder", use_container_width=True):
            try:
                folders.create_folder(name)
                st.rerun()
            except folders.FolderError as e:
                st.warning(str(e))

    for msg in ss.pop("flash", []):
        st.warning(msg)

    all_folders = folders.list_folders()
    if not all_folders:
        st.info("No folders yet. Create one above to start adding documents.")
    cols = st.columns(3)
    for i, folder in enumerate(all_folders):
        fid, docs = folder["id"], folder["documents"]
        with cols[i % 3].container(border=True):
            st.markdown(f"### 📂 {folder['name']}")
            st.markdown(f'<div class="folder-meta">{len(docs)} file(s)</div>', unsafe_allow_html=True)
            for d in docs:
                name_col, retry_col, del_col = st.columns([5, 1, 1], vertical_alignment="center")
                name_col.markdown(
                    f"{STATUS_ICON.get(d['status'], '📄')} {html.escape(d['filename'])} "
                    f"<span class='folder-meta'>· {_fmt_size(d['size'])}</span>",
                    unsafe_allow_html=True,
                )
                if d["status"] != "indexed" and retry_col.button(
                    "↻", key=f"retry_{d['id']}", help="Retry indexing"
                ):
                    with st.spinner("Indexing…"):
                        err = folders.reindex_document(d["id"])
                    if err:
                        ss.flash = [f"{d['filename']}: {err}"]
                    st.rerun()
                if del_col.button("🗑", key=f"deldoc_{d['id']}", help="Delete file"):
                    folders.delete_document(d["id"])
                    st.rerun()
            uploads = st.file_uploader(
                "Add documents",
                accept_multiple_files=True,
                type=[e.lstrip(".") for e in sorted(folders.ALLOWED_EXTENSIONS)],
                key=f"up_{fid}_{ss.get('up_gen', 0)}",
                label_visibility="collapsed",
            )
            b1, b2 = st.columns(2)
            if b1.button("Add", key=f"add_{fid}", disabled=not uploads, use_container_width=True):
                ss.flash = []
                with st.spinner("Indexing documents (the first run downloads the embedding model)…"):
                    for u in uploads:
                        try:
                            _, err = folders.add_document(fid, u.name, u.getvalue())
                            if err:
                                ss.flash.append(f"{u.name}: stored but not indexed — {err}")
                        except folders.FolderError as e:
                            ss.flash.append(str(e))
                ss.up_gen = ss.get("up_gen", 0) + 1
                st.rerun()
            if b2.button("Delete", key=f"del_{fid}", use_container_width=True):
                confirm_delete_folder(fid, folder["name"], len(docs))


(chat_view if ss.view == "chat" else docs_view)()
