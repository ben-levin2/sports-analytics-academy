import streamlit as st

from .. import config, progress, ui


def _exported():
    progress.mark_exported(ui.get_state())


def render():
    state = ui.get_state()
    ui.hero("Data", "Progress & Backup", "Where your progress lives, and how to keep it safe.")

    m = state["meta"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Last saved", ui.fmt_ts(m.get("last_saved")) if m.get("last_saved") else "never")
    c2.metric("Last backup", ui.fmt_ts(m.get("last_export")) if m.get("last_export") else "never")
    c3.metric("Changes since backup", m.get("changes_since_export", 0))

    st.markdown("### Export a backup")
    st.download_button("⬇ Download progress (JSON)", data=progress.to_json(state),
                       file_name=f"academy-progress-{progress.today().isoformat()}.json",
                       mime="application/json", type="primary", on_click=_exported)

    st.markdown("### Restore from a backup")
    up = st.file_uploader("Progress JSON file", type=["json"], key="restore_upload")
    mode = st.radio("How to restore", ["Merge with current progress (keeps everything)", "Replace current progress"],
                    key="restore_mode")
    if up is not None and st.button("Restore", key="restore_btn"):
        try:
            incoming = progress.from_json(up.getvalue().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            st.error(f"Could not restore: {exc}")
        else:
            new = progress.merge(state, incoming) if mode.startswith("Merge") else incoming
            st.session_state["progress"] = new
            progress.save(new)
            st.session_state["load_info"] = {"source": "file", "message": "Restored from backup."}
            st.success("Progress restored.")
            st.rerun()

    with st.expander("Reset all progress"):
        st.warning("This erases every quiz attempt, lab check and project step. Download a backup first.")
        if st.checkbox("I understand, reset everything", key="reset_confirm") and st.button("Reset now", key="reset_btn"):
            st.session_state["progress"] = progress.default_state()
            progress.save(st.session_state["progress"])
            st.rerun()

    st.markdown("### How storage works")
    st.markdown(f"""
- Progress is saved to **`{config.PROGRESS_FILE.relative_to(config.ROOT) if config.PROGRESS_FILE.is_relative_to(config.ROOT) else config.PROGRESS_FILE}`** after every quiz, lab check and project step. The write is atomic, so a crash can't half-write it.
- **Locally**, that file is permanent until you delete it.
- **On Streamlit Community Cloud**, the disk is temporary. The file is wiped whenever the app reboots, sleeps, or you push new code. **Download a backup regularly** (the sidebar reminds you after {config.BACKUP_NUDGE_AFTER} changes) and restore it after a restart.
- **One file, one learner.** A public Cloud app shares this file with every visitor. Set the app to private in Cloud settings.
- Uploaded datasets are stored separately in `data/uploads/` and are **not** part of the progress backup. Keep your CSVs.
- Streaks use the **{config.TIMEZONE}** time zone (change it in `academy/config.py`).
""")
    with st.expander("View raw progress JSON"):
        st.json(state, expanded=False)
