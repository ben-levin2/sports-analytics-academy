import pandas as pd
import streamlit as st

from .. import config, datasets, ui


def render():
    ui.hero("Data", "Data Sources",
            "Replace simulated data with your own CSVs. Projects pick up the new data automatically.")
    st.markdown("""
**Priority for every dataset:** your uploaded CSV → a path in `datasets.toml` (local runs) → simulated data.
Nothing here reads PSFAL files directly. Export what you need from your own projects as CSV.
""")
    if config.IS_CLOUD:
        st.warning("On Streamlit Community Cloud, uploaded files are erased when the app restarts or redeploys. "
                   "Keep your CSVs and re-upload after a restart. Your progress backup does not include datasets.")

    spec_id = st.selectbox("Dataset", list(datasets.SPECS), key="ds_pick",
                           format_func=lambda i: f"{datasets.SPECS[i].name}  (used by {', '.join(datasets.SPECS[i].used_by)})")
    spec = datasets.SPECS[spec_id]
    src = datasets.status(spec_id)
    st.markdown(f"**{spec.name}**: {spec.description}")
    st.markdown("| Column | Meaning | |\n|---|---|---|\n"
                + "\n".join(f"| `{c}` | {m} | required |" for c, m in spec.required.items())
                + "".join(f"\n| `{c}` | {m} | optional |" for c, m in spec.optional.items()))

    if src["source"] == "upload":
        st.success(f"Using uploaded data: **{src.get('label')}** ({src.get('rows')} rows, "
                   f"{'marked as real' if src.get('real') else 'NOT marked as real'}, saved {ui.fmt_ts(src.get('ts'))}).")
        if st.button("Remove upload and go back to simulated data", key=f"rm_{spec_id}"):
            datasets.remove_upload(spec_id)
            st.cache_data.clear()
            st.rerun()
    elif src["source"] == "path":
        st.info(f"Using configured file: `{src['label']}`")
    else:
        ui.sim_banner("This dataset currently uses simulated data. Simulated variants: "
                      + ", ".join(spec.simulated) + ".")

    st.markdown("### Upload a CSV")
    up = st.file_uploader("CSV file", type=["csv"], key=f"up_{spec_id}")
    if up is None:
        return
    try:
        raw = pd.read_csv(up)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    st.dataframe(raw.head(8), hide_index=True)

    st.markdown("**Map your columns** (pick which of your columns holds each field)")
    cols = list(raw.columns)
    mapping = {}
    grid = st.columns(3)
    for i, (field, meaning) in enumerate({**spec.required, **{k: v for k, v in spec.optional.items() if not k.startswith("<")}}.items()):
        opts = ["(none)"] + cols
        default = cols.index(field) + 1 if field in cols else 0
        choice = grid[i % 3].selectbox(f"`{field}`" + ("" if field in spec.required else " (optional)"), opts,
                                       index=default, key=f"map_{spec_id}_{field}", help=meaning)
        if choice != "(none)":
            mapping[choice] = field

    if spec.features:
        used = set(mapping)
        numeric = [c for c in cols if c not in used and pd.api.types.is_numeric_dtype(raw[c])]
        feats = st.multiselect("Feature columns to keep (three or more numeric stats)", numeric,
                               default=numeric[:8], key=f"feats_{spec_id}")
        for c in feats:
            mapping[c] = c

    label = st.text_input("Label for this dataset", value=up.name, key=f"lbl_{spec_id}")
    real = st.checkbox("This is real data that I have the right to use (required for Production Candidate gates)",
                       key=f"real_{spec_id}")
    if st.button("Validate and save", type="primary", key=f"save_{spec_id}"):
        df = raw[list(mapping)].rename(columns=mapping)
        errs = datasets.save_upload(spec_id, df, label, real)
        if errs:
            st.error("Not saved:\n\n" + "\n".join(f"- {e}" for e in errs))
        else:
            st.cache_data.clear()
            st.success(f"Saved {len(df)} rows. Projects using this dataset now use your file.")
