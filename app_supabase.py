import streamlit as st
import pandas as pd
import time
from io import BytesIO
from datetime import datetime, date
from supabase import create_client, Client
from postgrest.exceptions import APIError

# -------------------------------
# Page & style (simple B/W)
# -------------------------------
st.set_page_config(page_title="FED3 Manager", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #fff; color: #111; }
      .stTabs [data-baseweb="tab"] { font-weight: 500; }
      .stDataFrame, .stDataEditor { font-size: 14px; }
      .stButton button { border-radius: 6px; }
      .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Secrets / Supabase client
# -------------------------------
@st.cache_resource
def get_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

sb = get_client()
ADMIN_CODE = st.secrets.get("ADMIN_CODE", None)  # optional

# -------------------------------
# Constants / field lists
# -------------------------------
STATUS_OPTIONS = ["Ready for Use", "In Use", "To Test", "Unclear"]
USERS = ["Emma", "Sushma", "Taylor", "Keydy", "Unassigned"]
ISSUE_OPTIONS = [
    "Electronics: Screen",
    "Electronics: Photointerrupter",
    "Electronics: SD card",
    "Electronics: Motor",
    "Electronics: Buzzer",
    "Jammed Pellets",
    "Housing issue",
    "Electronics issue",
    "General: Not working",
    "General: Broken",
]

DEVICE_FIELDS_EDITABLE = [
    "housing_id", "electronics_id",
    "housing_status", "electronics_status",
    "in_use", "user",
    "current_location", "exp_start_date",
    "notes", "issue_tags"
]
DEVICE_FIELDS_ALL = DEVICE_FIELDS_EDITABLE + [
    "status_in_lab", "status_with_mice",
    "status_bucket"
]

# -------------------------------
# Helpers
# -------------------------------
def ensure_tables_exist():
    try:
        _ = sb.table("devices").select("id").limit(1).execute()
        _ = sb.table("inventory").select("id").limit(1).execute()
        _ = sb.table("actions").select("id").limit(1).execute()
        return True
    except Exception:
        st.error("Supabase tables not found. Create tables first (see README/DDL).")
        st.stop()

ensure_tables_exist()

def df_to_dicts(df: pd.DataFrame):
    recs = df.to_dict(orient="records")
    out = []
    for r in recs:
        o = {}
        for k, v in r.items():
            if pd.isna(v):
                o[k] = None
            elif isinstance(v, (pd.Timestamp, datetime, date)):
                try:
                    o[k] = pd.to_datetime(v).to_pydatetime().isoformat()
                except Exception:
                    o[k] = str(v)
            else:
                o[k] = v
        out.append(o)
    return out

def get_table_df(name: str) -> pd.DataFrame:
    data = sb.table(name).select("*").execute().data
    return pd.DataFrame(data) if data else pd.DataFrame()

def update_rows_by_ids(name: str, ids: list[int], updates: dict):
    if not ids:
        return
    sb.table(name).update(updates).in_("id", ids).execute()

def log_action(actor: str, action: str, housing_id=None, electronics_id=None, details=None):
    try:
        sb.table("actions").insert({
            "actor": actor, "action": action, "housing_id": housing_id,
            "electronics_id": electronics_id, "details": details
        }).execute()
    except Exception:
        pass

def compute_bucket(rec: dict) -> str:
    in_use = bool(rec.get("in_use") or False)
    hs = str(rec.get("housing_status") or "").strip().lower()
    es = str(rec.get("electronics_status") or "").strip().lower()
    has_h = bool((rec.get("housing_id") or "").strip())
    has_e = bool((rec.get("electronics_id") or "").strip())

    if in_use:
        return "In Use"
    if hs == "working" and es == "working":
        return "Ready for Use"
    if (has_h or has_e) and (hs != "working" or es != "working"):
        return "To Test"
    return "Unclear"

def normalize_user_val(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "":
        return None
    return None if str(x).strip() == "Unassigned" else str(x).strip()

def coerce_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1","true","yes","y")


def clean_id(x):
    if x is None:
        return None
    s = str(x).strip()
    return s.upper() if s else None

def normalize_device(rec: dict) -> dict:
    out = rec.copy()
    out["housing_id"] = clean_id(out.get("housing_id"))
    out["electronics_id"] = clean_id(out.get("electronics_id"))

    # statuses: Unknown/"" -> None, else Title Case
    for k in ["housing_status", "electronics_status", "status_in_lab", "status_with_mice"]:
        v = out.get(k)
        if v is None:
            out[k] = None
        else:
            s = str(v).strip()
            out[k] = None if s == "" or s.lower() == "unknown" else s.title()

    # booleans
    out["in_use"] = bool(out.get("in_use") or False)

    # compute bucket after cleaning
    out["status_bucket"] = compute_bucket(out)
    return out

# --- Pools helpers ---
def get_pool_df(name: str) -> pd.DataFrame:
    data = sb.table(name).select("*").order("id").execute().data
    return pd.DataFrame(data) if data else pd.DataFrame()

def working_housing_options() -> list[tuple[str, str]]:
    """Return [(value,label)] for selectbox; value is the actual housing_id (may be empty)."""
    df = get_pool_df("housing_pool")
    if df.empty:
        return []
    df = df[df["status"] == "Working"].copy()
    # show both ID and a short notes preview
    opts = []
    for _, r in df.iterrows():
        hid = (r.get("housing_id") or "").strip()
        label = f"{hid or '(unassigned)'} — {r.get('notes') or ''}".strip(" —")
        # value we store is the housing_id text (can be empty string if truly unassigned)
        opts.append((hid, label))
    return opts

def get_pool_df(name: str) -> pd.DataFrame:
    data = sb.table(name).select("*").order("id").execute().data
    return pd.DataFrame(data) if data else pd.DataFrame()


def working_electronics_options() -> list[tuple[str, str]]:
    df = get_pool_df("electronics_pool")
    if df.empty:
        return []
    df = df[df["status"] == "Working"].copy()
    opts = []
    for _, r in df.iterrows():
        eid = (r.get("electronics_id") or "").strip()
        label = f"{eid or '(unassigned)'} — {r.get('notes') or ''}".strip(" —")
        opts.append((eid, label))
    return opts

def lookup_pool_notes_status(housing_id: str | None, electronics_id: str | None) -> dict:
    """Fetch status/notes for chosen pool parts to prefill device fields."""
    out = {}
    if housing_id:
        q = sb.table("housing_pool").select("status,notes").eq("housing_id", housing_id).limit(1).execute().data
        if q:
            out["housing_status"] = q[0].get("status")
            out["housing_notes"] = q[0].get("notes")
    if electronics_id:
        q = sb.table("electronics_pool").select("status,notes").eq("electronics_id", electronics_id).limit(1).execute().data
        if q:
            out["electronics_status"] = q[0].get("status")
            out["electronics_notes"] = q[0].get("notes")
    return out

def fetch_actions_snapshot(housing_id: str | None, electronics_id: str | None) -> list[dict]:
    q = sb.table("actions").select("*")
    if housing_id:
        q = q.eq("housing_id", housing_id)
    if electronics_id:
        q = q.eq("electronics_id", electronics_id)
    q = q.order("ts", desc=True)
    return q.execute().data or []

def fetch_device_snapshot(housing_id: str | None, electronics_id: str | None) -> dict | None:
    # Try by housing_id first, else by electronics_id
    if housing_id:
        row = sb.table("devices").select("*").eq("housing_id", housing_id).limit(1).execute().data
        if row: return row[0]
    if electronics_id:
        row = sb.table("devices").select("*").eq("electronics_id", electronics_id).limit(1).execute().data
        if row: return row[0]
    return None

def archive_one(housing_id: str | None, electronics_id: str | None, reason: str | None, actor: str):
    dev_snap = fetch_device_snapshot(housing_id, electronics_id)
    acts_snap = fetch_actions_snapshot(housing_id, electronics_id)
    rec = {
        "housing_id": housing_id,
        "electronics_id": electronics_id,
        "reason": reason or None,
        "device_snapshot": dev_snap or None,
        "actions_snapshot": acts_snap or None,
    }
    sb.table("archive").insert(rec).execute()
    # write an action for visibility
    log_action(actor, "archive", housing_id=housing_id, electronics_id=electronics_id, details=(reason or ""))


def archive_selected_devices(rows: list[dict], archive_housing: bool, archive_electronics: bool, actor: str, note: str = ""):
    """
    For each selected device row:
      - Write archive snapshot(s)
      - Return the non-archived side to its pool as Working
      - Deactivate the device row (is_active=False)
    """
    for r in rows:
        rid = int(r["id"])
        hid = r.get("housing_id")
        eid = r.get("electronics_id")

        # 1) Record the combo in archive_pairs (one row per archive action)
        try:
            sb.table("archive_pairs").insert({
                "housing_id": hid, "electronics_id": eid,
                "archived_by": actor, "note": (note or "")
            }).execute()
        except Exception:
            pass

        # 2) Per-part archival logs
        try:
            if archive_housing and hid:
                sb.table("archival_logs").insert({
                    "kind": "housing", "id_value": hid,
                    "action": "archive", "note": note or ""
                }).execute()
            if archive_electronics and eid:
                sb.table("archival_logs").insert({
                    "kind": "electronics", "id_value": eid,
                    "action": "archive", "note": note or ""
                }).execute()
        except Exception:
            pass

        # 3) Return the surviving side to its pool as Working (per spec)
        try:
            if archive_housing and not archive_electronics and eid:
                sb.table("inventory_electronics").upsert(
                    {"electronics_id": eid, "status": "Working", "notes": "Returned from archive"},
                    on_conflict="electronics_id"
                ).execute()
            if archive_electronics and not archive_housing and hid:
                sb.table("inventory_housing").upsert(
                    {"housing_id": hid, "status": "Working", "notes": "Returned from archive"},
                    on_conflict="housing_id"
                ).execute()
        except Exception:
            pass

        # 4) Deactivate the device row so it disappears from Overview/My FEDs
        sb.table("devices").update({"is_active": False}).eq("id", rid).execute()

        # 5) History/logs
        try:
            pieces = []
            if archive_housing: pieces.append("housing")
            if archive_electronics: pieces.append("electronics")
            log_action(
                actor, "archive",
                housing_id=hid, electronics_id=eid,
                details=f"archived={' & '.join(pieces)}; note={note or ''}"
            )
        except Exception:
            pass



# -------------------------------
# Sidebar
# -------------------------------
st.title("FED3 Manager")

with st.sidebar:
    st.header("View")
    st.checkbox("Show internal IDs", value=False, key="show_ids")
    auto = st.checkbox("Auto-refresh every 10s", value=False)
    actor = st.text_input("Your name (for history)", value="lab-user")

    if auto:
        now = time.time()
        last = st.session_state.get("last_refresh", 0.0)
        if now - last > 10:
            st.session_state["last_refresh"] = now
            st.rerun()
            
    st.write("---")
    st.header("Admin")
    admin_enabled = False
    if ADMIN_CODE:
        code = st.text_input("Enter admin code", type="password")
        admin_enabled = (code == ADMIN_CODE)
        if not admin_enabled:
            st.caption("Admin tools hidden until code matches.")
    else:
        st.caption("ADMIN_CODE not set; admin tools disabled.")

# -------------------------------
# Tabs
# -------------------------------
tab_overview, tab_mine, tab_add, tab_history, tab_inventory, tab_admin = st.tabs(
    ["Overview (Search & Edit)", "My FEDs", "Add Device", "History", "Inventory", "Admin"]
)

# -------------------------------
# Overview: searchable + selectable + inline edit
# -------------------------------
with tab_overview:
    st.subheader("Devices — filter, select, edit")

    df = get_table_df("devices")
    if df.empty:
        st.info("No devices yet. Add one in the **Add Device** tab.")
    else:
        # basic missing columns
        for c in ["status_bucket","user","issue_tags","housing_status","electronics_status",
                  "current_location","exp_start_date","notes","in_use","housing_id","electronics_id"]:
            if c not in df.columns:
                df[c] = pd.Series(dtype="object")
                # Only show active, “assembled” devices (both IDs present)
            if "is_active" not in df.columns:
                df["is_active"] = True
            df = df[(df["is_active"] == True)]
            df = df[(df["housing_id"].notna()) & (df["electronics_id"].notna())]


        # Filters
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
        status_pick = c1.multiselect("Status", STATUS_OPTIONS, default=STATUS_OPTIONS)
        user_vals = sorted([x for x in df["user"].dropna().unique().tolist()] + ["Unassigned"])
        user_pick = c2.multiselect("User", user_vals, default=user_vals)
        issue_vals = sorted([x for x in df["issue_tags"].dropna().unique().tolist()])
        issue_pick = c3.multiselect("Issue", issue_vals, default=issue_vals)
        id_search = c4.text_input("ID contains")
        text_search = c5.text_input("Search notes/location")

        def sstr(s): return s.fillna("").astype(str)

        work = df.copy()
        if status_pick:
            work = work[work["status_bucket"].isin(status_pick)]
        ucol = sstr(work["user"]).replace({"": "Unassigned"})
        if user_pick:
            work = work[ucol.isin(user_pick)]
        if issue_pick:
            w_issues = sstr(work["issue_tags"])
            work = work[w_issues.isin(issue_pick)]
        if id_search:
            mask = sstr(work["housing_id"]).str.contains(id_search, case=False, na=False) | \
                   sstr(work["electronics_id"]).str.contains(id_search, case=False, na=False)
            work = work[mask]
        if text_search:
            mask = sstr(work["notes"]).str.contains(text_search, case=False, na=False) | \
                   sstr(work["current_location"]).str.contains(text_search, case=False, na=False)
            work = work[mask]

        # Build editor view
        show_cols = [
            "housing_id","electronics_id","status_bucket","user","issue_tags",
            "housing_status","electronics_status","in_use",
            "current_location","exp_start_date","notes"
        ]
        show_cols = [c for c in show_cols if c in work.columns]
        view = work[["id"] + show_cols].copy()

        # Add selection checkbox (not persisted)
        view.insert(0, "select", False)

        # Use id as index so we can hide it but still know which rows changed
        view = view.set_index("id", drop=True)

        # Configure editor
        colcfg = {
                "select": st.column_config.CheckboxColumn("Select"),
                "in_use": st.column_config.CheckboxColumn("In use"),
                "user": st.column_config.SelectboxColumn("User", options=USERS),
                "housing_status": st.column_config.SelectboxColumn("Housing status", options=["Working","Broken","Unknown"]),
                "electronics_status": st.column_config.SelectboxColumn("Electronics status", options=["Working","Broken","Unknown"]),
                "status_bucket": st.column_config.SelectboxColumn("Status (auto)", options=STATUS_OPTIONS, disabled=True),
                # was DatetimeColumn → use DateColumn
                "exp_start_date": st.column_config.DateColumn("Exp start", format="YYYY-MM-DD"),
            }
        # Coerce dtypes the editor expects
        if "in_use" in view.columns:
            view["in_use"] = view["in_use"].fillna(False).astype(bool)
        
        if "exp_start_date" in view.columns:
            # convert strings/None to datetime64[ns] (NaT on bad values)
            view["exp_start_date"] = pd.to_datetime(view["exp_start_date"], errors="coerce")

        edited = st.data_editor(
            view,
            hide_index=not st.session_state.get("show_ids", False),
            column_config=colcfg,
            width="stretch",
            num_rows="fixed",
            key="devices_editor",
        )

        # --- Bulk edit selected rows ---
with st.expander("Bulk edit selected rows"):
    c1, c2, c3 = st.columns(3)
    be_user = c1.selectbox("Set user", ["(no change)"] + USERS, index=0, key="overview_be_user")
    be_hs   = c2.selectbox("Set housing status", ["(no change)", "Working", "Broken", "Unknown"], index=0, key="overview_be_hs")
    be_es   = c3.selectbox("Set electronics status", ["(no change)", "Working", "Broken", "Unknown"], index=0, key="overview_be_es")

    c4, c5, c6 = st.columns(3)
    be_inuse = c4.selectbox("Set In use", ["(no change)", "True", "False"], index=0, key="overview_be_inuse")
    be_loc   = c5.text_input("Set location (leave blank = no change)", key="overview_be_loc")
    be_issue = c6.selectbox("Set issue tag", ["(no change)"] + ISSUE_OPTIONS, index=0, key="overview_be_issue")

    be_notes = st.text_input("Append to notes (added at end; leave blank = no change)", key="overview_be_notes")

    if st.button("Apply to selected", key="overview_apply_selected"):
        
        # Which rows are checked?
        selected_ids = [rid for rid, row in edited.iterrows() if bool(row.get("select"))]
        if not selected_ids:
            st.warning("Select at least one row with the checkbox column.")
        else:
            df_all = get_table_df("devices")
            if df_all.empty:
                st.warning("No devices in database.")
            else:
                # Work only on rows that are currently selected
                target = df_all[df_all["id"].isin(selected_ids)]
                updated = 0
                for _, before in target.iterrows():
                    updates = {}

                    # apply field changes if requested
                    if be_user != "(no change)":
                        updates["user"] = normalize_user_val(be_user)
                    if be_hs != "(no change)":
                        updates["housing_status"] = None if be_hs == "Unknown" else be_hs
                    if be_es != "(no change)":
                        updates["electronics_status"] = None if be_es == "Unknown" else be_es
                    if be_inuse != "(no change)":
                        updates["in_use"] = (be_inuse == "True")
                    if be_loc.strip() != "":
                        updates["current_location"] = be_loc.strip()
                    if be_issue != "(no change)":
                        updates["issue_tags"] = be_issue
                    if be_notes.strip():
                        old_notes = before.get("notes") or ""
                        sep = " | " if old_notes else ""
                        updates["notes"] = f"{old_notes}{sep}{be_notes.strip()}"

                    if not updates:
                        continue

                    # Recompute status_bucket using the would-be record
                    probe = {**before.to_dict(), **updates}
                    probe["user"] = normalize_user_val(probe.get("user"))
                    probe["in_use"] = coerce_bool(probe.get("in_use"))
                    updates["status_bucket"] = compute_bucket(probe)

                    # Write to DB and log
                    sb.table("devices").update(updates).eq("id", int(before["id"])).execute()
                    log_action(
                        actor, "bulk_edit",
                        housing_id=probe.get("housing_id"),
                        electronics_id=probe.get("electronics_id"),
                        details=f"fields={list(updates.keys())}"
                    )
                    updated += 1

                if updated == 0:
                    st.info("Nothing to update.")
                else:
                    st.success(f"Updated {updated} row(s).")
                    st.rerun()


        # Apply changes to selected rows
        colA, colB = st.columns([1,1])
        if colA.button("Save changes to selected"):
            # compare old vs new for selected rows
            changed_count = 0
            for rid, row in edited.iterrows():
                if not bool(row.get("select")):
                    continue

                before = view.loc[rid]
                updates = {}
                for c in DEVICE_FIELDS_EDITABLE:
                    if c not in edited.columns:
                        continue
                    newv = row.get(c)
                    oldv = before.get(c)
                    # normalize types
                    if c == "in_use":
                        newv = coerce_bool(newv)
                        oldv = coerce_bool(oldv)
                    if c == "user":
                        newv = normalize_user_val(newv)
                        oldv = normalize_user_val(oldv)
                    if c == "exp_start_date" and pd.notna(newv):
                        try:
                            newv = pd.to_datetime(newv).to_pydatetime().isoformat()
                        except Exception:
                            newv = str(newv)

                    if (pd.isna(oldv) and pd.notna(newv)) or (pd.isna(newv) and pd.notna(oldv)) or (newv != oldv):
                        updates[c] = newv if (newv is not None and not (isinstance(newv, float) and pd.isna(newv))) else None

                if updates:
                    # recompute status bucket if any relevant fields changed
                    probe = {**before.to_dict(), **updates}
                    probe["user"] = normalize_user_val(probe.get("user"))
                    probe["in_use"] = coerce_bool(probe.get("in_use"))
                    updates["status_bucket"] = compute_bucket(probe)

                    sb.table("devices").update(updates).eq("id", rid).execute()
                    changed_count += 1

                    log_action(
                        actor, "inline_update",
                        housing_id=probe.get("housing_id"),
                        electronics_id=probe.get("electronics_id"),
                        details=f"updated fields: {', '.join(updates.keys())}"
                    )

            if changed_count == 0:
                st.info("No selected rows had changes.")
            else:
                st.success(f"Saved {changed_count} row(s).")
                st.rerun()

        if colB.button("Delete selected"):
            ids = [rid for rid, row in edited.iterrows() if bool(row.get("select"))]
            if not ids:
                st.warning("Select at least one row to delete.")
            else:
                # Log and delete
                for rid in ids:
                    try:
                        rec = sb.table("devices").select("housing_id,electronics_id").eq("id", rid).single().execute().data
                    except Exception:
                        rec = {}
                    log_action(actor, "delete_device", rec.get("housing_id"), rec.get("electronics_id"), details=f"id={rid}")
                sb.table("devices").delete().in_("id", ids).execute()
                st.success(f"Deleted {len(ids)} row(s).")
                st.rerun()

with st.expander("Archive selected"):
    c1, c2, c3 = st.columns(3)
    arch_mode = c1.selectbox(
        "Archive what?",
        ["Housing only", "Electronics only", "Both (pair)"],
        key="ov_arch_mode"
    )
    arch_reason = c2.text_input("Reason / note (optional)", key="ov_arch_reason")
    confirm_txt = c3.text_input("Type ARCHIVE to confirm", key="ov_arch_confirm")

    if st.button("Archive", key="ov_arch_btn"):
        if confirm_txt.strip().upper() != "ARCHIVE":
            st.warning("Please type ARCHIVE to confirm.")
        else:
            # Which rows are checked in the editor?
            selected_ids = [rid for rid, row in edited.iterrows() if bool(row.get("select"))]
            if not selected_ids:
                st.warning("Select at least one row above.")
            else:
                df_all = get_table_df("devices")
                target = df_all[df_all["id"].isin(selected_ids)]
                count = 0
                for _, r in target.iterrows():
                    hid = r.get("housing_id")
                    eid = r.get("electronics_id")
                    if arch_mode == "Housing only" and hid:
                        archive_one(hid, None, arch_reason, actor)
                        count += 1
                    elif arch_mode == "Electronics only" and eid:
                        archive_one(None, eid, arch_reason, actor)
                        count += 1
                    elif arch_mode == "Both (pair)":
                        # Will store both ids in a single archive row
                        archive_one(hid if pd.notna(hid) else None, eid if pd.notna(eid) else None, arch_reason, actor)
                        count += 1
                if count:
                    st.success(f"Archived {count} selection(s).")
                    st.rerun()
                else:
                    st.info("Nothing archived (missing IDs?).")

        


        st.write("---")
        st.markdown("**Archive selected**")
        
        cA, cB, cC = st.columns(3)
        arch_mode = cA.selectbox(
            "What to archive?",
            ["(nothing)", "Housing only", "Electronics only", "Both"],
            index=0,
            key="ov_arch_mode"
        )
        arch_note = cB.text_input("Archive note (optional)", key="ov_arch_note")
        
        if cC.button("Archive now", key="ov_arch_btn"):
            ids = [rid for rid, row in edited.iterrows() if bool(row.get("select"))]
            if not ids:
                st.warning("Select at least one row first.")
            else:
                picked = get_table_df("devices")
                picked = picked[picked["id"].isin(ids)].to_dict(orient="records")
                ah = (arch_mode == "Housing only") or (arch_mode == "Both")
                ae = (arch_mode == "Electronics only") or (arch_mode == "Both")
                if not (ah or ae):
                    st.info("Archive mode is '(nothing)'.")
                else:
                    archive_selected_devices(picked, archive_housing=ah, archive_electronics=ae, actor=actor, note=arch_note)
                    st.success(f"Archived {len(picked)} device(s).")
                    st.rerun()


# -------------------------------
# My FEDs (researcher view)
# -------------------------------
with tab_mine:
    st.subheader("My FEDs")

    # Pick researcher
    me = st.selectbox("Researcher", USERS, index=USERS.index("Emma") if "Emma" in USERS else 0, key="mine_user")
    me_norm = normalize_user_val(me)  # None if Unassigned, else string

    df_all = get_table_df("devices")
    if df_all.empty:
        st.info("No devices yet.")
    else:
        # Ensure columns exist
        for c in ["status_bucket","user","issue_tags","housing_status","electronics_status",
                  "current_location","exp_start_date","notes","in_use","housing_id","electronics_id"]:
            if c not in df_all.columns:
                df_all[c] = pd.Series(dtype="object")

        # --- My currently allocated devices (In Use + user==me) ---
        mine_mask = (df_all["status_bucket"] == "In Use") & (df_all["user"] == (me_norm if me_norm else None))
        mine = df_all[mine_mask].copy()

        st.markdown("#### Your current FEDs")
        if mine.empty:
            st.caption("You don’t have any allocated FEDs yet.")
        else:
            # Build editor like Overview
            cols = [
                "housing_id","electronics_id","status_bucket","user","issue_tags",
                "housing_status","electronics_status","in_use",
                "current_location","exp_start_date","notes"
            ]
            cols = [c for c in cols if c in mine.columns]
            view_mine = mine[["id"] + cols].copy()
            view_mine.insert(0, "select", False)
            view_mine = view_mine.set_index("id", drop=True)

            colcfg_mine = {
                "select": st.column_config.CheckboxColumn("Select"),
                "in_use": st.column_config.CheckboxColumn("In use"),
                "user": st.column_config.SelectboxColumn("User", options=USERS),
                "housing_status": st.column_config.SelectboxColumn("Housing status", options=["Working","Broken","Unknown"]),
                "electronics_status": st.column_config.SelectboxColumn("Electronics status", options=["Working","Broken","Unknown"]),
                "status_bucket": st.column_config.SelectboxColumn("Status (auto)", options=STATUS_OPTIONS, disabled=True),
                # was DatetimeColumn → use DateColumn
                "exp_start_date": st.column_config.DateColumn("Exp start", format="YYYY-MM-DD"),
            }

           # Coerce dtypes the editor expects (use view_mine, not view)
            if "in_use" in view_mine.columns:
                view_mine["in_use"] = view_mine["in_use"].fillna(False).astype(bool)
            
            if "exp_start_date" in view_mine.columns:
                view_mine["exp_start_date"] = pd.to_datetime(view_mine["exp_start_date"], errors="coerce")

            
            edited_mine = st.data_editor(
                view_mine,
                hide_index=not st.session_state.get("show_ids", False),
                column_config=colcfg_mine,
                width="stretch",
                num_rows="fixed",
                key="devices_editor_mine",   # <-- unique
            )

            # ---- Save inline edits for selected rows
            colA, colB = st.columns([1,1])
            if colA.button("Save changes to selected (My FEDs)", key="myfeds_save_selected"):
                changed_count = 0
                for rid, row in edited_mine.iterrows():
                    if not bool(row.get("select")):
                        continue
                    before = view_mine.loc[rid]
                    updates = {}
                    for c in DEVICE_FIELDS_EDITABLE:
                        if c not in edited_mine.columns:
                            continue
                        newv = row.get(c)
                        oldv = before.get(c)
                        if c == "in_use":
                            newv = coerce_bool(newv); oldv = coerce_bool(oldv)
                        if c == "user":
                            newv = normalize_user_val(newv); oldv = normalize_user_val(oldv)
                        if c == "exp_start_date" and pd.notna(newv):
                            try:
                                newv = pd.to_datetime(newv).to_pydatetime().isoformat()
                            except Exception:
                                newv = str(newv)

                        # detect change
                        if (pd.isna(oldv) and pd.notna(newv)) or (pd.isna(newv) and pd.notna(oldv)) or (newv != oldv):
                            updates[c] = newv if (newv is not None and not (isinstance(newv, float) and pd.isna(newv))) else None

                    if updates:
                        probe = {**before.to_dict(), **updates}
                        probe["user"] = normalize_user_val(probe.get("user"))
                        probe["in_use"] = coerce_bool(probe.get("in_use"))
                        updates["status_bucket"] = compute_bucket(probe)

                        sb.table("devices").update(updates).eq("id", rid).execute()
                        changed_count += 1
                        log_action(
                            actor, "inline_update_mine",
                            housing_id=probe.get("housing_id"),
                            electronics_id=probe.get("electronics_id"),
                            details=f"updated fields: {', '.join(updates.keys())}"
                        )

                if changed_count == 0:
                    st.info("No selected rows had changes.")
                else:
                    st.success(f"Saved {changed_count} row(s).")
                    st.rerun()

            # ---- Request maintenance for selected
            with st.expander("Request maintenance for selected"):
                c1, c2, c3 = st.columns(3)
                issue_sel = c1.selectbox("Issue", ISSUE_OPTIONS, key="myfeds_issue")
                set_hs    = c2.selectbox("Set housing status", ["(no change)", "Working", "Broken", "Unknown"], index=0, key="myfeds_set_hs")
                set_es    = c3.selectbox("Set electronics status", ["(no change)", "Working", "Broken", "Unknown"], index=0, key="myfeds_set_es")
                note_add  = st.text_input("Maintenance note (append)", key="myfeds_note_add")


            with st.expander("Request maintenance for selected"):
                c1, c2, c3 = st.columns(3)
                issue_sel = c1.selectbox("Issue", ISSUE_OPTIONS, key="mine_maint_issue")
                set_hs = c2.selectbox("Set housing status", ["(no change)", "Working", "Broken", "Unknown"], index=0, key="mine_maint_hs")
                set_es = c3.selectbox("Set electronics status", ["(no change)", "Working", "Broken", "Unknown"], index=0, key="mine_maint_es")
                note_add = st.text_input("Maintenance note (append)", key="mine_maint_note")
            
                a1, a2 = st.columns(2)
                arch_choice = a1.selectbox(
                    "Also archive… (optional)",
                    ["No archive", "Housing only", "Electronics only", "Both (pair)"],
                    index=0,
                    key="mine_maint_archive_choice"
                )
                arch_reason = a2.text_input("Archive reason (optional)", key="mine_maint_archive_reason")
            
                if st.button("Submit maintenance request", key="mine_maint_submit"):
                    ids = [rid for rid, row in edited_mine.iterrows() if bool(row.get("select"))]
                    if not ids:
                        st.warning("Select at least one row above.")
                    else:
                        updated = 0
                        for rid in ids:
                            row = mine[mine["id"] == rid].iloc[0]
                            up = {
                                "issue_tags": issue_sel,
                                "in_use": False,
                                "user": None,
                                "status_bucket": "To Test",
                            }
                            if set_hs != "(no change)":
                                up["housing_status"] = None if set_hs == "Unknown" else set_hs
                            if set_es != "(no change)":
                                up["electronics_status"] = None if set_es == "Unknown" else set_es
                            if note_add.strip():
                                old = row.get("notes") or ""
                                sep = " | " if old else ""
                                up["notes"] = f"{old}{sep}{note_add.strip()}"
            
                            sb.table("devices").update(up).eq("id", int(rid)).execute()
                            log_action(
                                actor, "request_maintenance",
                                housing_id=row.get("housing_id"),
                                electronics_id=row.get("electronics_id"),
                                details=f"{issue_sel}"
                            )
            
                            # Optional archive alongside maintenance
                            hid = row.get("housing_id"); eid = row.get("electronics_id")
                            if arch_choice == "Housing only" and pd.notna(hid):
                                archive_one(hid, None, arch_reason, actor)
                            elif arch_choice == "Electronics only" and pd.notna(eid):
                                archive_one(None, eid, arch_reason, actor)
                            elif arch_choice == "Both (pair)":
                                archive_one(hid if pd.notna(hid) else None, eid if pd.notna(eid) else None, arch_reason, actor)
            
                            updated += 1
            
                        st.success(f"Submitted maintenance for {updated} device(s).")
                        st.rerun()



                
                if st.button("Submit maintenance request", key="myfeds_submit_maint"):
                    ids = [rid for rid, row in edited_mine.iterrows() if bool(row.get("select"))]
                    if not ids:
                        st.warning("Select at least one row above.")
                    else:
                        updated = 0
                        for rid in ids:
                            row = mine[mine["id"] == rid].iloc[0]
                            up = {
                                "issue_tags": issue_sel,
                                "in_use": False,
                                "user": None,
                                "status_bucket": "To Test",
                            }
                            if set_hs != "(no change)":
                                up["housing_status"] = None if set_hs == "Unknown" else set_hs
                            if set_es != "(no change)":
                                up["electronics_status"] = None if set_es == "Unknown" else set_es
                            if note_add.strip():
                                old = row.get("notes") or ""
                                sep = " | " if old else ""
                                up["notes"] = f"{old}{sep}{note_add.strip()}"

                            sb.table("devices").update(up).eq("id", int(rid)).execute()
                            log_action(
                                actor, "request_maintenance",
                                housing_id=row.get("housing_id"),
                                electronics_id=row.get("electronics_id"),
                                details=f"{issue_sel}"
                            )
                            updated += 1
                        st.success(f"Submitted maintenance for {updated} device(s).")
                        st.rerun()

            st.write("---")
            st.markdown("**Archive selected (My FEDs)**")
            
            m1, m2, m3 = st.columns(3)
            arch_mode_m = m1.selectbox(
                "What to archive?",
                ["(nothing)", "Housing only", "Electronics only", "Both"],
                index=0,
                key="mine_arch_mode"
            )
            arch_note_m = m2.text_input("Archive note (optional)", key="mine_arch_note")
            
            if m3.button("Archive selected", key="mine_arch_btn"):
                ids = [rid for rid, row in edited_mine.iterrows() if bool(row.get("select"))]
                if not ids:
                    st.warning("Select at least one row above.")
                else:
                    picked = mine[mine["id"].isin(ids)].to_dict(orient="records")
                    ah = (arch_mode_m == "Housing only") or (arch_mode_m == "Both")
                    ae = (arch_mode_m == "Electronics only") or (arch_mode_m == "Both")
                    if not (ah or ae):
                        st.info("Archive mode is '(nothing)'.")
                    else:
                        archive_selected_devices(picked, archive_housing=ah, archive_electronics=ae, actor=actor, note=arch_note_m)
                        st.success(f"Archived {len(picked)} device(s).")
                        st.rerun()
            
            


            
            # ---- Quick history peek for a selected single row
            with st.expander("View history for one of your devices"):
                # pick from your devices
                hid_opt = [""] + sorted(mine["housing_id"].dropna().unique().tolist())
                eid_opt = [""] + sorted(mine["electronics_id"].dropna().unique().tolist())
                hh = st.selectbox("housing_id", hid_opt, key="mine_hist_h")
                ee = st.selectbox("electronics_id", eid_opt, key="mine_hist_e")
                if hh or ee:
                    q = sb.table("actions").select("*")
                    if hh: q = q.eq("housing_id", hh)
                    if ee: q = q.eq("electronics_id", ee)
                    acts = q.order("ts", desc=True).execute().data or []
                    hist = pd.DataFrame(acts)
                    if hist.empty:
                        st.info("No history yet.")
                    else:
                        if not st.session_state.get("show_ids", False) and "id" in hist.columns:
                            hist = hist.drop(columns=["id"])
                        st.dataframe(hist, width="stretch")

        # --- Request/assign new FEDs from Ready to yourself
        st.markdown("#### Request FEDs from Ready pool")
        ready = df_all[df_all["status_bucket"] == "Ready for Use"].copy()
        if ready.empty:
            st.caption("No ready devices available.")
        else:
            ready_view = ready[["id","housing_id","electronics_id","current_location","notes"]].copy()
            ready_view.insert(0, "select", False)
            ready_view = ready_view.set_index("id", drop=True)

            ready_editor = st.data_editor(
                ready_view,
                hide_index=not st.session_state.get("show_ids", False),
                width="stretch",
                num_rows="fixed",
                key="mine_ready_editor",
                column_config={"select": st.column_config.CheckboxColumn("Select")}
            )

            c1, c2 = st.columns([2,1])
            exp_date = c1.date_input("Experiment start date", value=None, key="mine_exp_date")
            if c2.button("Assign selected to me"):
                ids = [rid for rid, row in ready_editor.iterrows() if bool(row.get("select"))]
                if not ids:
                    st.warning("Select at least one Ready device.")
                else:
                    exp_iso = None
                    if exp_date:
                        try:
                            exp_iso = pd.to_datetime(exp_date).to_pydatetime().isoformat()
                        except Exception:
                            exp_iso = None
                    up = {
                        "user": me_norm,
                        "in_use": True,
                        "status_bucket": "In Use",
                        "exp_start_date": exp_iso
                    }
                    sb.table("devices").update(up).in_("id", ids).execute()
                    # log each
                    picked = ready[ready["id"].isin(ids)].to_dict(orient="records")
                    for r in picked:
                        log_action(
                            actor, "allocate_to_self",
                            housing_id=r.get("housing_id"),
                            electronics_id=r.get("electronics_id"),
                            details=f"user={me_norm}"
                        )
                    st.success(f"Assigned {len(ids)} device(s) to {me}.")
                    st.rerun()

# -------------------------------
# Add Device
# -------------------------------
# -------------------------------
# Add Device
# -------------------------------
with tab_add:
    st.subheader("Add a device")
    c1, c2 = st.columns(2)
    with c1:
        housing_id = st.text_input("Housing ID (e.g., H36)")
        housing_status = st.selectbox("Housing status", ["Working","Broken","Unknown"], index=0)
        current_location = st.text_input("Location")
        notes = st.text_input("Notes")
    with c2:
        electronics_id = st.text_input("Electronics ID (e.g., E36)")
        electronics_status = st.selectbox("Electronics status", ["Working","Broken","Unknown"], index=0)
        user_val = st.selectbox("User", USERS, index=USERS.index("Unassigned"))
        in_use = st.checkbox("In use", value=False)
        exp_start = st.date_input("Experiment start (optional)", value=None)

    if st.button("Create device"):
        # Convert date picker first
        exp_iso = None
        if exp_start:
            try:
                exp_iso = pd.to_datetime(exp_start).to_pydatetime().isoformat()
            except Exception:
                exp_iso = None

        rec = {
            "housing_id": housing_id or None,
            "electronics_id": electronics_id or None,
            "housing_status": housing_status if housing_status != "Unknown" else None,
            "electronics_status": electronics_status if electronics_status != "Unknown" else None,
            "current_location": current_location or None,
            "notes": notes or None,
            "user": normalize_user_val(user_val),
            "in_use": bool(in_use),
            "exp_start_date": exp_iso,
        }

        # Normalize IDs / status and compute status_bucket
        rec = normalize_device(rec)

        # Make absolutely sure we never send an id to Supabase here
        rec.pop("id", None)

        try:
            if rec["housing_id"]:
                sb.table("devices").upsert(rec, on_conflict="housing_id").execute()
            elif rec["electronics_id"]:
                sb.table("devices").upsert(rec, on_conflict="electronics_id").execute()
            else:
                st.warning("Provide at least one of housing_id or electronics_id.")
                st.stop()

            log_action(
                actor, "create_device",
                rec.get("housing_id"), rec.get("electronics_id"),
                details=f"status={rec['status_bucket']}"
            )
            st.success("Device created.")
            st.rerun()
        except APIError as e:
            # Show the underlying Supabase error message
            msg = getattr(e, "message", None) or str(e)
            st.error(f"Could not create device: {msg}")

    # -----------------------------------------
    # 2) NEW DEVICE: CREATE NEW IDS INTO INVENTORY
    # -----------------------------------------
    st.markdown("### New device (create new housing/electronics IDs)")

    c1, c2 = st.columns(2)
    with c1:
        new_hid = st.text_input("New housing ID (e.g., H50)", key="new_hid")
        new_h_status = st.selectbox("Housing status", ["Working","Broken","Unknown"], index=0, key="new_h_status")
        new_h_notes = st.text_input("Housing notes", key="new_h_notes")
    with c2:
        new_eid = st.text_input("New electronics ID (e.g., E50)", key="new_eid")
        new_e_status = st.selectbox("Electronics status", ["Working","Broken","Unknown"], index=0, key="new_e_status")
        new_e_notes = st.text_input("Electronics notes", key="new_e_notes")

    if st.button("Save new IDs to inventory", key="new_ids_btn"):
        if not new_hid and not new_eid:
            st.warning("Enter at least a housing ID or an electronics ID.")
        else:
            try:
                # New housing → inventory_housing
                if new_hid:
                    hid = clean_id(new_hid)
                    h_status = None if new_h_status == "Unknown" else new_h_status
                    sb.table("inventory_housing").upsert(
                        {
                            "housing_id": hid,
                            "status": h_status,
                            "notes": new_h_notes or None,
                        },
                        on_conflict="housing_id",
                    ).execute()

                # New electronics → inventory_electronics
                if new_eid:
                    eid = clean_id(new_eid)
                    e_status = None if new_e_status == "Unknown" else new_e_status
                    sb.table("inventory_electronics").upsert(
                        {
                            "electronics_id": eid,
                            "status": e_status,
                            "notes": new_e_notes or None,
                        },
                        on_conflict="electronics_id",
                    ).execute()

                st.success("New IDs saved to inventory. You can now combine them into a FED in the section above.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not save to inventory: {e}")

# -------------------------------
# History
# -------------------------------

with tab_history:
    st.subheader("Device history")
    df_all = get_table_df("devices")
    if df_all.empty:
        st.info("No devices yet.")
    else:
        col1, col2 = st.columns(2)
        hid = col1.selectbox("housing_id", [""] + sorted(df_all["housing_id"].dropna().unique().tolist()))
        eid = col2.selectbox("electronics_id", [""] + sorted(df_all["electronics_id"].dropna().unique().tolist()))
        q = sb.table("actions").select("*")
        if hid:
            q = q.eq("housing_id", hid)
        if eid:
            q = q.eq("electronics_id", eid)
        acts = q.order("ts", desc=True).execute().data or []
        hist = pd.DataFrame(acts)
        if hist.empty:
            st.info("No history yet.")
        else:
            # Hide id unless toggled
            if not st.session_state.get("show_ids", False) and "id" in hist.columns:
                hist = hist.drop(columns=["id"])
            st.dataframe(hist, width="stretch")


    # --- Archive viewer (new) ---
st.write("---")
st.subheader("Archive")

arch = sb.table("archive").select("*").order("created_at", desc=True).execute().data or []
arch_df = pd.DataFrame(arch)

# quick filters
colA, colB = st.columns(2)
f_h = colA.text_input("Filter by housing_id (contains)", key="arch_filter_h")
f_e = colB.text_input("Filter by electronics_id (contains)", key="arch_filter_e")

show_arch = arch_df.copy()
for col, val in (("housing_id", f_h), ("electronics_id", f_e)):
    if val:
        show_arch = show_arch[show_arch[col].fillna("").str.contains(val, case=False, na=False)]

# Brief table
brief_cols = [c for c in ["id","created_at","housing_id","electronics_id","reason"] if c in show_arch.columns]
st.dataframe(show_arch[brief_cols] if brief_cols else show_arch, use_container_width=True)

# Drill-in: pick an archive row to inspect snapshots
arch_ids = show_arch["id"].tolist() if "id" in show_arch.columns else []
chosen_arch = st.selectbox("View snapshot for archive id", [""] + [str(x) for x in arch_ids], key="arch_pick")
if chosen_arch:
    chosen_arch = int(chosen_arch)
    rec = show_arch[show_arch["id"] == chosen_arch].iloc[0].to_dict()
    st.markdown("**Device snapshot (at archive time):**")
    st.json(rec.get("device_snapshot") or {})
    st.markdown("**Actions snapshot (at archive time):**")
    st.json(rec.get("actions_snapshot") or [])

# -------------------------------
# Inventory (General + Electronics + Housing)
# -------------------------------
with tab_inventory:
    st.subheader("Inventory")

    # ---- helpers (local to this tab) ----
    def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
        """Return only the columns that exist in df (for safe slicing)."""
        return [c for c in cols if c in df.columns]

    def _ensure_cols(df: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
        """Make sure df has these columns with safe dtypes."""
        for col, dtype in spec.items():
            if col not in df.columns:
                df[col] = pd.Series(dtype=dtype)
        return df

    def _bool(v):
        if isinstance(v, bool): return v
        if v is None or (isinstance(v, float) and pd.isna(v)): return False
        return str(v).strip().lower() in ("1","true","yes","y")

    # ===============================================================
    # A) GENERAL INVENTORY  (table: inventory)
    # ===============================================================
    st.markdown("### General Inventory")

    inv = get_table_df("inventory")
    inv = _ensure_cols(inv, {"id": "Int64", "item": "object", "qty": "float", "created_at": "object"})

    # Build editor view (add a selection checkbox)
    view = inv.copy()
    view.insert(0, "select", False)

    cfg = {
        "select": st.column_config.CheckboxColumn("Select"),
        "item": st.column_config.TextColumn("Item"),
        "qty": st.column_config.NumberColumn("Quantity", step=1.0, format="%.2f"),
        # show created_at if present; if not, it's silently omitted by _present()
        "created_at": st.column_config.TextColumn("Created"),
    }

    cols_general = _present(view, ["select", "id", "item", "qty", "created_at"])
    edited = st.data_editor(
        view[cols_general],
        hide_index=True,
        column_config=cfg,
        width="stretch",
        num_rows="dynamic",
        key="gen_inventory_editor",
    )

    cA, cB, cC = st.columns([1,1,2])
    if cA.button("Save selected (General)"):
        ids = []
        inserts = []
        for _, row in edited.iterrows():
            if not _bool(row.get("select")):
                continue
            rid = row.get("id")
            item = (row.get("item") or "").strip() or None
            qty  = float(row.get("qty") or 0)
            if pd.notna(rid):
                # update by id
                sb.table("inventory").update({"item": item, "qty": qty}).eq("id", int(rid)).execute()
            else:
                # insert new
                inserts.append({"item": item, "qty": qty})
        if inserts:
            sb.table("inventory").insert(inserts).execute()
        st.success("Saved.")
        st.rerun()

    if cB.button("Delete selected (General)"):
        ids = [int(r.get("id")) for _, r in edited.iterrows() if _bool(r.get("select")) and pd.notna(r.get("id"))]
        if ids:
            sb.table("inventory").delete().in_("id", ids).execute()
            st.success(f"Deleted {len(ids)} row(s).")
            st.rerun()
        else:
            st.info("Nothing selected with a saved ID to delete.")

    st.divider()

    # ===============================================================
    # B) ELECTRONICS INVENTORY  (table: inventory_electronics)
    # ===============================================================
    st.markdown("### Electronics Inventory")

    inv_e = get_table_df("inventory_electronics")
    inv_e = _ensure_cols(inv_e, {
        "id": "Int64",
        "electronics_id": "object",
        "status": "object",
        "notes": "object",
        "created_at": "object",
    })

    v_e = inv_e.copy()
    v_e.insert(0, "select", False)

    status_opts = ["Working", "Broken", "Unknown"]

    cfg_e = {
        "select": st.column_config.CheckboxColumn("Select"),
        "electronics_id": st.column_config.TextColumn("Electronics ID (optional)"),
        "status": st.column_config.SelectboxColumn("Status", options=status_opts),
        "notes": st.column_config.TextColumn("Notes"),
        "created_at": st.column_config.TextColumn("Created"),
    }

    cols_e = _present(v_e, ["select", "id", "electronics_id", "status", "notes", "created_at"])
    eedit = st.data_editor(
        v_e[cols_e],
        hide_index=True,
        column_config=cfg_e,
        width="stretch",
        num_rows="dynamic",
        key="inv_elec_editor",
    )

    ec1, ec2, ec3 = st.columns([1,1,2])
    if ec1.button("Save selected (Electronics)"):
        inserts = []
        for _, row in eedit.iterrows():
            if not _bool(row.get("select")):
                continue
            rid = row.get("id")
            eid = (row.get("electronics_id") or "").strip() or None
            stt = (row.get("status") or "").strip().title() or None
            nts = (row.get("notes") or "").strip() or None
            rec = {"electronics_id": eid, "status": stt, "notes": nts}
            if pd.notna(rid):
                sb.table("inventory_electronics").update(rec).eq("id", int(rid)).execute()
            else:
                inserts.append(rec)
        if inserts:
            sb.table("inventory_electronics").insert(inserts).execute()
        st.success("Saved.")
        st.rerun()

    if ec2.button("Delete selected (Electronics)"):
        ids = [int(r.get("id")) for _, r in eedit.iterrows() if _bool(r.get("select")) and pd.notna(r.get("id"))]
        if ids:
            sb.table("inventory_electronics").delete().in_("id", ids).execute()
            st.success(f"Deleted {len(ids)} row(s).")
            st.rerun()
        else:
            st.info("Nothing selected with a saved ID to delete.")

    st.divider()

    # ===============================================================
    # C) HOUSING INVENTORY  (table: inventory_housing)
    # ===============================================================
    st.markdown("### Housing Inventory")

    inv_h = get_table_df("inventory_housing")
    inv_h = _ensure_cols(inv_h, {
        "id": "Int64",
        "housing_id": "object",
        "status": "object",
        "notes": "object",
        "created_at": "object",
    })

    v_h = inv_h.copy()
    v_h.insert(0, "select", False)

    cfg_h = {
        "select": st.column_config.CheckboxColumn("Select"),
        "housing_id": st.column_config.TextColumn("Housing ID (optional)"),
        "status": st.column_config.SelectboxColumn("Status", options=status_opts),
        "notes": st.column_config.TextColumn("Notes"),
        "created_at": st.column_config.TextColumn("Created"),
    }

    cols_h = _present(v_h, ["select", "id", "housing_id", "status", "notes", "created_at"])
    hedit = st.data_editor(
        v_h[cols_h],
        hide_index=True,
        column_config=cfg_h,
        width="stretch",
        num_rows="dynamic",
        key="inv_housing_editor",
    )

    hc1, hc2, hc3 = st.columns([1,1,2])
    if hc1.button("Save selected (Housing)"):
        inserts = []
        for _, row in hedit.iterrows():
            if not _bool(row.get("select")):
                continue
            rid = row.get("id")
            hid = (row.get("housing_id") or "").strip() or None
            stt = (row.get("status") or "").strip().title() or None
            nts = (row.get("notes") or "").strip() or None
            rec = {"housing_id": hid, "status": stt, "notes": nts}
            if pd.notna(rid):
                sb.table("inventory_housing").update(rec).eq("id", int(rid)).execute()
            else:
                inserts.append(rec)
        if inserts:
            sb.table("inventory_housing").insert(inserts).execute()
        st.success("Saved.")
        st.rerun()

    if hc2.button("Delete selected (Housing)"):
        ids = [int(r.get("id")) for _, r in hedit.iterrows() if _bool(r.get("select")) and pd.notna(r.get("id"))]
        if ids:
            sb.table("inventory_housing").delete().in_("id", ids).execute()
            st.success(f"Deleted {len(ids)} row(s).")
            st.rerun()
        else:
            st.info("Nothing selected with a saved ID to delete.")


# -------------------------------
# Admin (only if admin_enabled)
# -------------------------------
with tab_admin:
    st.subheader("Admin")
    if not admin_enabled:
        st.caption("Enter a valid admin code in the sidebar to use admin tools.")
    else:
        st.warning("Destructive actions ahead.")
        col1, col2 = st.columns(2)
        if col1.button("Wipe ALL data (devices, inventory, actions)"):
            sb.rpc("exec", {"q":"truncate table actions restart identity cascade;"}).execute() if hasattr(sb, "rpc") else None
            sb.table("actions").delete().neq("id",-1).execute()
            sb.table("inventory").delete().neq("id",-1).execute()
            sb.table("devices").delete().neq("id",-1).execute()
            st.success("All data wiped.")
            st.rerun()

        # Export for sanity checks
        actions = get_table_df("actions")
        devices = get_table_df("devices")
        inv = get_table_df("inventory")
        with col2:
            if st.button("Download Excel snapshot"):
                bio = BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as w:
                    devices.to_excel(w, sheet_name="Devices", index=False)
                    inv.to_excel(w, sheet_name="Inventory", index=False)
                    actions.to_excel(w, sheet_name="Actions", index=False)
                bio.seek(0)
                st.download_button("Download", data=bio.getvalue(), file_name="FED3_DB_Snapshot.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
