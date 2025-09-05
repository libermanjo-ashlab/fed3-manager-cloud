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
tab_overview, tab_add, tab_history, tab_inventory, tab_admin = st.tabs(
    ["Overview (Search & Edit)", "Add Device", "History", "Inventory", "Admin"]
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
            "exp_start_date": st.column_config.DatetimeColumn("Exp start"),
        }

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
    be_user = c1.selectbox("Set user", ["(no change)"] + USERS, index=0)
    be_hs   = c2.selectbox("Set housing status", ["(no change)", "Working", "Broken", "Unknown"], index=0)
    be_es   = c3.selectbox("Set electronics status", ["(no change)", "Working", "Broken", "Unknown"], index=0)

    c4, c5, c6 = st.columns(3)
    be_inuse = c4.selectbox("Set In use", ["(no change)", "True", "False"], index=0)
    be_loc   = c5.text_input("Set location (leave blank = no change)")
    be_issue = c6.selectbox("Set issue tag", ["(no change)"] + ISSUE_OPTIONS, index=0)

    be_notes = st.text_input("Append to notes (added at end; leave blank = no change)")

    if st.button("Apply to selected"):
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

    rec = normalize_device(rec)

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
    except Exception as e:
        st.error(f"Could not save device: {e}")


        
        if exp_start:
            try:
                rec["exp_start_date"] = pd.to_datetime(exp_start).to_pydatetime().isoformat()
            except Exception:
                rec["exp_start_date"] = None

        rec["status_bucket"] = compute_bucket(rec)

        # Insert (unique constraints on housing_id / electronics_id handle duplicates)
        try:
            sb.table("devices").insert(rec).execute()
        except APIError as e:
            st.error(f"Insert failed: {getattr(e,'message',e)}")
        else:
            log_action(actor, "create_device", rec.get("housing_id"), rec.get("electronics_id"),
                       details=f"status={rec['status_bucket']}")
            st.success("Device created.")
            st.rerun()

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

# -------------------------------
# Inventory (optional)
# -------------------------------
with tab_inventory:
    st.subheader("Inventory")
    inv = get_table_df("inventory")
    if inv.empty:
        st.caption("No items yet.")
    else:
        if not st.session_state.get("show_ids", False) and "id" in inv.columns:
            inv = inv.drop(columns=["id"])
        st.dataframe(inv, width="stretch")

    st.write("Add / update an item")
    item = st.text_input("Item")
    qty = st.number_input("Quantity", value=0.0, step=1.0)
    cA, cB, cC = st.columns(3)
    if cA.button("Add / Update"):
        if not item:
            st.warning("Enter an item name.")
        else:
            ex = sb.table("inventory").select("id").eq("item", item).execute().data
            if not ex:
                sb.table("inventory").insert({"item": item, "qty": float(qty)}).execute()
                log_action("system", "inv_add", details=f"{item}={qty}")
            else:
                sb.table("inventory").update({"qty": float(qty)}).eq("item", item).execute()
                log_action("system", "inv_update", details=f"{item}={qty}")
            st.success("Inventory updated.")
            st.rerun()
    if cB.button("Delete"):
        if not item:
            st.warning("Enter an item name.")
        else:
            sb.table("inventory").delete().eq("item", item).execute()
            log_action("system", "inv_delete", details=item)
            st.success("Deleted.")
            st.rerun()

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
