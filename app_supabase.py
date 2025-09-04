import streamlit as st
import pandas as pd
from io import BytesIO
from supabase import create_client, Client
from postgrest.exceptions import APIError
from datetime import datetime, date
from collections import Counter



st.set_page_config(page_title="FED3 Manager — Minimal (Admin + History + Search)", layout="wide")

# -------------------------------
# Secrets / clients
# -------------------------------
@st.cache_resource
def get_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

sb = get_client()

ADMIN_CODE = st.secrets.get("ADMIN_CODE", None)  # set in Cloud → Settings → Secrets

# -------------------------------
# Constants
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

# Only these columns go into devices (whitelist)
DEVICE_FIELDS = [
    "housing_id", "housing_status",
    "electronics_id", "electronics_status",
    "status_in_lab", "status_with_mice",
    "in_use", "user",
    "current_location", "exp_start_date",
    "notes", "status_bucket", "issue_tags"
]

# -------------------------------
# Helpers
# -------------------------------
def norm(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def compute_bucket(row: dict) -> str:
    """Only mark In Use when explicit in_use==True.
    Otherwise decide from part statuses."""
    in_use = bool(row.get("in_use") or False)
    housing_working = str(row.get("housing_status") or "").strip().lower() == "working"
    board_working   = str(row.get("electronics_status") or "").strip().lower() == "working"
    has_housing = bool((row.get("housing_id") or "").strip())
    has_board   = bool((row.get("electronics_id") or "").strip())

    if in_use:
        return "In Use"
    if housing_working and board_working:
        return "Ready for Use"
    if (has_housing or has_board) and (not housing_working or not board_working):
        return "To Test"
    return "Unclear"

def df_to_dicts(df: pd.DataFrame):
    recs = df.to_dict(orient="records")
    out = []
    for r in recs:
        o = {}
        for k, v in r.items():
            if pd.isna(v):
                o[k] = None
            elif isinstance(v, (pd.Timestamp, datetime, date)):
                # ISO 8601 string; keep just date if no time info
                try:
                    # pandas Timestamp has .tz_convert sometimes; safest: to_pydatetime then isoformat()
                    o[k] = pd.to_datetime(v).to_pydatetime().isoformat()
                except Exception:
                    o[k] = str(v)
            else:
                o[k] = v
        out.append(o)
    return out

def filter_device_fields(rows):
    return [{k: r.get(k) for k in DEVICE_FIELDS if k in r} for r in rows]

def get_table_df(name: str) -> pd.DataFrame:
    data = sb.table(name).select("*").execute().data
    return pd.DataFrame(data) if data else pd.DataFrame()

def delete_all(name: str):
    sb.table(name).delete().neq("id", -1).execute()

def insert_rows(name: str, rows: list[dict], chunk: int = 500):
    for i in range(0, len(rows), chunk):
        part = rows[i:i+chunk]
        try:
            sb.table(name).insert(part).execute()
        except APIError as e:
            st.error(f"Supabase insert error on table '{name}': {getattr(e, 'message', e)}")
            if part:
                st.write("First row that failed:")
                st.json(part[0])
            raise

def update_rows(name: str, ids: list[int], updates: dict):
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

def ensure_tables_exist():
    try:
        _ = sb.table("devices").select("id").limit(1).execute()
        _ = sb.table("inventory").select("id").limit(1).execute()
        _ = sb.table("actions").select("id").limit(1).execute()
        return True
    except Exception:
        st.error("Tables not found. Run the SQL patch noted in the README/SQL file.")
        st.stop()

ensure_tables_exist()

def maybe_hide_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not st.session_state.get("show_ids", False) and "id" in df.columns:
        return df.drop(columns=["id"])
    return df

def get_history(housing_id: str | None = None, electronics_id: str | None = None) -> pd.DataFrame:
    """Timeline = current device snapshot + any actions (most recent first)."""
    # Fetch actions
    q = sb.table("actions").select("*")
    if housing_id:
        q = q.eq("housing_id", housing_id)
    if electronics_id:
        q = q.eq("electronics_id", electronics_id)
    acts = q.order("ts", desc=True).execute().data or []
    hist = pd.DataFrame(acts)

    # Current snapshot from devices
    cur = []
    if housing_id:
        cur = sb.table("devices").select("*").eq("housing_id", housing_id).limit(1).execute().data or []
    if not cur and electronics_id:
        cur = sb.table("devices").select("*").eq("electronics_id", electronics_id).limit(1).execute().data or []

    if cur:
        d = cur[0]
        snap = {
            "ts": None,
            "actor": "current",
            "action": "current_state",
            "details": f"status={d.get('status_bucket')}; user={d.get('user')}; "
                       f"loc={d.get('current_location')}; notes={d.get('notes')}",
            "housing_id": d.get("housing_id"),
            "electronics_id": d.get("electronics_id"),
            "id": None
        }
        hist = pd.concat([pd.DataFrame([snap]), hist], ignore_index=True) if not hist.empty else pd.DataFrame([snap])

    cols = [c for c in ["ts","actor","action","details","housing_id","electronics_id","id"] if c in hist.columns]
    return hist[cols] if not hist.empty else hist

# --- Key & compare helpers ---
KEY_FIELDS = ["housing_id", "electronics_id"]

def device_key(r: dict) -> str:
    # prefer housing_id; fall back to electronics_id
    h = (r.get("housing_id") or "").strip()
    e = (r.get("electronics_id") or "").strip()
    return f"H:{h}" if h else (f"E:{e}" if e else "")

def clean_for_compare(r: dict) -> dict:
    # normalize whitespace/case for comparison
    out = {}
    for k, v in r.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out[k] = None
        elif isinstance(v, str):
            out[k] = v.strip()
        else:
            out[k] = v
    return out

def dicts_indexed(dicts: list[dict]) -> dict:
    return {device_key(d): clean_for_compare(d) for d in dicts if device_key(d)}

    
    
# -------------------------------
# Sidebar (Admin mode + common)
# -------------------------------
st.title("FED3 Manager — Lab")

with st.sidebar:
    st.header("Controls")
    st.checkbox("Show internal IDs", value=False, key="show_ids")
    actor = st.text_input("Your name (for log)", value="lab-user")

    st.write("---")
    st.subheader("Admin")
    admin_enabled = False
    if ADMIN_CODE:
        code = st.text_input("Enter admin code", type="password")
        admin_enabled = (code == ADMIN_CODE)
        if not admin_enabled:
            st.caption("Upload & destructive actions are hidden unless the code matches.")
    else:
        st.caption("ADMIN_CODE not set in secrets; admin features are disabled.")

# Admin-only: initialize/refresh from workbook
if admin_enabled:
    st.write("Initialize / refresh from workbook")
    load_mode = st.radio("Load mode", ["Replace (match workbook)", "Merge (no deletes)"], horizontal=False)
    upl = st.file_uploader("Upload combined workbook (.xlsx)", type=["xlsx"])
    if upl is not None and st.button("Load workbook into database"):
        sheets = pd.read_excel(BytesIO(upl.read()), sheet_name=None, engine="openpyxl")

        # --- DEVICES ---
        devices = sheets.get("Master List")
        if devices is None:
            st.error("Workbook must include a sheet named 'Master List'.")
            st.stop()

        rename = {"status_(in_lab)": "status_in_lab", "status_(with_mice)": "status_with_mice"}
        devices = devices.rename(columns=rename)

        for c in ["in_use","user","housing_id","housing_status","electronics_id","electronics_status",
                  "status_in_lab","status_with_mice","current_location","exp_start_date","notes","issue_tags"]:
            if c not in devices.columns: devices[c] = None
            # normalize strings
            devices[c] = devices[c].apply(lambda x: None if (pd.isna(x) or str(x).strip()=="") else (str(x).strip() if isinstance(x,str) else x))

        def to_bool(x): return str(x).lower() in ("1","yes","true","y")
        devices["in_use"] = devices["in_use"].apply(to_bool)

        dicts = df_to_dicts(devices)
        # compute status_bucket with the stricter logic
        for r in dicts:
            r["status_bucket"] = compute_bucket(r)
        dicts = filter_device_fields(dicts)

        # INDEX BY KEY
        incoming = dicts_indexed(dicts)

        # fetch current DB
        existing_df = get_table_df("devices")
        existing = dicts_indexed(df_to_dicts(existing_df)) if not existing_df.empty else {}

        def _norm_id(x):
            if x is None:
                return None
            s = str(x).strip()
            return None if s == "" or s.lower() in {"nan","none","null"} else s

        def _dedupe_by(rows, key):
            # keep the LAST occurrence for each key
            m = {}
            keys = []
            for r in rows:
                k = _norm_id(r.get(key))
                if k:
                    r[key] = k
                    m[k] = r
                    keys.append(k)
            dup_keys = [k for k, c in Counter(keys).items() if c > 1]
            return list(m.values()), dup_keys

        to_upsert_h_all = [r for r in dicts if _norm_id(r.get("housing_id"))]
        to_upsert_e_all = [r for r in dicts if not _norm_id(r.get("housing_id")) and _norm_id(r.get("electronics_id"))]

        to_upsert_h, dup_h = _dedupe_by(to_upsert_h_all, "housing_id")
        to_upsert_e, dup_e = _dedupe_by(to_upsert_e_all, "electronics_id")

        if dup_h or dup_e:
            msg = []
            if dup_h: msg.append("housing: " + ", ".join(dup_h[:5]) + ("…" if len(dup_h)>5 else ""))
            if dup_e: msg.append("electronics: " + ", ".join(dup_e[:5]) + ("…" if len(dup_e)>5 else ""))
            st.warning("Deduped duplicate IDs in upload — " + " | ".join(msg))

        def safe_upsert(rows_h, rows_e):
            """Try ON CONFLICT; fallback to manual merge if anything fails."""
            try:
                if rows_h:
                    sb.table("devices").upsert(rows_h, on_conflict="housing_id").execute()
                if rows_e:
                    sb.table("devices").upsert(rows_e, on_conflict="electronics_id").execute()
                return True
            except Exception:
                # Manual merge (update if exists, else insert)
                cur = get_table_df("devices")
                by_h = {}
                by_e = {}
                if not cur.empty:
                    if "housing_id" in cur.columns and "id" in cur.columns:
                        by_h = {str(x).strip(): int(i)
                                for i, x in cur[["id","housing_id"]].dropna().itertuples(index=False, name=None)}
                    if "electronics_id" in cur.columns and "id" in cur.columns:
                        by_e = {str(x).strip(): int(i)
                                for i, x in cur[["id","electronics_id"]].dropna().itertuples(index=False, name=None)}
                for r in rows_h or []:
                    hid = (r.get("housing_id") or "").strip()
                    if hid and hid in by_h:
                        sb.table("devices").update(r).eq("id", by_h[hid]).execute()
                    else:
                        sb.table("devices").insert(r).execute()
                for r in rows_e or []:
                    eid = (r.get("electronics_id") or "").strip()
                    if eid and eid in by_e:
                        sb.table("devices").update(r).eq("id", by_e[eid]).execute()
                    else:
                        sb.table("devices").insert(r).execute()
                st.info("Performed manual merge (fallback).")
                return False

        used_on_conflict = safe_upsert(to_upsert_h, to_upsert_e)
        # ---------------------------------------------------

        # 2) In Replace mode: delete rows that are in DB but not in workbook
        if load_mode.startswith("Replace"):
            missing_keys = set(existing.keys()) - set(incoming.keys())
            if missing_keys:
                # map keys back to filters
                to_del_h = [existing[k]["housing_id"] for k in missing_keys if existing[k].get("housing_id")]
                to_del_e = [existing[k]["electronics_id"] for k in missing_keys if existing[k].get("electronics_id")]

                # delete by ids found via those keys
                if to_del_h:
                    ids = sb.table("devices").select("id").in_("housing_id", to_del_h).execute().data
                    if ids:
                        sb.table("devices").delete().in_("id", [x["id"] for x in ids]).execute()

                if to_del_e:
                    ids = sb.table("devices").select("id").in_("electronics_id", to_del_e).execute().data
                    if ids:
                        sb.table("devices").delete().in_("id", [x["id"] for x in ids]).execute()

        # Log snapshot_import entries (per-device)
        for r in dicts:
            detail = (
                f"user={r.get('user')}, status={r.get('status_bucket')}, "
                f"loc={r.get('current_location')}, notes={r.get('notes')}"
            )
            log_action(actor, "snapshot_import", r.get("housing_id"), r.get("electronics_id"), detail)

        # --- INVENTORY (optional) ---
        inv = sheets.get("Inventory", None) or sheets.get("To Test (Inventory)", None)
        if isinstance(inv, pd.DataFrame) and not inv.empty:
            inv = inv.rename(columns={
                "Item": "item", "Qty": "qty",
                "inventory for fed3s": "item", "unnamed: 1": "qty"
            })
            if "item" not in inv.columns or "qty" not in inv.columns:
                inv.columns = ["item","qty"][:len(inv.columns)]
            inv["item"] = inv["item"].apply(lambda x: None if (pd.isna(x) or str(x).strip()=="") else str(x).strip())
            inv["qty"] = pd.to_numeric(inv["qty"], errors="coerce").fillna(0)
            inv = inv.dropna(subset=["item"])
            # Replace inventory to match workbook
            sb.table("inventory").delete().neq("id",-1).execute()
            insert_rows("inventory", df_to_dicts(inv))
            log_action(actor, "bulk_upsert_inventory", details=f"rows={len(inv)}")

        st.success(f"Loaded {len(dicts)} device rows ({'replace' if load_mode.startswith('Replace') else 'merge'} mode).")

        st.write("---")
        st.subheader("Add / Remove Device")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("Add / Update")
            ah = st.text_input("Housing ID (e.g., H36)", key="add_h")
            ae = st.text_input("Electronics ID (e.g., E36)", key="add_e")
            ahs = st.selectbox("Housing status", ["Working","Broken","Unknown"], index=0, key="add_hs")
            aes = st.selectbox("Electronics status", ["Working","Broken","Unknown"], index=0, key="add_es")
            auser = st.selectbox("User", USERS, index=USERS.index("Unassigned"), key="add_user")
            ainuse = st.checkbox("In use", value=False, key="add_inuse")
            aloc = st.text_input("Location", key="add_loc")
            anotes = st.text_input("Notes", key="add_notes")
        
            if st.button("Save device"):
                rec = {
                    "housing_id": ah or None,
                    "electronics_id": ae or None,
                    "housing_status": ahs if ahs!="Unknown" else None,
                    "electronics_status": aes if aes!="Unknown" else None,
                    "user": None if auser=="Unassigned" else auser,
                    "in_use": bool(ainuse),
                    "current_location": aloc or None,
                    "notes": anotes or None,
                }
                rec["status_bucket"] = compute_bucket(rec)
                # upsert by housing_id or electronics_id
                if rec["housing_id"]:
                    sb.table("devices").upsert(rec, on_conflict="housing_id").execute()
                elif rec["electronics_id"]:
                    sb.table("devices").upsert(rec, on_conflict="electronics_id").execute()
                else:
                    st.warning("Provide at least one of housing_id or electronics_id.")
                log_action(actor, "admin_save_device", rec.get("housing_id"), rec.get("electronics_id"),
                           f"status={rec['status_bucket']}; user={rec.get('user')}; loc={rec.get('current_location')}")
                st.success("Saved.")
        
        with c2:
            st.write("Remove")
            rm_mode = st.radio("Identify by", ["housing_id","electronics_id"], horizontal=True)
            df_all = get_table_df("devices")
            opts = sorted(df_all.get(rm_mode, pd.Series([], dtype="object")).dropna().unique().tolist())
            rid = st.selectbox(rm_mode, [""] + opts)
            if st.button("Delete device"):
                if not rid:
                    st.warning("Pick an ID to delete.")
                else:
                    sb.table("devices").delete().eq(rm_mode, rid).execute()
                    log_action(actor, "admin_delete_device",
                               housing_id=rid if rm_mode=="housing_id" else None,
                               electronics_id=rid if rm_mode=="electronics_id" else None,
                               details="deleted from Admin")
                    st.success("Deleted.")

            
            # --- INVENTORY (optional) ---
            inv = sheets.get("Inventory", None)
            if inv is None:
                inv = sheets.get("To Test (Inventory)", None)

            if isinstance(inv, pd.DataFrame) and not inv.empty:
                inv = inv.rename(columns={
                    "Item": "item", "Qty": "qty",
                    "inventory for fed3s": "item", "unnamed: 1": "qty"
                })
                if "item" not in inv.columns or "qty" not in inv.columns:
                    inv.columns = ["item", "qty"][:len(inv.columns)]
                inv["item"] = inv["item"].apply(lambda x: None if pd.isna(x) or str(x).strip()=="" else str(x).strip())
                inv["qty"] = pd.to_numeric(inv["qty"], errors="coerce").fillna(0)
                inv = inv.dropna(subset=["item"])
                delete_all("inventory")
                insert_rows("inventory", df_to_dicts(inv))
                log_action(actor, "bulk_upsert_inventory", details=f"rows={len(inv)}")
            else:
                st.info("No Inventory sheet found; skipped inventory load.")

            st.success("Database initialized from workbook.")

# -------------------------------
# Tabs
# -------------------------------
tab_devices, tab_request, tab_users, tab_to_test, tab_inventory, tab_admin = st.tabs(
    ["Devices (Search)", "Request Devices", "My FEDs", "Mark & Repair", "Inventory", "Admin / Export"]
)

# -------------------------------
# Devices (Searchable Overview)
# -------------------------------
with tab_devices:
    st.subheader("Devices — filter & explore")
    df = get_table_df("devices")

    # Build filter widgets
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
    status_pick = c1.multiselect("Status", STATUS_OPTIONS, default=STATUS_OPTIONS)
    user_vals = sorted([x for x in df.get("user", pd.Series([], dtype="object")).dropna().unique().tolist()] + ["Unassigned"])
    user_pick = c2.multiselect("User", user_vals, default=user_vals)
    issue_vals = sorted([x for x in df.get("issue_tags", pd.Series([], dtype="object")).dropna().unique().tolist()])
    issue_pick = c3.multiselect("Issue", issue_vals, default=issue_vals)
    id_search = c4.text_input("ID contains (housing/electronics)")
    text_search = c5.text_input("Search notes/location")

    # Apply filters safely
    def series_str(s):  # normalize to string series for safe .str ops
        return s.fillna("").astype(str)

    if "status_bucket" not in df.columns:
        df["status_bucket"] = pd.Series(dtype="object")
    if "user" not in df.columns:
        df["user"] = pd.Series(dtype="object")
    if "issue_tags" not in df.columns:
        df["issue_tags"] = pd.Series(dtype="object")

    work = df.copy()
    work = work[work["status_bucket"].isin(status_pick)] if status_pick else work
    # treat NaN user as "Unassigned" for filter comparison
    ucol = series_str(work["user"]).replace({"": "Unassigned"})
    work = work[ucol.isin(user_pick)] if user_pick else work
    if issue_pick:
        w_issues = series_str(work["issue_tags"])
        work = work[w_issues.isin(issue_pick) | (w_issues.eq("") & ("Unassigned" in issue_pick))]
    if id_search:
        hs = series_str(work.get("housing_id", pd.Series([], dtype="object")))
        es = series_str(work.get("electronics_id", pd.Series([], dtype="object")))
        mask = hs.str.contains(id_search, case=False, na=False) | es.str.contains(id_search, case=False, na=False)
        work = work[mask]
    if text_search:
        notes = series_str(work.get("notes", pd.Series([], dtype="object")))
        loc   = series_str(work.get("current_location", pd.Series([], dtype="object")))
        mask = notes.str.contains(text_search, case=False, na=False) | loc.str.contains(text_search, case=False, na=False)
        work = work[mask]

    # Show table
    show_cols = [c for c in ["id","housing_id","electronics_id","status_bucket","user","issue_tags",
                             "current_location","exp_start_date","notes"] if c in work.columns]
    st.dataframe(maybe_hide_id(work[show_cols] if show_cols else work), width='stretch')

    # Quick history viewer
    st.write("---")
    left, right = st.columns([1,2])
    with left:
        sel_h = st.selectbox("View history for housing_id", [""] + sorted(series_str(df.get("housing_id", pd.Series([]))).unique().tolist()))
        sel_e = st.selectbox("View history for electronics_id", [""] + sorted(series_str(df.get("electronics_id", pd.Series([]))).unique().tolist()))
        st.caption("Pick either (or both).")
    with right:
        hist = get_history(sel_h or None, sel_e or None)
        if hist.empty:
            st.info("No history for the selected IDs.")
        else:
            st.dataframe(maybe_hide_id(hist), width='stretch')

# -------------------------------
# Request Devices
# -------------------------------
with tab_request:
    st.subheader("Request Ready Devices")
    who = st.selectbox("Researcher", USERS, index=0)
    n = st.number_input("How many devices?", min_value=1, value=1, step=1)
    df = get_table_df("devices")
    if "status_bucket" not in df.columns:
        df["status_bucket"] = pd.Series(dtype="object")
    ready = df[df["status_bucket"]=="Ready for Use"].sort_values("id").head(int(n)) if "status_bucket" in df.columns else pd.DataFrame()
    avail = int((df["status_bucket"]=="Ready for Use").sum()) if "status_bucket" in df.columns else 0
    st.write(f"Ready available: {int(avail)}")
    show_cols = [c for c in ["id","housing_id","electronics_id","current_location","notes"] if c in ready.columns]
    st.write("Preview")
    st.dataframe(maybe_hide_id(ready[show_cols] if show_cols else ready), width='stretch')

    # Inline history: show for first item in preview
    if not ready.empty:
        r0 = ready.iloc[0]
        st.write("History (first in preview):")
        st.dataframe(maybe_hide_id(get_history(r0.get("housing_id"), r0.get("electronics_id"))), width='stretch')

    if st.button("Allocate"):
        if ready is None or ready.empty:
            st.warning("No devices available.")
        else:
            ids = ready["id"].tolist() if "id" in ready.columns else []
            updates = {"user": None if who=="Unassigned" else who, "in_use": who!="Unassigned", "status_bucket":"In Use"}
            update_rows("devices", ids, updates)
            for _, r in ready.iterrows():
                log_action("system", "allocate", r.get("housing_id"), r.get("electronics_id"), f"allocated to {who}")
            st.success(f"Allocated {len(ids)} device(s) to {who}.")

# -------------------------------
# My FEDs
# -------------------------------
with tab_users:
    st.subheader("My FEDs")
    who2 = st.selectbox("Researcher", USERS, index=0, key="user_view")
    df = get_table_df("devices")
    if "status_bucket" not in df.columns:
        df["status_bucket"] = pd.Series(dtype="object")
    if who2 == "Unassigned":
        mine = df[(df["status_bucket"]=="In Use") & (df["user"].isna())] if "user" in df.columns else pd.DataFrame()
    else:
        mine = df[(df["status_bucket"]=="In Use") & (df["user"]==who2)] if "user" in df.columns else pd.DataFrame()
    show_cols = [c for c in ["id","housing_id","electronics_id","current_location","exp_start_date","notes"] if c in mine.columns]
    st.dataframe(maybe_hide_id(mine[show_cols] if show_cols else mine), width='stretch')

    # quick release
    choices = mine["housing_id"].dropna().tolist() if "housing_id" in mine.columns and not mine.empty else []
    sel = st.multiselect("Select housing_id(s) to release to Ready", choices)
    if st.button("Release to Ready"):
        if not sel:
            st.warning("Select at least one device.")
        else:
            rows = df[df["housing_id"].isin(sel)][["id","housing_status","electronics_status"]] if set(["housing_status","electronics_status","id"]).issubset(df.columns) else pd.DataFrame()
            if not rows.empty:
                qualified = rows[(rows["housing_status"].str.lower()=="working") & (rows["electronics_status"].str.lower()=="working")]
                ids = qualified["id"].tolist()
            else:
                ids = []
            if not ids:
                st.warning("No selected devices qualify (both parts must be Working).")
            else:
                update_rows("devices", ids, {"in_use": False, "user": None, "status_bucket":"Ready for Use"})
                for hid in sel:
                    log_action("system", "release", housing_id=hid, details="released to Ready")
                st.success(f"Released {len(ids)} device(s) to Ready.")

# -------------------------------
# To Test / Repair
# -------------------------------
with tab_to_test:
    st.subheader("Mark Issue / Move to To Test")
    df = get_table_df("devices")
    id_mode = st.radio("Identify device by", ["housing_id", "electronics_id"], horizontal=True)
    options = sorted(df[id_mode].dropna().unique().tolist()) if not df.empty and id_mode in df.columns else []
    pick = st.selectbox(id_mode, options)
    issue = st.selectbox("Issue", ISSUE_OPTIONS)
    notes_add = st.text_input("Additional note (optional)")
    colA, colB = st.columns(2)
    if colA.button("Move to To Test"):
        if not pick:
            st.warning("Choose a device.")
        else:
            mask = (df[id_mode] == pick) if id_mode in df.columns else pd.Series([], dtype=bool)
            ids = df[mask]["id"].tolist() if "id" in df.columns else []
            if issue.startswith("Housing"):
                update_rows("devices", ids, {"housing_status":"Broken"})
            elif issue.startswith("Electronics"):
                update_rows("devices", ids, {"electronics_status":"Broken"})
            update_rows("devices", ids, {"in_use": False, "user": None, "status_bucket":"To Test"})
            for row in df[mask].to_dict(orient="records"):
                old = row.get("notes") or ""
                extra = f"[Issue] {issue}" + (f"; {notes_add}" if notes_add else "")
                new_notes = (old + (" | " if old else "") + extra)
                update_rows("devices", [row["id"]], {"notes": new_notes, "issue_tags": issue})
            log_action("system", "mark_to_test", details=f"{id_mode}={pick} | {issue}")
            st.success("Updated.")

    st.write("Repair / Return to Ready")
    tofix = df[(df.get("status_bucket","")=="To Test") & df.get("housing_id").notna()] if not df.empty and "status_bucket" in df.columns and "housing_id" in df.columns else pd.DataFrame()
    pick2 = st.selectbox("housing_id to mark repaired", sorted(tofix["housing_id"].unique().tolist()) if not tofix.empty else [])
    col1, col2 = st.columns(2)
    housing_ok = col1.checkbox("Housing working")
    board_ok = col2.checkbox("Electronics working")
    if st.button("Mark repaired"):
        if not pick2:
            st.warning("Choose a device.")
        else:
            ids = df[df["housing_id"]==pick2]["id"].tolist() if "id" in df.columns and "housing_id" in df.columns else []
            if housing_ok:
                update_rows("devices", ids, {"housing_status":"Working"})
            if board_ok:
                update_rows("devices", ids, {"electronics_status":"Working"})
            rows = df[df["housing_id"]==pick2].to_dict(orient="records") if "housing_id" in df.columns else []
            for r in rows:
                hs = "working" if housing_ok else str(r.get("housing_status","")).lower()
                es = "working" if board_ok else str(r.get("electronics_status","")).lower()
                if hs=="working" and es=="working":
                    update_rows("devices", [r["id"]], {"status_bucket":"Ready for Use"})
            log_action("system", "repair", housing_id=pick2, details=f"housing_ok={housing_ok}, board_ok={board_ok}")
            st.success("Updated.")

# -------------------------------
# Inventory
# -------------------------------
with tab_inventory:
    st.subheader("Inventory")
    inv = get_table_df("inventory")
    inv_show = inv.drop(columns=["id"]) if (not st.session_state.get("show_ids", False) and "id" in inv.columns) else inv
    st.dataframe(inv_show, width='stretch')

    st.write("Add / update an item")
    item = st.text_input("Item")
    qty = st.number_input("Quantity", value=0.0, step=1.0)
    colx, coly, colz = st.columns(3)
    if colx.button("Add / Update"):
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
    if coly.button("Delete item"):
        if not item:
            st.warning("Enter an item name.")
        else:
            sb.table("inventory").delete().eq("item", item).execute()
            log_action("system", "inv_delete", details=f"{item}")
            st.success("Deleted item.")

# -------------------------------
# Admin / Export
# -------------------------------
with tab_admin:
    st.subheader("Admin / Export")
    actions = get_table_df("actions")
    if "ts" in actions.columns:
        actions = actions.sort_values("ts", ascending=False)
    st.write("Recent Actions")
    st.dataframe(maybe_hide_id(actions.head(50)), width='stretch')

    if st.button("Generate export"):
        ddf = get_table_df("devices")
        ivf = get_table_df("inventory")
        if "id" in ddf.columns:
            ddf = ddf.sort_values("id")
        if "item" in ivf.columns:
            ivf = ivf.sort_values("item")
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            ddf.to_excel(writer, sheet_name="Master List", index=False)
            ivf.to_excel(writer, sheet_name="Inventory", index=False)
            actions.to_excel(writer, sheet_name="Actions", index=False)
        bio.seek(0)
        st.download_button("Download Excel", data=bio.getvalue(), file_name="FED3_DB_Export.xlsx")

st.caption("Refer to READ ME file for use guide.")
