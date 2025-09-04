# app_supabase.py  —  FED3 Manager (manual entry, history, search)
import streamlit as st
import pandas as pd
from datetime import datetime, date
from supabase import create_client, Client
from postgrest.exceptions import APIError
from collections import Counter  # used in some helpers

# -------------------------------
# Page + Style (black & white)
# -------------------------------
st.set_page_config(page_title="FED3 Manager — Minimal", layout="wide")

MONO_CSS = """
<style>
:root { --fg:#111; --muted:#666; --border:#ddd; --bg:#fff; }
html,body,.stApp { background: var(--bg); color: var(--fg); }
section[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
h1,h2,h3,h4 { color: var(--fg) !important; font-weight:600; }
div[data-baseweb="select"] * { color: var(--fg) !important; }
.stButton>button { border:1px solid var(--fg); background:#fff; color:#000; }
.stButton>button:hover { background:#000; color:#fff; }
.stDataFrame { border:1px solid var(--border); }
hr { border:none; border-top:1px solid var(--border); margin:0.75rem 0; }
small, .stCaption { color: var(--muted) !important; }
</style>
"""
st.markdown(MONO_CSS, unsafe_allow_html=True)

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

DEVICE_FIELDS = [
    "housing_id","housing_status",
    "electronics_id","electronics_status",
    "status_in_lab","status_with_mice",
    "in_use","user",
    "current_location","exp_start_date",
    "notes","status_bucket","issue_tags"
]

# -------------------------------
# Supabase client
# -------------------------------
@st.cache_resource
def get_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

sb: Client = get_client()
ADMIN_CODE = st.secrets.get("ADMIN_CODE", None)

# -------------------------------
# Helpers
# -------------------------------
def _iso(v):
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.to_datetime(v).to_pydatetime().isoformat()
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day).isoformat()
    return v

def df_to_dicts(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    recs = df.to_dict(orient="records")
    out = []
    for r in recs:
        o = {}
        for k,v in r.items():
            if pd.isna(v):
                o[k] = None
            else:
                o[k] = _iso(v)
        out.append(o)
    return out

def get_table_df(name: str) -> pd.DataFrame:
    try:
        data = sb.table(name).select("*").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def maybe_hide_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not st.session_state.get("show_ids", False) and "id" in df.columns:
        return df.drop(columns=["id"])
    return df

def compute_bucket(row: dict) -> str:
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

def update_rows(name: str, ids: list[int], updates: dict):
    if not ids:
        return
    # ensure status_bucket stays coherent if related fields changed
    if {"in_use","housing_status","electronics_status"}.intersection(updates.keys()):
        # fetch rows to recompute per-row status if not explicitly set
        if "status_bucket" not in updates:
            cur = sb.table(name).select("id,in_use,housing_status,electronics_status").in_("id", ids).execute().data
            if cur:
                for r in cur:
                    merged = {
                        "in_use": updates.get("in_use", r.get("in_use")),
                        "housing_status": updates.get("housing_status", r.get("housing_status")),
                        "electronics_status": updates.get("electronics_status", r.get("electronics_status")),
                    }
                    merged["status_bucket"] = compute_bucket(merged)
                    sb.table(name).update(merged).eq("id", r["id"]).execute()
                return
    sb.table(name).update(updates).in_("id", ids).execute()

def log_action(actor: str, action: str, housing_id=None, electronics_id=None, details=None):
    try:
        sb.table("actions").insert({
            "actor": actor, "action": action,
            "housing_id": housing_id, "electronics_id": electronics_id,
            "details": details
        }).execute()
    except Exception:
        pass

def get_history(housing_id: str|None=None, electronics_id: str|None=None) -> pd.DataFrame:
    q = sb.table("actions").select("*")
    if housing_id:
        q = q.eq("housing_id", housing_id)
    if electronics_id:
        q = q.eq("electronics_id", electronics_id)
    acts = q.order("ts", desc=True).execute().data or []
    hist = pd.DataFrame(acts)
    # include a current snapshot row
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
    return hist[cols] if cols else hist

# -------------------------------
# Sidebar
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
            st.caption("Admin features hidden unless code matches.")
    else:
        st.caption("ADMIN_CODE not set; admin features disabled.")

# =========================================================
# MAIN TABS (no upload)
# =========================================================
tab_overview, tab_add_edit, tab_request, tab_users, tab_to_test, tab_inventory, tab_audit = st.tabs(
    ["Overview (Search)","Add / Edit Device","Request Devices","My FEDs","Mark & Repair","Inventory","Audit Log"]
)

# -------------------------------
# Overview (Search)
# -------------------------------
with tab_overview:
    st.subheader("Devices — filter & explore")
    df = get_table_df("devices")
    # ensure columns exist for filters
    for c in ["status_bucket","user","issue_tags","housing_id","electronics_id","current_location","notes","exp_start_date"]:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")

    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2])
    status_pick = c1.multiselect("Status", STATUS_OPTIONS, default=STATUS_OPTIONS)
    user_vals = sorted([x for x in df["user"].dropna().unique().tolist()] + ["Unassigned"])
    user_pick = c2.multiselect("User", user_vals, default=user_vals)
    issue_vals = sorted([x for x in df["issue_tags"].dropna().unique().tolist()])
    issue_pick = c3.multiselect("Issue", issue_vals)  # default none (narrow down)
    id_search   = c4.text_input("ID contains (housing/electronics)")
    text_search = c5.text_input("Search notes/location")

    work = df.copy()
    if status_pick:
        work = work[work["status_bucket"].isin(status_pick)]
    # treat NaN user as Unassigned for filtering
    ucol = work["user"].fillna("Unassigned").astype(str)
    if user_pick:
        work = work[ucol.isin(user_pick)]
    if issue_pick:
        work = work[work["issue_tags"].fillna("").astype(str).isin(issue_pick)]
    if id_search:
        hs = work["housing_id"].fillna("").astype(str)
        es = work["electronics_id"].fillna("").astype(str)
        mask = hs.str.contains(id_search, case=False) | es.str.contains(id_search, case=False)
        work = work[mask]
    if text_search:
        notes = work["notes"].fillna("").astype(str)
        loc   = work["current_location"].fillna("").astype(str)
        mask = notes.str.contains(text_search, case=False) | loc.str.contains(text_search, case=False)
        work = work[mask]

    show_cols = [c for c in ["id","housing_id","electronics_id","status_bucket","user","issue_tags",
                             "current_location","exp_start_date","notes"] if c in work.columns]
    st.dataframe(maybe_hide_id(work[show_cols] if show_cols else work), width='stretch')

    st.write("---")
    left, right = st.columns([1,2])
    with left:
        sel_h = st.selectbox("History: housing_id", [""] + sorted(df["housing_id"].dropna().astype(str).unique().tolist()))
        sel_e = st.selectbox("History: electronics_id", [""] + sorted(df["electronics_id"].dropna().astype(str).unique().tolist()))
    with right:
        hist = get_history(sel_h or None, sel_e or None)
        if hist.empty:
            st.info("No history for the selection.")
        else:
            st.dataframe(maybe_hide_id(hist), width='stretch')

# -------------------------------
# Add / Edit Device (manual only)
# -------------------------------
with tab_add_edit:
    st.subheader("Add or Edit a Device")
    df = get_table_df("devices")
    # pick existing to edit (by housing_id or electronics_id)
    c1,c2 = st.columns(2)
    pick_h = c1.selectbox("Select housing_id to edit (optional)", [""] + sorted(df["housing_id"].dropna().astype(str).unique().tolist()))
    pick_e = c2.selectbox("Select electronics_id to edit (optional)", [""] + sorted(df["electronics_id"].dropna().astype(str).unique().tolist()))

    # preload existing if one selected
    existing = None
    if pick_h:
        res = sb.table("devices").select("*").eq("housing_id", pick_h).limit(1).execute().data
        existing = res[0] if res else None
    elif pick_e:
        res = sb.table("devices").select("*").eq("electronics_id", pick_e).limit(1).execute().data
        existing = res[0] if res else None

    # form
    housing_id = st.text_input("Housing ID", value=(existing.get("housing_id") if existing else ""))
    electronics_id = st.text_input("Electronics ID", value=(existing.get("electronics_id") if existing else ""))
    housing_status = st.selectbox("Housing status", ["Unknown","Working","Broken"],
                                  index={"Unknown":0,"Working":1,"Broken":2}[str(existing.get("housing_status","Unknown")) if existing else "Unknown"])
    electronics_status = st.selectbox("Electronics status", ["Unknown","Working","Broken"],
                                  index={"Unknown":0,"Working":1,"Broken":2}[str(existing.get("electronics_status","Unknown")) if existing else "Unknown"])
    user = st.selectbox("User", USERS, index=(USERS.index(existing["user"]) if existing and existing.get("user") in USERS else USERS.index("Unassigned")))
    in_use_flag = st.checkbox("In Use", value=bool(existing.get("in_use")) if existing else False)
    current_location = st.text_input("Location", value=(existing.get("current_location") or "") if existing else "")
    exp_start_date = st.date_input("Experiment start date", value=None if not existing or not existing.get("exp_start_date") else pd.to_datetime(existing["exp_start_date"]))
    issue_tags = st.selectbox("Issue tag", [""] + ISSUE_OPTIONS, index=(0 if not existing or not existing.get("issue_tags") else (1+ISSUE_OPTIONS.index(existing["issue_tags"])) ))
    notes = st.text_area("Notes", value=(existing.get("notes") or "") if existing else "")

    colA,colB,colC = st.columns(3)
    if colA.button("Save"):
        rec = {
            "housing_id": housing_id.strip() or None,
            "electronics_id": electronics_id.strip() or None,
            "housing_status": None if housing_status=="Unknown" else housing_status,
            "electronics_status": None if electronics_status=="Unknown" else electronics_status,
            "user": None if user=="Unassigned" else user,
            "in_use": bool(in_use_flag),
            "current_location": current_location.strip() or None,
            "exp_start_date": _iso(exp_start_date) if exp_start_date else None,
            "issue_tags": issue_tags or None,
            "notes": notes.strip() or None
        }
        rec["status_bucket"] = compute_bucket(rec)
        # Upsert by housing_id if present, else by electronics_id
        if rec["housing_id"]:
            sb.table("devices").upsert(rec, on_conflict="housing_id").execute()
        elif rec["electronics_id"]:
            sb.table("devices").upsert(rec, on_conflict="electronics_id").execute()
        else:
            st.warning("Provide at least one of housing_id or electronics_id.")
        log_action(actor, "save_device", rec.get("housing_id"), rec.get("electronics_id"),
                   f"status={rec['status_bucket']}; user={rec.get('user')}; loc={rec.get('current_location')}")
        st.success("Saved.")

    if colB.button("Delete"):
        if not housing_id and not electronics_id:
            st.warning("Enter housing_id or electronics_id to delete.")
        else:
            if housing_id:
                sb.table("devices").delete().eq("housing_id", housing_id).execute()
                log_action(actor, "delete_device", housing_id=housing_id, details="deleted")
            if electronics_id:
                sb.table("devices").delete().eq("electronics_id", electronics_id).execute()
                log_action(actor, "delete_device", electronics_id=electronics_id, details="deleted")
            st.success("Deleted if existed.")

    if colC.button("Add history note"):
        ident = housing_id or electronics_id
        if not ident:
            st.warning("Provide an ID to attach a note.")
        else:
            who = actor or "lab-user"
            log_action(who, "note",
                       housing_id=(housing_id or None),
                       electronics_id=(electronics_id or None),
                       details=(notes.strip() or "(no details)"))
            st.success("History note added.")

# -------------------------------
# Request Devices (from Ready)
# -------------------------------
with tab_request:
    st.subheader("Request Ready Devices")
    who = st.selectbox("Researcher", USERS, index=0, key="req_user")
    n = st.number_input("How many devices?", min_value=1, value=1, step=1)
    df = get_table_df("devices")
    if df.empty:
        st.info("No devices yet.")
    else:
        ready = df[df["status_bucket"]=="Ready for Use"].sort_values("id").head(int(n))
        st.write(f"Ready available: {int((df['status_bucket']=='Ready for Use').sum())}")
        show_cols = [c for c in ["id","housing_id","electronics_id","current_location","notes"] if c in ready.columns]
        st.write("Preview")
        st.dataframe(maybe_hide_id(ready[show_cols] if show_cols else ready), width='stretch')

        if st.button("Allocate"):
            if ready.empty:
                st.warning("No devices available.")
            else:
                ids = ready["id"].tolist()
                update_rows("devices", ids, {"user": None if who=="Unassigned" else who, "in_use": who!="Unassigned"})
                # ensure status becomes "In Use" when allocated
                update_rows("devices", ids, {"status_bucket":"In Use"})
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
    if df.empty:
        st.info("No devices yet.")
    else:
        if who2 == "Unassigned":
            mine = df[(df["status_bucket"]=="In Use") & (df["user"].isna())]
        else:
            mine = df[(df["status_bucket"]=="In Use") & (df["user"]==who2)]
        show_cols = [c for c in ["id","housing_id","electronics_id","current_location","exp_start_date","notes"] if c in mine.columns]
        st.dataframe(maybe_hide_id(mine[show_cols] if show_cols else mine), width='stretch')

        # release back to Ready
        choices = mine["housing_id"].dropna().tolist() if "housing_id" in mine.columns and not mine.empty else []
        sel = st.multiselect("Select housing_id(s) to release to Ready", choices)
        if st.button("Release"):
            if not sel:
                st.warning("Select at least one device.")
            else:
                rows = df[df["housing_id"].isin(sel)][["id","housing_status","electronics_status"]]
                qualified = rows[(rows["housing_status"].str.lower()=="working") & (rows["electronics_status"].str.lower()=="working")]
                ids = qualified["id"].tolist()
                if not ids:
                    st.warning("No selected devices qualify (both parts must be Working).")
                else:
                    update_rows("devices", ids, {"in_use": False, "user": None, "status_bucket":"Ready for Use"})
                    for hid in sel:
                        log_action("system", "release", housing_id=hid, details="released to Ready")
                    st.success(f"Released {len(ids)} device(s).")

# -------------------------------
# To Test / Repair
# -------------------------------
with tab_to_test:
    st.subheader("Mark Issue / Move to To Test")
    df = get_table_df("devices")
    if df.empty:
        st.info("No devices yet.")
    else:
        id_mode = st.radio("Identify by", ["housing_id","electronics_id"], horizontal=True)
        options = sorted(df[id_mode].dropna().astype(str).unique().tolist())
        pick = st.selectbox(id_mode, options)
        issue = st.selectbox("Issue", ISSUE_OPTIONS)
        notes_add = st.text_input("Additional note (optional)")
        colA, colB = st.columns(2)
        if colA.button("Move to To Test"):
            if not pick:
                st.warning("Choose a device.")
            else:
                mask = (df[id_mode]==pick)
                ids = df[mask]["id"].tolist()
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
        tofix = df[(df["status_bucket"]=="To Test") & df["housing_id"].notna()]
        pick2 = st.selectbox("housing_id to mark repaired", sorted(tofix["housing_id"].astype(str).unique().tolist()) if not tofix.empty else [])
        col1, col2 = st.columns(2)
        housing_ok = col1.checkbox("Housing working")
        board_ok   = col2.checkbox("Electronics working")
        if st.button("Mark repaired"):
            if not pick2:
                st.warning("Choose a device.")
            else:
                ids = df[df["housing_id"]==pick2]["id"].tolist()
                if housing_ok:
                    update_rows("devices", ids, {"housing_status":"Working"})
                if board_ok:
                    update_rows("devices", ids, {"electronics_status":"Working"})
                rows = df[df["housing_id"]==pick2].to_dict(orient="records")
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
# Audit Log
# -------------------------------
with tab_audit:
    st.subheader("Audit Log")
    actions = get_table_df("actions")
    if "ts" in actions.columns and not actions.empty:
        actions = actions.sort_values("ts", ascending=False)
    st.dataframe(maybe_hide_id(actions), width='stretch')

st.caption("Simple monochrome UI • Manual entry only • History by device • Search & filter")
