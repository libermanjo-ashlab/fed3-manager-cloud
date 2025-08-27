
import streamlit as st
import pandas as pd
from io import BytesIO
from supabase import create_client, Client
from postgrest.exceptions import APIError

st.set_page_config(page_title="FED3 Manager", layout="wide")

@st.cache_resource
def get_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

sb = get_client()

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

# Only these columns are inserted into 'devices' (whitelist)
DEVICE_FIELDS = [
    "housing_id","housing_status",
    "electronics_id","electronics_status",
    "status_in_lab","status_with_mice",
    "in_use","user",
    "current_location","exp_start_date",
    "notes","status_bucket","issue_tags"
]

def norm(s):
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None

def compute_bucket(row: dict) -> str:
    in_use = bool(row.get("in_use") or False)
    with_mice_working = str(row.get("status_with_mice") or "").lower() == "working"
    housing_working = str(row.get("housing_status") or "").lower() == "working"
    board_working   = str(row.get("electronics_status") or "").lower() == "working"
    has_housing = bool(norm(row.get("housing_id")))
    has_board   = bool(norm(row.get("electronics_id")))
    if in_use or (norm(row.get("user")) and with_mice_working):
        return "In Use"
    if housing_working and board_working and (not with_mice_working) and (not in_use):
        return "Ready for Use"
    if has_housing and has_board and (not housing_working or not board_working):
        return "To Test"
    if has_housing and (not has_board and not housing_working):
        return "To Test"
    if has_board   and (not has_housing and not board_working):
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
            elif isinstance(v, (pd.Timestamp,)):
                o[k] = v.isoformat()
            else:
                o[k] = v
        out.append(o)
    return out

def filter_device_fields(rows):
    filtered = []
    for r in rows:
        f = {k: r.get(k) for k in DEVICE_FIELDS if k in r}
        filtered.append(f)
    return filtered

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
        st.error("Tables not found. Create them in Supabase with SQL_SCHEMA.sql or the patch provided.")
        st.stop()

ensure_tables_exist()

st.title("FED3 Manager — Minimal (Hardened)")

with st.sidebar:
    st.header("Setup / Admin")
    st.checkbox("Show internal IDs", value=False, key="show_ids")
    actor = st.text_input("Your name (for log)", value="lab-user")
    st.write("---")
    st.write("Initialize / refresh from workbook")
    upl = st.file_uploader("Upload combined workbook (.xlsx)", type=["xlsx"])
    if upl is not None and st.button("Load workbook into database"):
        sheets = pd.read_excel(BytesIO(upl.read()), sheet_name=None, engine="openpyxl")
        devices = sheets.get("Master List")
        if devices is None:
            st.error("Workbook must include a sheet named 'Master List'.")
        else:
            rename = {"status_(in_lab)":"status_in_lab", "status_(with_mice)":"status_with_mice"}
            devices = devices.rename(columns=rename)
            for c in ["in_use","user","housing_id","housing_status","electronics_id","electronics_status",
                      "status_in_lab","status_with_mice","current_location","exp_start_date","notes","issue_tags"]:
                if c not in devices.columns: devices[c] = None
                devices[c] = devices[c].apply(norm)
            def to_bool(x):
                sx = str(x).lower()
                return sx in ("1","yes","true","y")
            devices["in_use"] = devices["in_use"].apply(to_bool)
            dicts = df_to_dicts(devices)
            for r in dicts:
                r["status_bucket"] = compute_bucket(r)
            dicts = filter_device_fields(dicts)  # whitelist
            delete_all("devices")
            insert_rows("devices", dicts)
            log_action(actor, "bulk_upsert_devices", details=f"rows={len(dicts)}")
            inv = sheets.get("Inventory") or sheets.get("To Test (Inventory)")
            if inv is not None:
                inv = inv.rename(columns={"Item":"item","Qty":"qty","inventory for fed3s":"item","unnamed: 1":"qty"})
                if "item" not in inv.columns or "qty" not in inv.columns:
                    inv.columns = ["item","qty"][:len(inv.columns)]
                inv["item"] = inv["item"].apply(norm)
                inv["qty"] = pd.to_numeric(inv["qty"], errors="coerce").fillna(0)
                inv = inv.dropna(subset=["item"])
                delete_all("inventory")
                insert_rows("inventory", df_to_dicts(inv))
                log_action(actor, "bulk_upsert_inventory", details=f"rows={len(inv)}")
            st.success("Database initialized from workbook.")

def maybe_hide_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: 
        return df
    if not st.session_state.get("show_ids", False) and "id" in df.columns:
        return df.drop(columns=["id"])
    return df

def counts_df():
    df = get_table_df("devices")
    if "status_bucket" not in df.columns:
        df["status_bucket"] = pd.Series(dtype="object")
    if df.empty:
        return pd.DataFrame({"status_bucket": STATUS_OPTIONS, "n":[0,0,0,0]})
    c = df.groupby("status_bucket").size().reset_index(name="n")
    return c

tab_dash, tab_request, tab_users, tab_to_test, tab_inventory, tab_admin = st.tabs(
    ["Dashboard", "Request Devices", "My FEDs", "Mark & Repair", "Inventory", "Admin / Export"]
)

with tab_dash:
    st.subheader("Overview")
    c = counts_df()
    def getc(name):
        row = c[c["status_bucket"]==name]["n"]
        return int(row.iloc[0]) if not row.empty else 0
    st.write(f"Ready for Use: {getc('Ready for Use')}")
    st.write(f"In Use: {getc('In Use')}")
    st.write(f"To Test: {getc('To Test')}")
    st.write(f"Unclear: {getc('Unclear')}")

    df = get_table_df("devices")
    if "status_bucket" not in df.columns:
        df["status_bucket"] = pd.Series(dtype="object")
    g = df[df["status_bucket"]=="In Use"].copy() if "status_bucket" in df.columns else pd.DataFrame()
    if not g.empty and "user" in g.columns:
        g["user"] = g["user"].fillna("Unassigned")
        by_user = g.groupby("user").size().reset_index(name="n").sort_values("n", ascending=False)
    else:
        by_user = pd.DataFrame(columns=["user","n"])
    st.write("In Use by User")
    st.dataframe(by_user, width='stretch')

    actions = get_table_df("actions")
    if "ts" in actions.columns:
        actions = actions.sort_values("ts", ascending=False)
    st.write("Recent Actions")
    st.dataframe(maybe_hide_id(actions.head(20)), width='stretch')

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
    show_cols = [c for c in ["id","housing_id","electronics_id","current_location"] if c in ready.columns]
    st.write("Preview")
    st.dataframe(maybe_hide_id(ready[show_cols] if show_cols else ready), width='stretch')
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

with tab_admin:
    st.subheader("Admin / Export")
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
        bio.seek(0)
        st.download_button("Download Excel", data=bio.getvalue(), file_name="FED3_DB_Export.xlsx")

st.caption("Minimal black & white UI — hardened to accept varied workbook columns and resilient to missing DB columns.")
