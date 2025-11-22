import streamlit as st
import pandas as pd
import os
import json
import shutil
from datetime import datetime

# We assume these files (main.py, omr.py, omr_util.py) exist in the same directory
from main import sheet_template, grade_these
from omr import AnswerSheetTemplate
from omr_util import extract_images_from_folder

# ==========================================
# 0. CONFIG & SETUP
# ==========================================

st.set_page_config(page_title="OMR Grader", layout="wide")

# ==========================================
# 1. AUTHENTICATION & SETUP
# ==========================================

# Check Auth (Requires secrets.toml and Google Cloud setup)
if not st.user.is_logged_in:
    st.title("🎓 OMR Grading System")
    st.warning("Please log in to access your workspace.")
    if st.button("Log in with Google"):
        st.login()
    st.stop()

# User is logged in
user_email = st.user.email
st.sidebar.write(f"👤 **{user_email}**")
if st.sidebar.button("Logout"):
    st.logout()

# ==========================================
# 2. SIDEBAR: NAMESPACE MANAGEMENT
# ==========================================

# Updated to your specific path
BASE_UPLOAD_DIR = "C:/Users/Lenovo/PycharmProjects/OpticalMarkRecognition/uploads"

# Ensure base user dir exists
user_dir = os.path.join(BASE_UPLOAD_DIR, user_email)
if not os.path.exists(user_dir):
    os.makedirs(user_dir)

# Get existing namespaces
existing_namespaces = [d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))]
if "default" not in existing_namespaces:
    existing_namespaces.append("default")
    os.makedirs(os.path.join(user_dir, "default"), exist_ok=True)

st.sidebar.header("📂 Workspace")
selected_namespace = st.sidebar.selectbox("Select Working Folder", existing_namespaces)

# Create new namespace
with st.sidebar.expander("Create New Folder"):
    new_ns = st.text_input("Folder Name")
    if st.button("Create"):
        if new_ns and new_ns not in existing_namespaces:
            os.makedirs(os.path.join(user_dir, new_ns))
            st.rerun()

# Define paths for current namespace
NS_DIR = os.path.join(user_dir, selected_namespace)
RAW_DIR = os.path.join(NS_DIR, "raw")
PAGES_DIR = os.path.join(NS_DIR, "pages")
GRADED_DIR = os.path.join(NS_DIR, "graded")

for d in [RAW_DIR, PAGES_DIR, GRADED_DIR]:
    os.makedirs(d, exist_ok=True)

st.title(f"Correction Dashboard: {selected_namespace}")

# ==========================================
# 3. TABS UI IMPLEMENTATION
# ==========================================

# Combined Tab 3 and 4 into "Grading & Review"
tab1, tab2, tab3 = st.tabs(["1. Configuration", "2. Upload & Process", "3. Grading & Review"])

# --- TAB 1: CONFIGURATION (Roster & Template) ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Answer Sheet Template")
        # Load existing or default
        template_path = os.path.join(NS_DIR, "template.json")
        # Using your sheet_template from main
        default_template = json.dumps(sheet_template.to_dict())

        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                current_template = f.read()
        else:
            current_template = default_template

        template_str = st.text_area("JSON Config", current_template, height=300)
        if st.button("Save Template"):
            try:
                json_obj = json.loads(template_str)  # Validate JSON
                with open(template_path, "w") as f:
                    f.write(template_str)
                st.success("Template saved!")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")

    with col2:
        st.subheader("Student Roster (CSV)")
        st.info("Required columns: student_id, student_firstname, student_lastname")
        roster_file = st.file_uploader("Upload Class Roster", type=["csv"])

        roster_path = os.path.join(NS_DIR, "roster.csv")
        if roster_file:
            df = pd.read_csv(roster_file)
            # Basic validation
            required_cols = {'student_id', 'student_firstname', 'student_lastname'}
            if required_cols.issubset(df.columns):
                df.to_csv(roster_path, index=False)
                st.success(f"Roster saved with {len(df)} students.")
            else:
                st.error(f"CSV missing columns. Found: {list(df.columns)}")

        # Load Roster for later use
        if os.path.exists(roster_path):
            roster_df = pd.read_csv(roster_path)
            # Convert ID to string for consistency
            roster_df['student_id'] = roster_df['student_id'].astype(str)
            st.dataframe(roster_df, height=200)
        else:
            roster_df = pd.DataFrame(columns=['student_id', 'student_firstname', 'student_lastname'])
            st.warning("No roster uploaded yet.")

# --- TAB 2: UPLOAD & PROCESS ---
with tab2:
    col_up, col_man = st.columns([2, 1])

    with col_up:
        st.subheader("Upload Scans")
        uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True,
                                          type=['pdf', 'png', 'jpg', 'jpeg'])

        if uploaded_files:
            if st.button(f"Save {len(uploaded_files)} files to Raw"):
                for u_file in uploaded_files:
                    with open(os.path.join(RAW_DIR, u_file.name), "wb") as f:
                        f.write(u_file.getbuffer())
                st.success("Files saved to Raw folder.")
                st.rerun()

    with col_man:
        st.subheader("Manage Files")
        # 1. Raw Files Management
        raw_files = os.listdir(RAW_DIR)
        with st.expander(f"Raw Files ({len(raw_files)})", expanded=False):
            if len(raw_files) == 0:
                st.caption("No files.")
            else:
                for f in raw_files:
                    c1, c2 = st.columns([4, 1])
                    c1.write(f)
                    if c2.button("🗑️", key=f"del_raw_{f}"):
                        os.remove(os.path.join(RAW_DIR, f))
                        st.rerun()

        # 2. Processed Pages Management
        pages_files = os.listdir(PAGES_DIR)
        with st.expander(f"Extracted Pages ({len(pages_files)})", expanded=False):
            if len(pages_files) == 0:
                st.caption("No pages.")
            else:
                # Option to clear all pages
                if st.button("Clear All Pages"):
                    for p in pages_files:
                        os.remove(os.path.join(PAGES_DIR, p))
                    st.rerun()

                for p in pages_files:
                    c1, c2 = st.columns([4, 1])
                    c1.write(p)
                    if c2.button("🗑️", key=f"del_page_{p}"):
                        os.remove(os.path.join(PAGES_DIR, p))
                        st.rerun()

    st.divider()

    if st.button("⚙️ Process Raw Files (Extract Pages)", type="primary"):
        with st.spinner("Extracting pages..."):
            # Calling real function from omr_util
            count = extract_images_from_folder(raw_folder=RAW_DIR, output_folder=PAGES_DIR)
            st.success(f"Extraction complete! {count} pages ready.")
            st.session_state['extraction_done'] = True
            st.rerun()

    # Show Previews if extraction done
    if len(pages_files) > 0:
        st.write(f"### Extracted Pages Preview")
        cols = st.columns(4)
        for i, img_file in enumerate(pages_files[:8]):  # Show first 8 only
            cols[i % 4].image(os.path.join(PAGES_DIR, img_file), caption=img_file)

# --- TAB 3: GRADING & REVIEW (COMBINED) ---
with tab3:
    st.subheader("1. Grading Execution")

    col_action, col_status = st.columns([1, 3])
    with col_action:
        if st.button("📝 Run Grading", type="primary", use_container_width=True):
            if not os.path.exists(os.path.join(NS_DIR, "template.json")):
                st.error("Please save an AnswerSheetTemplate in Tab 1 first.")
            else:
                with st.spinner("Grading in progress..."):
                    # Calling real grading function
                    results = grade_these(AnswerSheetTemplate.from_dict(json.loads(current_template)),
                                          PAGES_DIR, highlight_folder=GRADED_DIR)
                    st.session_state['grading_results'] = results
                    # Clear overrides when re-grading to avoid stale data
                    st.session_state['manual_overrides'] = {}
                    st.success("Grading Complete!")
                    st.rerun()

    with col_status:
        if 'grading_results' in st.session_state:
            st.info(f"Last graded: {len(st.session_state['grading_results'])} pages processed.")
        else:
            st.warning("No results yet.")

    st.divider()

    # --- REVIEW SECTION ---
    if 'grading_results' in st.session_state:
        st.subheader("2. Results Review")

        results = st.session_state['grading_results']

        # Convert results objects to DataFrame
        result_columns = ["filepath", "output_path", "student_id", "all_scores"]
        result_dicts = []
        for _result in results:
            result_dicts.append(dict([(_k, _v) for _k, _v in vars(_result).items() if _k in result_columns]))

        res_df = pd.DataFrame(result_dicts)

        if not res_df.empty:
            res_df["filename"] = res_df.filepath.apply(os.path.basename)

            # Ensure overrides dictionary exists
            if 'manual_overrides' not in st.session_state:
                st.session_state['manual_overrides'] = {}


            # --- PRE-PROCESSING FOR LOGIC AND SORTING ---
            # We create a temporary 'effective' dataframe that includes the manual overrides
            # This allows us to sort duplicates together properly even if one was manually fixed

            def get_effective_id(row):
                override = st.session_state['manual_overrides'].get(row['filename'])
                if override == "DISCARD IMAGE":
                    return "DISCARDED"
                elif override:
                    return override.split(" - ")[0]
                return row['student_id']


            res_df['effective_id'] = res_df.apply(get_effective_id, axis=1)

            # Filter out discarded for stats (but maybe keep in list to allow undoing?)
            # Let's keep them in list but mark them.

            # Merge with Roster using EFFECTIVE ID
            # Rename roster cols to avoid collision if necessary, though simple merge is fine here
            # We merge on effective_id against roster's student_id
            full_df = pd.merge(res_df, roster_df, left_on='effective_id', right_on='student_id', how='left',
                               suffixes=('', '_roster'))

            # Fill NaN firstnames for Unknown/Discarded
            full_df['student_firstname'] = full_df['student_firstname'].fillna("-")
            full_df['student_lastname'] = full_df['student_lastname'].fillna("-")

            # Detect Duplicates (on Effective ID), ignoring UNKNOWN/DISCARDED
            ids_series = full_df['effective_id']
            duplicate_mask = ids_series.duplicated(keep=False) & (~ids_series.isin(["UNKNOWN", "DISCARDED"]))
            full_df['is_duplicate'] = duplicate_mask

            # Identify Errors (Unknowns or No Roster Match)
            # Note: A valid ID not in roster will have NaNs for names
            full_df['is_unknown'] = full_df['effective_id'] == "UNKNOWN"
            full_df['missing_roster'] = (full_df['effective_id'] != "UNKNOWN") & (
                        full_df['effective_id'] != "DISCARDED") & (full_df['student_firstname'] == "-")

            # --- CALCULATE AVAILABLE STUDENTS FOR DROPDOWN ---
            # IDs that are currently "occupied" in the full list (excluding discarded/unknown)
            taken_ids = set(full_df[~full_df['effective_id'].isin(["UNKNOWN", "DISCARDED"])]['effective_id'].unique())

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Pages", len(full_df))
            c2.metric("Matched", (~full_df['is_unknown'] & ~full_df['missing_roster'] & ~full_df['is_duplicate'] & (
                        full_df['effective_id'] != "DISCARDED")).sum())
            c3.metric("Duplicates", full_df['is_duplicate'].sum())
            c4.metric("Unknown/Missing", (full_df['is_unknown'] | full_df['missing_roster']).sum())

            st.write("### 🔍 Detailed Review")


            # --- SORTING ---
            # Priority:
            # 0: Unknowns/Missing Roster (Errors)
            # 1: Duplicates (Warnings) -> So they stay together
            # 2: Normal (Success)
            # 3: Discarded

            def get_sort_priority(row):
                if row['effective_id'] == "DISCARDED": return 3
                if row['is_unknown'] or row['missing_roster']: return 0
                if row['is_duplicate']: return 1
                return 2


            full_df['sort_prio'] = full_df.apply(get_sort_priority, axis=1)

            # Sort by Priority, then by Effective ID (to group duplicates), then by Filename
            full_df = full_df.sort_values(by=['sort_prio', 'effective_id', 'filename'], ascending=[True, True, True])

            # --- RENDER LOOP ---
            for index, row in full_df.iterrows():

                # Setup display variables
                e_id = row['effective_id']
                e_name = f"{row['student_firstname']} {row['student_lastname']}"
                if e_id == "DISCARDED": e_name = "Discarded Image"

                # Determine Border Color / Status
                # Streamlit containers don't support color borders natively yet, but we use status elements

                with st.container(border=True):
                    c_img, c_details = st.columns([1, 4])

                    with c_img:
                        if os.path.exists(row['output_path']):
                            st.image(row['output_path'])
                            with st.popover("🔍 Zoom"):
                                st.image(row['output_path'])
                        else:
                            st.warning("Img Missing")

                    with c_details:
                        # --- ROW 1: STATUS | NAME | FILENAME ---
                        r1_c1, r1_c2, r1_c3 = st.columns([2, 3, 2])

                        with r1_c1:
                            if e_id == "DISCARDED":
                                st.caption("🗑️ IGNORED")
                            elif row['is_unknown']:
                                st.error("❌ UNKNOWN ID")
                            elif row['missing_roster']:
                                st.warning("⚠️ NOT IN ROSTER")
                            elif row['is_duplicate']:
                                st.error("👯 DUPLICATE ID")
                            else:
                                st.success("✅ MATCHED")

                        with r1_c2:
                            if e_id != "DISCARDED" and e_id != "UNKNOWN":
                                st.markdown(f"**{e_name}**")
                            else:
                                st.markdown("---")

                        with r1_c3:
                            st.caption(f"📄 {row['filename']}")

                        # --- ROW 2: ID | SCORE ---
                        r2_c1, r2_c2 = st.columns([2, 5])
                        with r2_c1:
                            st.markdown(f"🆔 **{e_id}**")
                        with r2_c2:
                            st.markdown(f"📊 Score: **{row['all_scores']}**")

                        # --- ROW 3: OVERRIDE DROPDOWN ---
                        # Logic: Available = (All Roster - Taken IDs) + (Current ID if valid)

                        # 1. Base Available
                        available_roster = roster_df[~roster_df['student_id'].isin(taken_ids)]

                        # 2. Build Options List
                        options = available_roster.apply(
                            lambda x: f"{x['student_id']} - {x['student_firstname']} {x['student_lastname']}",
                            axis=1).tolist()

                        # 3. Add Current ID back to options if it exists in roster (so it shows as selected)
                        if e_id not in ["UNKNOWN", "DISCARDED"]:
                            # It might be marked as duplicate or taken, so we force add it
                            # Find name in roster (we use the original roster df for this lookup)
                            curr_rec = roster_df[roster_df['student_id'] == e_id]
                            if not curr_rec.empty:
                                curr_str = f"{curr_rec.iloc[0]['student_id']} - {curr_rec.iloc[0]['student_firstname']} {curr_rec.iloc[0]['student_lastname']}"
                                if curr_str not in options:
                                    options.insert(0, curr_str)

                        options.sort()
                        options.insert(0, "Select Student...")
                        options.append("DISCARD IMAGE")

                        # Determine current selection index
                        current_val_str = "Select Student..."
                        if e_id == "DISCARDED":
                            current_val_str = "DISCARD IMAGE"
                        elif e_id != "UNKNOWN" and not row['missing_roster']:
                            # Try to match the string format
                            match = [o for o in options if o.startswith(f"{e_id} -")]
                            if match: current_val_str = match[0]

                        sel_index = 0
                        if current_val_str in options:
                            sel_index = options.index(current_val_str)

                        # Render Selectbox
                        new_val = st.selectbox(
                            "Override / Fix:",
                            options=options,
                            key=f"fix_{row['filename']}",
                            index=sel_index,
                            label_visibility="collapsed"  # Cleaner look since we know what it is
                        )

                        # Update State
                        current_override = st.session_state['manual_overrides'].get(row['filename'])
                        # We only update if it changed from what the SELECTBOX thinks it is vs what Logic thinks
                        # Actually, comparing new_val to current_val_str is safer
                        if new_val != current_val_str:
                            st.session_state['manual_overrides'][row['filename']] = new_val
                            st.rerun()

            st.divider()
            st.subheader("3. Export")

            # --- EXPORT PREPARATION ---
            # Filter out discards
            final_export = full_df[full_df['effective_id'] != "DISCARDED"].copy()

            # Clean columns
            # We want effective_id to be the student_id in CSV
            final_export['student_id'] = final_export['effective_id']

            export_columns = ['student_id', 'student_firstname', 'student_lastname', 'all_scores', 'filename']
            # Keep only existing columns
            final_cols = [c for c in export_columns if c in final_export.columns]
            final_csv = final_export[final_cols].to_csv(index=False).encode('utf-8')

            st.download_button(
                label="💾 Download Graded CSV",
                data=final_csv,
                file_name=f"grades_{selected_namespace}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary"
            )

            with st.expander("Preview Final Data"):
                st.dataframe(final_export[final_cols])

        else:
            st.warning("No results parsed successfully.")