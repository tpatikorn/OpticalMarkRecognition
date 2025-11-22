import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

from main import sheet_template, grade_these
from omr import AnswerSheetTemplate
from omr_util import extract_images_from_folder

# ==========================================
# 0. CONFIG & MOCK FUNCTIONS (REPLACE THESE)
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

tab1, tab2, tab3, tab4 = st.tabs(["1. Configuration", "2. Upload & Process", "3. Grading", "4. Review & Export"])

# --- TAB 1: CONFIGURATION (Roster & Template) ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Answer Sheet Template")
        # Load existing or default
        template_path = os.path.join(NS_DIR, "template.json")
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
    st.subheader("Upload Scans (PDF or Images)")
    uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])

    if uploaded_files:
        if st.button(f"Save {len(uploaded_files)} files to Raw"):
            for u_file in uploaded_files:
                with open(os.path.join(RAW_DIR, u_file.name), "wb") as f:
                    f.write(u_file.getbuffer())
            st.success("Files saved to Raw folder.")

    st.divider()

    # Check Raw Folder
    raw_files = os.listdir(RAW_DIR)
    st.write(f"Files in Raw Queue: **{len(raw_files)}**")

    if st.button("⚙️ Process These (Extract Pages)"):
        with st.spinner("Extracting pages..."):
            # CALL YOUR FUNCTION HERE
            count = extract_images_from_folder(raw_folder=RAW_DIR, output_folder=PAGES_DIR)
            st.success(f"Extraction complete! {count} pages ready.")
            st.session_state['extraction_done'] = True

    # Show Previews if extraction done
    pages_files = os.listdir(PAGES_DIR)
    if len(pages_files) > 0:
        st.write(f"### Extracted Pages ({len(pages_files)})")
        with st.expander("View Page Previews"):
            cols = st.columns(4)
            for i, img_file in enumerate(pages_files[:8]):  # Show first 8 only to save memory
                cols[i % 4].image(os.path.join(PAGES_DIR, img_file), caption=img_file)
            if len(pages_files) > 8:
                st.write("...and more.")

# --- TAB 3: GRADING ---
with tab3:
    st.subheader("Grading Execution")

    if st.button("📝 Grade These"):
        if not os.path.exists(os.path.join(NS_DIR, "template.json")):
            st.error("Please save an AnswerSheetTemplate in Tab 1 first.")
        else:
            with st.spinner("Grading..."):
                # CALL YOUR FUNCTION HERE
                results = grade_these(AnswerSheetTemplate.from_dict(json.loads(current_template)),
                                      PAGES_DIR, highlight_folder=GRADED_DIR)
                st.session_state['grading_results'] = results
                st.success("Grading Complete!")

    # Display raw results if available
    if 'grading_results' in st.session_state:
        st.write("Grading finished. Go to **Tab 4** to review and finalize.")
        st.json(st.session_state['grading_results'][0])  # Show sample

# --- TAB 4: REVIEW & RESULTS (The UI Logic you asked for) ---
with tab4:
    if 'grading_results' not in st.session_state:
        st.info("No grading results yet. Run grading in Tab 3.")
    else:
        results = st.session_state['grading_results']
        result_columns = ["filepath", "output_path", "student_id", "all_scores"]
        result_dicts = []
        for _result in results:
            result_dicts.append(dict([(_k, _v) for _k, _v in vars(_result).items() if _k in result_columns]))

        # Convert results to DataFrame for easier manipulation
        res_df = pd.DataFrame(result_dicts)
        res_df["filename"] = res_df.filepath.apply(os.path.basename)

        # Merge with Roster to get Names
        # We assume 'student_id' matches
        full_df = pd.merge(res_df, roster_df, on='student_id', how='left')

        # Calculate valid students for dropdown (Students who don't have a score yet)
        # In a real app, you might want to allow overwriting, but let's filter for now
        assigned_ids = full_df['student_id'].unique()
        available_students = roster_df[~roster_df['student_id'].isin(assigned_ids)]

        # Create options list for dropdown: "ID - Name"
        student_options = available_students.apply(
            lambda x: f"{x['student_id']} - {x['student_firstname']} {x['student_lastname']}", axis=1).tolist()
        student_options.insert(0, "Select Student...")
        student_options.append("DISCARD IMAGE")

        st.subheader("Class Overview")

        # Summary Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Scanned", len(full_df))
        c2.metric("Matched Students", full_df['student_firstname'].notna().sum())
        c3.metric("Unknown/Errors", full_df['student_firstname'].isna().sum())

        st.divider()

        # --- THE REVIEW UI ---
        st.write("### 🔍 Detailed Review")
        st.info("Review entries with 'UNKNOWN' ID or low confidence.")

        # Iterate through results to create a custom row for each
        # We use session_state to track manual overrides
        if 'manual_overrides' not in st.session_state:
            st.session_state['manual_overrides'] = {}

        for index, row in full_df.iterrows():
            # Determine if this row needs attention
            is_error = row['student_id'] == "UNKNOWN" or pd.isna(row['student_firstname'])

            # Visual container
            with st.container(border=True):
                c_img, c_info, c_action = st.columns([1, 2, 2])

                with c_img:
                    # Thumbnail that expands
                    st.image(row['output_path'], width=100)
                    with st.popover("🔍 Zoom"):
                        st.image(row['output_path'])

                with c_info:
                    if is_error:
                        st.error(f"❌ ID: {row['student_id']}")
                    else:
                        st.success(f"✅ {row['student_firstname']} {row['student_lastname']}")
                    st.write(f"**Score:** {row['all_scores']}")
                    st.caption(f"File: {row['filename']}")

                with c_action:
                    # If it's an error or collision, show dropdown
                    if is_error:
                        # Check if we already fixed it in this session
                        current_override = st.session_state['manual_overrides'].get(row['filename'])

                        val = st.selectbox(
                            "Assign to:",
                            options=student_options,
                            key=f"fix_{index}",
                            index=0 if not current_override else student_options.index(
                                current_override) if current_override in student_options else 0
                        )

                        if val != "Select Student...":
                            st.session_state['manual_overrides'][row['filename']] = val
                            st.write(f"⚠️ Marked as: **{val}**")

        # Export Button
        if st.button("💾 Save Final Grades to CSV"):
            # Logic to merge manual overrides into the final dataframe
            final_export = full_df.copy()

            for filename, correction in st.session_state['manual_overrides'].items():
                if correction == "DISCARD IMAGE":
                    final_export = final_export[final_export['filename'] != filename]
                else:
                    # Extract ID from "123 - John Doe"
                    new_id = correction.split(" - ")[0]
                    # Update the row
                    mask = final_export['filename'] == filename
                    final_export.loc[mask, 'student_id'] = new_id

            # Re-merge to get names for corrected IDs
            # (Simplification for demo: In production, you'd re-map the names properly)

            save_path = os.path.join(GRADED_DIR, f"final_grades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            final_export.to_csv(save_path, index=False)
            st.success(f"Exported to {save_path}")
            st.dataframe(final_export)