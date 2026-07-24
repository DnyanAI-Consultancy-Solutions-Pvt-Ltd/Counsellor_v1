import io
import re
import uuid

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="MHT-CET AI Counsellor",
    page_icon="🎓",
    layout="wide",
)

API = st.sidebar.text_input(
    "Backend URL",
    value="http://127.0.0.1:8000",
    label_visibility="collapsed",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "recommendation_df" not in st.session_state:
    st.session_state.recommendation_df = pd.DataFrame()

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 5rem; max-width: 100%;}
        [data-testid="stSidebar"] {background: #f3f5f9;}
        .main-title {text-align:center; font-size:2.35rem; font-weight:800; margin-bottom:.2rem;}
        .main-subtitle {text-align:center; color:#64748b; margin-bottom:1.4rem;}
        .section-title {font-size:1.25rem; font-weight:750; margin-top:.6rem; margin-bottom:.7rem;}
        .feedback-card {
            border:1px solid #d9dee8;
            border-radius:10px;
            padding:1rem 1.2rem;
            background:white;
            min-height:120px;
        }
        .muted {color:#64748b;}
        div[data-testid="stDataFrame"] {border:1px solid #dfe3eb; border-radius:8px; overflow:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='main-title'>🎓 MHT-CET Engineering CAP Round AI Counsellor</div>"
    "<div class='main-subtitle'>Profile-aware, document-grounded counselling</div>",
    unsafe_allow_html=True,
)


def parse_markdown_table(text: str) -> pd.DataFrame:
    """Extract the first valid Markdown table from an LLM response."""
    lines = [line.strip() for line in (text or "").splitlines()]
    for index in range(len(lines) - 2):
        header = lines[index]
        separator = lines[index + 1]
        if "|" not in header or "|" not in separator:
            continue
        if not re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", separator):
            continue

        rows = []
        cursor = index + 2
        while cursor < len(lines) and "|" in lines[cursor] and lines[cursor].strip():
            rows.append(lines[cursor])
            cursor += 1

        columns = [cell.strip() for cell in header.strip("|").split("|")]
        parsed_rows = []
        for row in rows:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if len(cells) == len(columns):
                parsed_rows.append(cells)

        if columns and parsed_rows:
            frame = pd.DataFrame(parsed_rows, columns=columns)
            frame.insert(0, "Preference No.", range(1, len(frame) + 1))
            return frame

    return pd.DataFrame()


def clean_feedback_text(text: str) -> str:
    """Remove the Markdown table and keep the counselling explanation."""
    if not text:
        return ""
    lines = text.splitlines()
    output = []
    inside_table = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        starts_table = "|" in stripped and bool(
            re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", next_line)
        )
        if starts_table:
            inside_table = True
            continue
        if inside_table:
            if "|" in stripped or re.fullmatch(
                r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", stripped
            ):
                continue
            inside_table = False
        output.append(line)
    return "\n".join(output).strip()


def dataframe_to_excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Preference Sheet")
    return buffer.getvalue()


def submit_question(question: str, profile: dict) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    payload = {
        "session_id": st.session_state.session_id,
        "message": question,
        "profile": profile,
    }
    try:
        response = requests.post(f"{API}/chat", json=payload, timeout=180)
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"Backend error ({response.status_code}): {detail}")
            return

        data = response.json()
        answer = data.get("answer", "")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_result = data
        recommendations = data.get("recommendations", [])
        if recommendations:
            frame = pd.DataFrame(recommendations)
            rename_map = {
                "institute_code": "Institute Code",
                "college": "College",
                "city": "City",
                "course_code": "Course Code",
                "course": "Course",
                "seat_type": "Seat Type",
                "cutoff_percentile": "Cutoff Percentile",
                "match_category": "Match Category",
                "reasoning_logic": "Reasoning Logic",
                "source_filename": "Source Filename",
                "source_page": "Source Page",
            }
            frame = frame.rename(columns=rename_map)
            frame.insert(0, "Preference No.", range(1, len(frame) + 1))
            st.session_state.recommendation_df = frame
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Cannot reach backend: {exc}")


student_tab, admin_tab = st.tabs(["Student Counsellor", "Admin Knowledge Base"])

with student_tab:
    with st.sidebar:
        st.header("📋 Candidate Profile Controls")
        mht = st.number_input("MHT-CET Percentile Score", 0.0, 100.0, 95.0, 0.01)
        jee = st.number_input("JEE Percentile Score", 0.0, 100.0, 0.0, 0.01)
        category = st.selectbox(
            "Caste Reservation Pool",
            ["OPEN", "OBC", "SC", "ST", "VJ/DT", "NT-B", "NT-C", "NT-D", "EWS", "Other"],
        )
        branches = st.multiselect(
            "Target Branches",
            [
                "Computer Engineering",
                "Computer Science and Engineering",
                "Information Technology",
                "AI and Data Science",
                "Electronics and Telecommunication",
                "Electrical Engineering",
                "Mechanical Engineering",
                "Civil Engineering",
            ],
            default=["Computer Engineering"],
        )
        pwd = st.checkbox("Registered PWD Status Profile")

        st.markdown("**Optional Preferences:**")
        university = st.text_input("Preferred University / College Name", placeholder="Any")
        cities = st.multiselect(
            "Preferred City Name / Location",
            ["Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad", "Kolhapur", "Other"],
        )
        home = st.text_input("Home University")
        gender = st.radio("Gender Category Pool Allocation", ["General", "Female"])
        tfws = st.checkbox("Consider TFWS")

        st.divider()
        generate_clicked = st.button("🚀 Direct Generate Report Sheet", use_container_width=True)
        if st.button("Start new counselling session", use_container_width=True):
            try:
                requests.delete(f"{API}/sessions/{st.session_state.session_id}", timeout=15)
            except Exception:
                pass
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.recommendation_df = pd.DataFrame()
            st.rerun()

    profile = {
        "mht_cet_percentile": mht,
        "jee_percentile": jee if jee > 0 else None,
        "category": category,
        "gender": gender,
        "home_university": home or None,
        "preferred_branches": branches,
        "preferred_cities": cities,
        "institute_preferences": [university] if university else [],
        "tfws": tfws,
        "pwd": pwd,
    }

    if generate_clicked:
        submit_question(
            "Generate a ranked CAP preference sheet using my complete profile. "
            "Create the initial cutoff_list.xlsx from indexed cutoff evidence. Include at least 30 unique grounded college-course options across Dream, Target and Safe zones. Rank them as a practical CAP preference sheet.",
            profile,
        )

    st.markdown("<div class='section-title'>📊 Master Output Preview Dashboard</div>", unsafe_allow_html=True)

    recommendation_df = st.session_state.recommendation_df
    if recommendation_df.empty:
        st.info(
            "Generate a report from the sidebar or ask the counsellor for ranked college recommendations. "
            "The resulting preference sheet will appear here."
        )
    else:
        try:
            workbook_response = requests.get(
                f"{API}/sessions/{st.session_state.session_id}/workbook", timeout=30
            )
            workbook_data = workbook_response.content if workbook_response.ok else dataframe_to_excel_bytes(recommendation_df)
        except requests.RequestException:
            workbook_data = dataframe_to_excel_bytes(recommendation_df)
        st.download_button(
            "📥 Download Master Preference Sheet (Excel)",
            data=workbook_data,
            file_name="cutoff_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
        )
        counts = (st.session_state.last_result or {}).get("counts", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric("Total Colleges", counts.get("Total", len(recommendation_df)))
        metric_cols[1].metric("Dream", counts.get("Dream", 0))
        metric_cols[2].metric("Target", counts.get("Target", 0))
        metric_cols[3].metric("Safe", counts.get("Safe", 0))
        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True,
            height=min(520, 82 + 36 * len(recommendation_df)),
        )

    st.divider()
    st.markdown(
        "<div class='section-title'>💬 Interactive Feedback & Customization Control Box</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.last_result:
        answer = st.session_state.last_result.get("answer", "")
        feedback = clean_feedback_text(answer)
        confidence = st.session_state.last_result.get("confidence", "Low")
        st.markdown("<div class='feedback-card'>", unsafe_allow_html=True)
        if feedback:
            st.markdown(feedback)
        else:
            st.markdown("The recommendation sheet has been generated from your profile and indexed CAP documents.")
        st.caption(f"Counsellor confidence: {confidence}")
        st.markdown("</div>", unsafe_allow_html=True)

        sources = st.session_state.last_result.get("sources", [])
        trace = st.session_state.last_result.get("trace", [])
        source_col, trace_col = st.columns(2)
        with source_col:
            with st.expander("Sources used"):
                if not sources:
                    st.write("No source metadata was returned.")
                for source in sources:
                    st.write(
                        f"• {source.get('filename')} — page {source.get('page')} "
                        f"({source.get('document_type')})"
                    )
        with trace_col:
            with st.expander("Agent trace"):
                if not trace:
                    st.write("No trace was returned.")
                for item in trace:
                    st.write(f"✓ {item}")
    else:
        st.markdown(
            "<div class='feedback-card muted'>The counsellor's explanation and refinement guidance will appear here.</div>",
            unsafe_allow_html=True,
        )

    question = st.chat_input(
        "Tell the agent how to adjust your list, for example: Include COEP, only Pune colleges, or target lower cutoffs"
    )
    if question:
        submit_question(question, profile)

with admin_tab:
    st.subheader("Admin Knowledge Base")
    key = st.text_input("Admin key", type="password")
    uploads = st.file_uploader(
        "Upload official CAP documents",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
    )
    if st.button("Index selected documents", type="primary"):
        if not uploads:
            st.warning("Select at least one file.")
        else:
            for file in uploads:
                with st.spinner(f"Indexing {file.name}..."):
                    response = requests.post(
                        f"{API}/documents/upload",
                        headers={"X-Admin-Key": key},
                        files={
                            "file": (
                                file.name,
                                file.getvalue(),
                                file.type or "application/octet-stream",
                            )
                        },
                        timeout=300,
                    )
                    if response.ok:
                        st.success(f"Indexed {file.name}: {response.json()['chunks']} chunks")
                    else:
                        st.error(f"{file.name}: {response.text}")
    if st.button("Refresh indexed documents"):
        try:
            st.session_state.docs = requests.get(f"{API}/documents", timeout=30).json()
        except Exception as exc:
            st.error(str(exc))
    for document in st.session_state.get("docs", []):
        st.write(
            f"• **{document['filename']}** — {document['document_type']} — "
            f"{document['chunks']} chunks"
        )
