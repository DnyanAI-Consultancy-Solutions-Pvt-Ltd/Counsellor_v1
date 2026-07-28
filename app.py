from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

CATEGORY_OPTIONS = [
    "OPEN", "OBC", "SC", "ST", "EWS", "NT-A", "NT-B", "NT-C", "NT-D", "VJ", "SBC"
]
BRANCH_OPTIONS = [
    "Any",
    "Computer Engineering",
    "Computer Science and Engineering",
    "Information Technology",
    "Artificial Intelligence",
    "Artificial Intelligence and Data Science",
    "Data Science",
    "Cyber Security",
    "Computer Science and Business Systems",
    "Electronics and Telecommunication Engineering",
    "Electronics Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Instrumentation Engineering",
    "Robotics and Automation",
]
SEAT_TYPE_OPTIONS = [
    "Any", "Government", "Government Aided", "Autonomous", "Private", "Minority"
]
LOCATION_OPTIONS = [
    "Any", "Pune", "Mumbai", "Navi Mumbai", "Thane", "Nagpur", "Nashik",
    "Kolhapur", "Sangli", "Satara", "Chhatrapati Sambhajinagar", "Amravati",
    "Solapur", "Ahmednagar"
]
COLLEGE_PREFERENCE_OPTIONS = [
    "No Preference", "Top Colleges Only", "Government Colleges",
    "Government + Autonomous", "Private Colleges", "Minority Colleges"
]
GENDER_OPTIONS = ["Not Specified", "Male", "Female", "Other", "Prefer Not To Say"]
HOME_UNIVERSITY_OPTIONS = [
    "No Preference",
    "Savitribai Phule Pune University",
    "University of Mumbai",
    "Shivaji University",
    "Dr. Babasaheb Ambedkar Marathwada University",
    "Rashtrasant Tukadoji Maharaj Nagpur University",
    "Kavayitri Bahinabai Chaudhari North Maharashtra University",
    "Sant Gadge Baba Amravati University",
    "SNDT Women's University",
]
COLLEGE_COUNT_OPTIONS = [10, 20, 30, 40, 50]


def _without_any(values: list[str]) -> list[str]:
    """Treat 'Any' as no restriction when sending the profile to the API."""
    return [] if "Any" in values else values


st.set_page_config(page_title="MHT-CET Agentic RAG Counsellor", page_icon="🎓", layout="wide")
st.title("🎓 MHT-CET Agentic RAG Counsellor V2")
st.caption("Upload cutoff evidence, generate recommendations, and refine the result through feedback.")

with st.sidebar:
    st.header("Knowledge Base")
    files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md"],
    )
    document_type = st.selectbox("Document type", ["cutoff", "general"])
    if st.button("Upload and Index", use_container_width=True):
        if not files:
            st.warning("Select at least one file.")
        else:
            payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
            response = requests.post(
                f"{API_URL}/upload",
                files=payload,
                data={"document_type": document_type},
                timeout=1800,
            )
            response.raise_for_status()
            st.success("Documents indexed.")
            st.json(response.json())

st.subheader("Candidate Profile")
col1, col2, col3 = st.columns(3)

with col1:
    percentile = st.number_input("Percentile", 0.0, 100.0, 90.0, 0.01)
    category = st.selectbox("Category", CATEGORY_OPTIONS, index=0)
    gender = st.selectbox("Gender", GENDER_OPTIONS, index=0)

with col2:
    branches = st.multiselect(
        "Preferred branches",
        BRANCH_OPTIONS,
        default=["Computer Engineering", "Information Technology"],
        help="Select one or more branches. Select Any to remove the branch restriction.",
    )
    locations = st.multiselect(
        "Preferred locations",
        LOCATION_OPTIONS,
        default=[],
        help="Leave empty or select Any when location is not a restriction.",
    )
    home_university = st.selectbox("Home university", HOME_UNIVERSITY_OPTIONS, index=0)

with col3:
    seat_type = st.selectbox("Seat type", SEAT_TYPE_OPTIONS, index=0)
    college_preference = st.selectbox(
        "College preference", COLLEGE_PREFERENCE_OPTIONS, index=0
    )
    college_count = st.selectbox(
        "College count",
        COLLEGE_COUNT_OPTIONS,
        index=2,
        help="The agent will try to produce this many unique, evidence-backed choices.",
    )

user_request = st.text_area(
    "Additional request",
    "Generate a balanced ranked preference list.",
    height=100,
)

if st.button("Generate Recommendations", type="primary"):
    selected_branches = _without_any(branches)
    selected_locations = _without_any(locations)
    body = {
        "student_profile": {
            "percentile": percentile,
            "category": category,
            "gender": None if gender == "Not Specified" else gender,
            "preferred_branches": selected_branches,
            "preferred_locations": selected_locations,
            "home_university": None if home_university == "No Preference" else home_university,
            "seat_type": None if seat_type == "Any" else seat_type,
            "college_preference": (
                None if college_preference == "No Preference" else college_preference
            ),
            "college_count": college_count,
        },
        "user_request": user_request,
    }
    with st.spinner("Counsellor Agent is analysing retrieved evidence..."):
        response = requests.post(f"{API_URL}/counsel", json=body, timeout=1200)
        response.raise_for_status()
        st.session_state.result = response.json()

result = st.session_state.get("result")
if result:
    st.subheader("Counselling Result")
    st.write(result.get("summary", ""))
    if result.get("strategy"):
        st.info(result["strategy"])

    requested_count = result.get("requested_college_count")
    actual_count = len(result.get("recommendations", []))
    if requested_count is not None:
        st.caption(f"Requested colleges: {requested_count} | Generated colleges: {actual_count}")

    recommendations = result.get("recommendations", [])
    if recommendations:
        display_df = pd.DataFrame(recommendations)
        preferred_columns = [
            "rank", "zone", "college", "branch", "location",
            "category_or_seat_type", "seat_allocation",
            "historical_cutoff", "student_percentile", "cutoff_gap", "reason",
        ]
        visible_columns = [column for column in preferred_columns if column in display_df.columns]
        remaining_columns = [column for column in display_df.columns if column not in visible_columns and column != "evidence_ids"]
        st.dataframe(
            display_df[visible_columns + remaining_columns],
            use_container_width=True,
            hide_index=True,
        )

    if result.get("evidence_warning"):
        st.warning(result["evidence_warning"])

    download_url = result.get("excel_download_url")
    if download_url:
        response = requests.get(f"{API_URL}{download_url}", timeout=120)
        response.raise_for_status()
        st.download_button(
            "Download Excel",
            response.content,
            file_name=download_url.rsplit("/", 1)[-1],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Feedback")
    feedback_text = st.text_area(
        "Example: remove colleges outside Pune and rerank the remaining choices."
    )
    if st.button("Apply Feedback"):
        response = requests.post(
            f"{API_URL}/feedback",
            json={"session_id": result["session_id"], "feedback": feedback_text},
            timeout=1200,
        )
        response.raise_for_status()
        st.session_state.result = response.json()
        st.rerun()
