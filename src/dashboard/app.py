"""
EvoFinGraph Dashboard - Integration version
Investigation Platform Engineer (ECE) deliverable.

Calls the FastAPI service, which now queries real Neo4j data
(CSE's actual community detection, AFEI, and tracking results)
instead of mock data.
"""

import requests
import streamlit as st
import pandas as pd

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="EvoFinGraph Investigation Dashboard", layout="wide")
st.title("EvoFinGraph — Investigation Dashboard")

LABEL_MAP = {"0": "licit", "1": "illicit", "unknown": "unknown"}

# -----------------------------------------------------------------------
# Sidebar: pick which community to investigate
# -----------------------------------------------------------------------
st.sidebar.header("Select Community")
pcid_input = st.sidebar.text_input("PCID (numeric)", value="161")
fetch_clicked = st.sidebar.button("Fetch Community")


def fetch_json(endpoint: str):
    """
    Calls the FastAPI backend and handles errors gracefully.
    Returns (data, error_message) - only one will be non-None.
    """
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Is uvicorn running on port 8000?"

    if response.status_code == 404:
        return None, response.json().get("detail", "Not found")
    if response.status_code != 200:
        return None, f"Unexpected error (status {response.status_code})"

    return response.json(), None


# -----------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------
if fetch_clicked:
    try:
        pcid = int(pcid_input)
    except ValueError:
        st.error("PCID must be a number (e.g. 161).")
        pcid = None

    if pcid is not None:
        community, err = fetch_json(f"/community/{pcid}")

        if err:
            st.error(err)
        else:
            # --- Community statistics ---
            st.subheader(f"Community {community['pcid']} — Snapshot {community['timeStep']}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lifetime (snapshots)", community["lifetime"])
            col2.metric("Density Change", f"{community['densityChange']:+.2f}")
            col3.metric("Fraud Growth", f"{community['fraudGrowth']:+.2f}")
            col4.metric("Growth Rate", f"{community['growthRate']:+.2f}")

            col5, col6 = st.columns(2)
            col5.metric("AFEI (Adaptive Fraud Evolution Index)", f"{community['afei']:.2f}")
            col6.metric("Risk Score", f"{community['riskScore']:.2f}")

            if community.get("isFraudCommunity"):
                st.warning("This community is flagged as a confirmed fraud ring.")

            # --- Timeline ---
            st.subheader("Timeline")
            timeline_data, timeline_err = fetch_json(f"/timeline/{pcid}")

            if timeline_err:
                st.warning(f"Timeline unavailable: {timeline_err}")
            else:
                df = pd.DataFrame(timeline_data["timeline"])
                st.line_chart(df.set_index("timeStep")[["afei", "riskScore"]])
                st.dataframe(df, use_container_width=True)

            # --- Community members ---
            st.subheader("Community Members")
            graph_data, graph_err = fetch_json(f"/graph/{pcid}")

            if graph_err:
                st.warning(f"Member list unavailable: {graph_err}")
            else:
                members_df = pd.DataFrame(graph_data["nodes"])
                members_df["label"] = members_df["label"].map(LABEL_MAP).fillna(members_df["label"])
                st.dataframe(members_df, use_container_width=True)
                st.caption(f"{len(graph_data['edges'])} internal transaction flows in this snapshot")

else:
    st.info("Enter a numeric PCID in the sidebar (e.g. 161) and click 'Fetch Community' to begin.")

# -----------------------------------------------------------------------
# Alerts panel - always visible, independent of the selected community
# -----------------------------------------------------------------------
st.divider()
st.subheader("Active Alerts (Confirmed Fraud Communities)")

alerts_data, alerts_err = fetch_json("/alerts")

if alerts_err:
    st.warning(f"Alerts unavailable: {alerts_err}")
elif not alerts_data["alerts"]:
    st.success("No high-risk fraud communities currently flagged.")
else:
    alerts_df = pd.DataFrame(alerts_data["alerts"])
    st.dataframe(alerts_df, use_container_width=True)