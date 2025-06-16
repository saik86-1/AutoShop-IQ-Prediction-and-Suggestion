import streamlit as st
import os
import re
import chromadb
from openai import OpenAI

# === CONFIG ===
client_openai = OpenAI()
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4"
DB_DIR = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/chroma_repair_orders"
COLLECTION_NAME = "repair_orders"

# === INITIALIZE CHROMADB ===
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

st.set_page_config(page_title="AutoShot IQ Repair Assistant", layout="wide", page_icon="🚗")

# === HEADER ===
st.markdown("""
    <div style='display: flex; align-items: center; justify-content: space-between;'>
        <h1 style='font-size: 2.5rem;'>🚘 AutoShot IQ</h1>
        <h2 style='font-size: 1.2rem;'>🔧 Vehicle Repair Diagnostic Assistant</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    "Describe your car's issue and we’ll suggest the top 3 possible repair solutions with estimated cost and parts."
)

repair_summary = ""

with st.form(key="vehicle_form"):
    concern = st.text_input("🔍 What's the issue with your vehicle?", placeholder="e.g. Car shakes when braking")
    col1, col2, col3 = st.columns(3)
    with col1:
        make = st.text_input("🚘 Make", placeholder="Toyota")
    with col2:
        model = st.text_input("📄 Model", placeholder="Camry")
    with col3:
        year = st.number_input("📆 Year", min_value=1990, max_value=2025, step=1)
    submitted = st.form_submit_button("🚗 Get Repair Recommendations")

if submitted and concern and make and model and year:
    query = f"{concern} {make} {model} {int(year)}"
    pattern = r"(.*)\s+(\w+)\s+(\w+)\s+(\d{4})$"
    match = re.match(pattern, query.strip())
    if match:
        concern_text, make, model, year = match.group(1).strip().lower(), match.group(2).lower(), match.group(3).lower(), int(match.group(4))

        with st.spinner("🚗 Diagnosing and fetching repair suggestions..."):
            embedding = client_openai.embeddings.create(model=EMBED_MODEL, input=[concern_text]).data[0].embedding
            results = collection.query(query_embeddings=[embedding], n_results=5)

            retrieved_context = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                retrieved_context.append(
                    f"Concern: {meta.get('customer_concerns', 'N/A')}\n"
                    f"Vehicle: {meta.get('vehicle_make')} {meta.get('vehicle_model')} {meta.get('vehicle_year')}\n"
                )

            prompt = f"""
            You are an automotive repair assistant.
            User reports: \"{concern_text}\"
            Vehicle: {make.title()} {model.title()} {year}

            Retrieved Similar Cases:
            {''.join(retrieved_context)}

            Based on the cases above, generate the top 3 repair recommendations in the following structured format:

            1. **<Job Title>**
               - Parts Required: <Part Name 1> (Part ID), <Part Name 2> (Part ID)
               - Estimated Labor Cost: $<amount>
               - Estimated Labor Hours: <hours>
               - Confidence: <probability %>

            Only return the structured repair plan.
            """

            response = client_openai.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful automotive repair assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

        st.markdown("## 🔎 Top 3 Repair Solutions")
        repair_summary = response.choices[0].message.content.strip()

        top_distances = results["distances"][0]
        max_distance = max(top_distances)
        min_distance = min(top_distances)
        range_distance = max_distance - min_distance or 1

        for i, block in enumerate(repair_summary.split("\n\n")):
            if block.strip():
                similarity_score = 1 - top_distances[i]
                norm_conf = int((similarity_score - (1 - max_distance)) / range_distance * 100)
                norm_conf = max(0, min(100, norm_conf))

                cols = st.columns([5, 1])
                with cols[0]:
                    st.markdown(block)
                with cols[1]:
                    st.progress(norm_conf / 100)

                    if norm_conf > 75:
                        icon = "👍👍"
                    elif norm_conf >= 50:
                        icon = "👍"
                    else:
                        icon = "<span style='color:red;'>❗</span>"

                    st.markdown(f"<div style='font-size:1.1rem; font-weight:600;'>{icon} {norm_conf}%</div>", unsafe_allow_html=True)

        # store summary into session state to make accessible to chat
        st.session_state.repair_output = repair_summary

# === TECHNICIAN CHAT ASSISTANT ===
st.markdown("---")
st.markdown("## 🧑‍🔧 Technician Assistant Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a technical assistant for vehicle repair staff. Help with repair steps, parts compatibility, availability, and any clarification related to the diagnosis. Use the repair suggestion summary if available."}
    ]

if "repair_output" in st.session_state and not any("repair_output" in msg.get("content", "") for msg in st.session_state.chat_history):
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"Repair summary to keep in mind:\n{st.session_state.repair_output}"
    })

user_input = st.chat_input("Ask a technical question, e.g., 'Are the brake pads BP2018 compatible with Camry 2018?'...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    chat_response = client_openai.chat.completions.create(
        model=GPT_MODEL,
        messages=st.session_state.chat_history
    )
    assistant_message = chat_response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})

for msg in st.session_state.chat_history[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])