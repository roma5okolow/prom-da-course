import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="NER Demo", page_icon="🏷️")

st.title("NER Model Interface")
st.markdown("Получите разбор предложения на именованные сущности.")

# Text area for input
input_text = st.text_area(
    "Входной текст", placeholder="Введите текст на русском языке...", height=200
)

if st.button("Extract Entities", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://backend:8000/forward", json={"text": input_text}, timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    result = data.get("result", [])

                    if result:
                        st.subheader("Results")
                        df = pd.DataFrame(result)
                        st.table(df)

                        with st.expander("Show Raw JSON"):
                            st.json(data)
                    else:
                        st.info("No entities found.")
                else:
                    st.error(f"Backend error: {response.status_code}")

            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

if st.sidebar.button("Get Model Metadata"):
    try:
        meta_res = requests.get("http://backend:8000/metadata")
        st.sidebar.json(meta_res.json())
    except Exception:
        st.sidebar.error("Metadata unavailable")
