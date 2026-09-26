import streamlit as st
import pandas as pd

from modules.cache_manager import CacheManager

# =====================================================
# PAGE TITLE
# =====================================================

st.set_page_config(
    page_title="Centrality Comparison",
    layout="wide"
)

st.title("📊 Centrality Comparison")

# =====================================================
# DATASET SELECTION
# =====================================================

dataset = st.selectbox(
    "Choose Dataset",
    [
        "Karate Club",
        "Email-Eu-Core",
        "Web-Google"
    ]
)

dataset_map = {
    "Karate Club": "karate",
    "Email-Eu-Core": "email",
    "Web-Google": "web_google"
}

dataset_key = dataset_map[dataset]

# =====================================================
# CHECK CACHE
# =====================================================

required_files = [
    "stats.json",
    "centrality.csv",
    "correlation.json"
]

missing = [
    file for file in required_files
    if not CacheManager.exists(dataset_key, file)
]

if missing:

    st.warning(
        "⚠ Please run this dataset once from the Home page using 'Run Analysis' to generate the cached results."
    )

    st.stop()

# =====================================================
# LOAD FROM CACHE
# =====================================================

stats = CacheManager.load_json(
    dataset_key,
    "stats.json"
)

centrality = CacheManager.load_csv(
    dataset_key,
    "centrality.csv"
)

corr = pd.DataFrame(
    CacheManager.load_json(
        dataset_key,
        "correlation.json"
    )
)

# =====================================================
# LOAD TOP 10
# =====================================================

try:

    if CacheManager.exists(dataset_key, "top10.csv"):

        top10 = CacheManager.load_csv(
            dataset_key,
            "top10.csv"
        )

    else:

        raise FileNotFoundError

except Exception:

    top10 = centrality.sort_values(
        "Degree Centrality",
        ascending=False
    ).head(10)

    CacheManager.save_csv(
        dataset_key,
        "top10.csv",
        top10
    )

# =====================================================
# DISPLAY RESULTS
# =====================================================

st.success("⚡ Loaded from disk cache")

st.subheader("📌 Graph Statistics")

stats_df = pd.DataFrame({
    "Metric": stats.keys(),
    "Value": stats.values()
})

st.dataframe(
    stats_df,
    use_container_width=True
)

# =====================================================

st.subheader("📈 Centrality Scores")

st.dataframe(
    centrality,
    use_container_width=True
)

# =====================================================

st.subheader("🏆 Top 10 Degree Centrality")

st.dataframe(
    top10,
    use_container_width=True
)

# =====================================================

st.subheader("🔗 Spearman Correlation")

st.dataframe(
    corr,
    use_container_width=True
)

# =====================================================

st.success("✅ No computations performed. Results loaded directly from cache.")