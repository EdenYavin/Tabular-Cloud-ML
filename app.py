import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Experiment Dashboard")
st.title("Experiment Results Comparison Dashboard")


# --- 1. Helper Functions ---
def clean_cloud_column(val):
    """
    Robustly determines if a row is 'Cloud' or 'No Cloud'.
    """
    s_val = str(val).strip().lower()
    if s_val == 'nan' or s_val == '[]' or s_val == '' or s_val == 'none':
        return "No Cloud"
    return "With Cloud"


def clean_model_name(val):
    """
    Cleans the cloud model string for display in dropdown.
    e.g. "['xception']" -> "xception"
    """
    s = str(val)
    for char in ["['", "']", '["', '"]', "'", '"', "[", "]"]:
        s = s.replace(char, "")
    return s


# --- 2. Upload Section ---
st.sidebar.header("1. Upload Data")
uploaded_files = st.sidebar.file_uploader("Upload CSV files", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)

        # --- 3. Preprocessing ---
        # Normalize cloud models column
        cloud_col = None
        if 'cloud_models' in df_all.columns:
            cloud_col = 'cloud_models'
        elif 'cloud_model' in df_all.columns:
            cloud_col = 'cloud_model'

        if cloud_col:
            df_all['cloud_status'] = df_all[cloud_col].apply(clean_cloud_column)
        else:
            st.error("Could not find a 'cloud_models' column. Please check your CSV format.")
            st.stop()

        # Handle NaNs in key columns to prevent filtering issues
        # We replace NaNs with "N/A" so they show up clearly in dropdowns
        filter_cols = ['dataset_name', 'iim_name', 'triangulation_embedding', 'triangulation_mode']
        for col in filter_cols:
            if col in df_all.columns:
                df_all[col] = df_all[col].fillna('N/A').astype(str)

        # --- 4. Configuration ---
        st.sidebar.divider()
        st.sidebar.header("2. Configuration")

        # Metric Selector
        metric_type = st.sidebar.radio("Metric", ["Accuracy", "AUC"], horizontal=True)
        metric_substring = 'acc' if metric_type == "Accuracy" else 'auc'

        # Aggregation Selector
        agg_method = st.sidebar.radio("Aggregation (over k-folds)", ["Mean", "Max", "Min"], horizontal=True)

        # Calculate Metric
        metric_cols = [c for c in df_all.columns if metric_substring in c.lower() and 'k_fold' in c.lower()]

        if not metric_cols:
            st.warning(f"No columns found for {metric_type}. Checking for columns containing '{metric_substring}'...")
            df_all['calculated_metric'] = 0
        else:
            if agg_method == "Mean":
                df_all['calculated_metric'] = df_all[metric_cols].mean(axis=1)
            elif agg_method == "Max":
                df_all['calculated_metric'] = df_all[metric_cols].max(axis=1)
            elif agg_method == "Min":
                df_all['calculated_metric'] = df_all[metric_cols].min(axis=1)

        # --- 5. Filters (With "All" Option) ---
        st.sidebar.divider()
        st.sidebar.header("3. Filters")

        # Dataset Filter
        datasets = sorted(df_all['dataset_name'].unique())
        datasets.insert(0, "All")
        dataset = st.sidebar.selectbox("Dataset", datasets)

        if dataset != "All":
            filtered_df = df_all[df_all['dataset_name'] == dataset]
        else:
            filtered_df = df_all

        # IIM Filter
        if 'iim_name' in filtered_df.columns:
            iims = sorted(filtered_df['iim_name'].unique())
            iims.insert(0, "All")
            iim = st.sidebar.selectbox("IIM Name", iims)
            if iim != "All":
                filtered_df = filtered_df[filtered_df['iim_name'] == iim]

        # Embedding Filter
        if 'triangulation_embedding' in filtered_df.columns:
            embeddings = sorted(filtered_df['triangulation_embedding'].unique())
            embeddings.insert(0, "All")
            embedding = st.sidebar.selectbox("Triangulation Embedding", embeddings)
            if embedding != "All":
                filtered_df = filtered_df[filtered_df['triangulation_embedding'] == embedding]

        # Mode Filter
        if 'triangulation_mode' in filtered_df.columns:
            modes = sorted(filtered_df['triangulation_mode'].unique())
            modes.insert(0, "All")
            mode = st.sidebar.selectbox("Triangulation Mode", modes)
            if mode != "All":
                filtered_df = filtered_df[filtered_df['triangulation_mode'] == mode]

        # --- Cloud Model Filter (ADDED) ---
        available_cloud_rows = filtered_df[filtered_df['cloud_status'] == 'With Cloud']

        if not available_cloud_rows.empty:
            unique_models = sorted(available_cloud_rows[cloud_col].unique())
            unique_models.insert(0, "All")

            cloud_model_selection = st.sidebar.selectbox(
                "Cloud Model Configuration",
                unique_models,
                format_func=lambda x: "All" if x == "All" else clean_model_name(x)
            )
        else:
            cloud_model_selection = "All"
            st.sidebar.info("No cloud models available in current selection.")

        # --- 6. Results Display ---
        st.divider()
        title_parts = [p for p in [dataset, iim, embedding, mode] if p != "All"]
        if cloud_model_selection != "All":
            title_parts.append(clean_model_name(cloud_model_selection))

        title_str = " | ".join(title_parts) if title_parts else "All Data"
        st.header(f"Results: {title_str}")

        # Split into Cloud and No Cloud
        nocloud_df = filtered_df[filtered_df['cloud_status'] == 'No Cloud']
        cloud_df_raw = filtered_df[filtered_df['cloud_status'] == 'With Cloud']

        # Apply Cloud Model Filter
        if cloud_model_selection != "All":
            cloud_df = cloud_df_raw[cloud_df_raw[cloud_col] == cloud_model_selection]
        else:
            cloud_df = cloud_df_raw

        # Calculate metrics
        val_cloud = cloud_df['calculated_metric'].mean() if not cloud_df.empty else None
        val_nocloud = nocloud_df['calculated_metric'].mean() if not nocloud_df.empty else None

        col1, col2 = st.columns(2)

        # Display Logic
        display_cols = ['dataset_name', 'iim_name', 'triangulation_embedding', 'triangulation_mode',
                        cloud_col] + metric_cols
        display_cols = [c for c in display_cols if c in df_all.columns]

        with col1:
            st.subheader("No Cloud")
            if not nocloud_df.empty:
                if val_nocloud is not None:
                    st.metric(f"{agg_method} {metric_type}", f"{val_nocloud:.4f}")
                st.dataframe(nocloud_df[display_cols], height=200, use_container_width=True)
            else:
                st.info("No matching baseline (No Cloud) data.")

        with col2:
            st.subheader(
                f"With Cloud ({'All' if cloud_model_selection == 'All' else clean_model_name(cloud_model_selection)})")
            if not cloud_df.empty:
                if val_cloud is not None:
                    st.metric(f"{agg_method} {metric_type}", f"{val_cloud:.4f}")
                st.dataframe(cloud_df[display_cols], height=200, use_container_width=True)
            else:
                st.info("No matching Cloud data.")

        # --- 7. Comparison ---
        if val_cloud is not None and val_nocloud is not None:
            st.divider()
            diff = val_cloud - val_nocloud

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.metric(f"Improvement (Cloud - No Cloud)", f"{diff:.4f}", delta=diff)

            # Chart
            plot_df = pd.DataFrame({
                'Setting': ['No Cloud', 'With Cloud'],
                'Score': [val_nocloud, val_cloud]
            })

            y_min = min(val_nocloud, val_cloud)
            y_max = max(val_nocloud, val_cloud)
            padding = (y_max - y_min) * 0.2 if y_max != y_min else 0.05

            fig = px.bar(plot_df, x='Setting', y='Score', color='Setting',
                         title=f"{metric_type} Comparison", text_auto='.4f',
                         range_y=[max(0, y_min - padding), min(1.0, y_max + padding)])
            st.plotly_chart(fig, use_container_width=True)

        elif val_cloud is None and val_nocloud is None:
            st.warning("No data found. Try setting filters to 'All'.")
else:
    st.info("Please upload your CSV files to start.")