import pandas as pd
import numpy as np
import scipy.stats as stats
import streamlit as st

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("Statistical Analysis of Meal Photo Queues")

# File upload for yesterday's data and last week's data
uploaded_file_yesterday = st.file_uploader("Upload Yesterday's CSV File", type="csv")
uploaded_file_last_week = st.file_uploader("Upload Last Week's CSV File", type="csv")

if uploaded_file_yesterday and uploaded_file_last_week:
    # Read the uploaded CSVs into DataFrames
    df_yesterday = pd.read_csv(uploaded_file_yesterday)
    df_last_week = pd.read_csv(uploaded_file_last_week)

    # Rename columns (adjust based on your dataset)
    df_yesterday.rename(columns={
        'Meal Photo Queues - Mpq → Access Duration Minutes': 'Access Duration Minutes',
        'Meal Photo Queues - Mpq → Review Duration Minutes': 'Review Duration Minutes',
        'Meal Photo Queues - Mpq → Reviewed Time': 'Reviewed Date',
        'Meal Photo Queue Patient Nth Queue → Patient Nth Queue': 'Patient Nth Queue',
        'Patient Default Clinic - Patient → Clinic Name': 'Clinic Name',
        'Meal Photo Queues - Mpq → Queue Type': 'Queue Type',
        'Meal Photo Queues - Mpq → First Reviewer ID': 'Reviewer ID',
        'Meal Photo Queue N Queues By Same Reviewer - MPQ → N Queues Processed By Same Reviewer': 'N Queues Processed By Same Reviewer'
    }, inplace=True)

    df_last_week.rename(columns={
        'Meal Photo Queues - Mpq → Access Duration Minutes': 'Access Duration Minutes',
        'Meal Photo Queues - Mpq → Review Duration Minutes': 'Review Duration Minutes',
        'Meal Photo Queues - Mpq → Reviewed Time': 'Reviewed Date',
        'Meal Photo Queue Patient Nth Queue → Patient Nth Queue': 'Patient Nth Queue',
        'Patient Default Clinic - Patient → Clinic Name': 'Clinic Name',
        'Meal Photo Queues - Mpq → Queue Type': 'Queue Type',
        'Meal Photo Queues - Mpq → First Reviewer ID': 'Reviewer ID',
        'Meal Photo Queue N Queues By Same Reviewer - MPQ → N Queues Processed By Same Reviewer': 'N Queues Processed By Same Reviewer'
    }, inplace=True)

    # Convert columns to datetime and numeric types
    df_yesterday['Reviewed Date'] = pd.to_datetime(df_yesterday['Reviewed Date'])
    df_last_week['Reviewed Date'] = pd.to_datetime(df_last_week['Reviewed Date'])
    
    df_yesterday['Access Duration Seconds'] = pd.to_numeric(df_yesterday['Access Duration Minutes'], errors='coerce') * 60
    df_yesterday['Review Duration Minutes']  = pd.to_numeric(df_yesterday['Review Duration Minutes'],  errors='coerce')

    df_last_week['Access Duration Seconds'] = pd.to_numeric(df_last_week['Access Duration Minutes'], errors='coerce') * 60
    df_last_week['Review Duration Minutes']  = pd.to_numeric(df_last_week['Review Duration Minutes'],  errors='coerce')

    def calculate_statistics(data):
        mean_val = round(np.mean(data), 2)
        median_val = round(np.median(data), 2)
        sample_size = len(data)
        std_dev = np.std(data, ddof=1)
        se = std_dev / np.sqrt(sample_size) if sample_size > 1 else 0

        ci_80 = stats.norm.interval(0.80, loc=mean_val, scale=se) if sample_size > 1 else (np.nan, np.nan)
        ci_90 = stats.norm.interval(0.90, loc=mean_val, scale=se) if sample_size > 1 else (np.nan, np.nan)
        ci_95 = stats.norm.interval(0.95, loc=mean_val, scale=se) if sample_size > 1 else (np.nan, np.nan)
        
        return {
            "Mean": round(mean_val, 2),
            "Median": round(median_val, 2),
            "Sample Size (n)": sample_size,
            "80% CI": (round(ci_80[0], 2), round(ci_80[1], 2)),
            "90% CI": (round(ci_90[0], 2), round(ci_90[1], 2)),
            "95% CI": (round(ci_95[0], 2), round(ci_95[1], 2))
        }

    # Calculate statistics for yesterday's and last week's data
    stats_yesterday_access = calculate_statistics(df_yesterday['Access Duration Seconds'].dropna())
    stats_yesterday_review = calculate_statistics(df_yesterday['Review Duration Minutes'].dropna())
    stats_last_week_access = calculate_statistics(df_last_week['Access Duration Seconds'].dropna())
    stats_last_week_review = calculate_statistics(df_last_week['Review Duration Minutes'].dropna())
    
    # Create a summary table for yesterday's data
    st.subheader("Yesterday's Data Summary")
    yesterday_table = pd.DataFrame({
        "Metric": ["Mean", "Median", "Sample Size (n)", "80% CI", "90% CI", "95% CI"],
        "Access Duration (s)": [
            f"{stats_yesterday_access['Mean']:.2f}",
            f"{stats_yesterday_access['Median']:.2f}",
            stats_yesterday_access["Sample Size (n)"],
            f"{stats_yesterday_access['80% CI'][0]} - {stats_yesterday_access['80% CI'][1]}",
            f"{stats_yesterday_access['90% CI'][0]} - {stats_yesterday_access['90% CI'][1]}",
            f"{stats_yesterday_access['95% CI'][0]} - {stats_yesterday_access['95% CI'][1]}"
        ],
        "Review Duration (min)": [
            f"{stats_yesterday_review['Mean']:.2f}",
            f"{stats_yesterday_review['Median']:.2f}",
            stats_yesterday_review["Sample Size (n)"],
            f"{stats_yesterday_review['80% CI'][0]} - {stats_yesterday_review['80% CI'][1]}",
            f"{stats_yesterday_review['90% CI'][0]} - {stats_yesterday_review['90% CI'][1]}",
            f"{stats_yesterday_review['95% CI'][0]} - {stats_yesterday_review['95% CI'][1]}"
        ]
    })

    # Center table values using Streamlit styling
    st.table(yesterday_table.style.set_properties(**{'text-align': 'center'}))

    # Calculate differences bewteen yesterday's and last week's data
    def calculate_differences(stats_today, stats_previous):
        return {
            "Mean Difference": round(stats_today["Mean"] - stats_previous["Mean"], 2),
            "Median Difference": round(stats_today["Median"] - stats_previous["Median"], 2),
            "Sample Size Difference": round(stats_today["Sample Size (n)"] - stats_previous["Sample Size (n)"], 2)
        }

    diff_access = calculate_differences(stats_yesterday_access, stats_last_week_access)
    diff_review = calculate_differences(stats_yesterday_review, stats_last_week_review)

    # Add up/down arrows to difference table
    def format_difference(value, opposite=False):
        if value > 0:
            return f"<span style='color: {'green' if opposite else 'red'};'>&uarr; {value:.2f}</span>"
        elif value < 0 :
            return f"<span style='color: {'red' if opposite else 'green'}';'>&darr; {abs(value):.2f}</span>"
        else:
            return f"<span>{value:.2f}</span>"

    difference_table_html = f"""
    <table style="width:100%; text-align:center; border-collapse: collapse;">
      <tr>
          <th>Metric</th>
          <th>Access Duration (s)</th>
          <th>Review Duration (min)</th>
      </tr>
      <tr>
          <td>Mean Difference</td>
          <td>{format_difference(diff_access["Mean Difference"])}</td>
          <td>{format_difference(diff_review["Mean Difference"])}</td>
      </tr>
      <tr>
          <td>Median Difference</td>
          <td>{format_difference(diff_access["Median Difference"])}</td>
          <td>{format_difference(diff_review["Median Difference"])}</td>
      </tr>
      <tr>
          <td>Sample Size Difference</td>
          <td>{format_difference(diff_access["Sample Size Difference"], True)}</td>
          <td>{format_difference(diff_review["Sample Size Difference"], True)}</td>
      </tr>
    </table>
    """
    st.subheader("Difference Between Yesterday and Last Week")
    st.markdown(difference_table_html, unsafe_allow_html=True)


    # Create a "cleaned" version of yesterday's data (no NA in these columns)
    df_cleaned_yesterday = df_yesterday.dropna(
        subset=['Access Duration Minutes', 'Review Duration Minutes']
    ).copy()

    # Identify outliers for yesterday's data using IQR method for Review Duration
    review_duration_latest = df_cleaned_yesterday['Review Duration Minutes']
    outliers_df_latest = df_cleaned_yesterday[review_duration_latest >= 5].copy()

    # Create hyperlinks for MPQ IDs formatted as clickable MPQ ID text.
    outliers_df_latest['Hyperlink'] = outliers_df_latest[
        'Meal Photo Queue Patient Nth Queue → Mpq ID'
    ].apply(lambda x: f'<a href="https://auditor.savoro.app/queue-item/{x}">{x}</a>' if pd.notnull(x) else '')

    # Pivot table by MPQ ID to combine Questionnaire Responses into a single comma-separated string.
    outliers_pivot_df_latest = outliers_df_latest.groupby('Meal Photo Queue Patient Nth Queue → Mpq ID').agg({
        'Hyperlink': 'first',
        'Access Duration Minutes': 'first',
        'Review Duration Minutes': 'first',
        'Net Review Time': 'first',
        'Reviewed Date': 'first',
        'Patient Nth Queue': 'first',
        'Questionnaire Response': lambda x: ', '.join(x.dropna().astype(str)) if 'Questionnaire Response' in outliers_df_latest.columns else '',
        'Questionnaire Value': lambda x: ', '.join(x.dropna().astype(str)) if 'Questionnaire Value' in outliers_df_latest.columns else '',
        'Clinic Name': 'first',
        'Reviewer Name': 'first',
        'Reviewer ID': 'first',
        'N Queues Processed By Same Reviewer': 'first',
        'Review Level': 'first',
        'Queue Type': 'first',
    }).reset_index(drop=True)

    # Sort the pivot table by Net Review Time descending order.
    outliers_sorted_pivot_df_descending = outliers_pivot_df_latest.sort_values(
        by='Review Duration Minutes', 
        ascending=False
    )

    # Rename columns for final HTML
    outliers_sorted_pivot_df_descending.rename(columns={
        'Reviewed Date': 'Reviewer Date',
        'Access Duration Minutes': 'Access Minutes',
        'Review Duration Minutes': 'Review Minutes'
    }, inplace=True)

    # 1) Our CSS block
    styles_html_content = """
    <style>
    table {
        width: 100% !important;
        border-collapse: collapse !important;
        font-size: 12px !important;
    }
    th, td {
        border: 1px solid black !important;
        padding: 6px !important;
        text-align: center !important;
    }
    th {
        background-color: #f2f2f2 !important;
    }
    </style>
    """

    # 2) Show custom CSS so it applies to the table
    st.subheader("Outliers (Review Duration >= 5 min)")
    st.markdown(styles_html_content, unsafe_allow_html=True)

    # 3) Generate HTML for the outliers table
    outliers_html = outliers_sorted_pivot_df_descending.to_html(escape=False, index=False)

    # 4) Render table HTML
    st.markdown(outliers_html, unsafe_allow_html=True)

    # New Outliers2: Filter items with Access >= 1.5 min and Review <= 1.5 min.
    access_duration_threshold_lower_bound = 1.5  # in minutes
    outliers2_df = df_cleaned_yesterday[df_cleaned_yesterday['Access Duration Minutes'] >= access_duration_threshold_lower_bound].copy()

    # Create hyperlinks for MPQ IDs formatted as clickable MPQ ID text.
    outliers2_df['Hyperlink'] = outliers2_df[
        'Meal Photo Queue Patient Nth Queue → Mpq ID'
    ].apply(lambda x: f'<a href="https://auditor.savoro.app/queue-item/{x}">{x}</a>' if pd.notnull(x) else '')

    outliers2_pivot_df = outliers2_df.groupby('Meal Photo Queue Patient Nth Queue → Mpq ID').agg({
        'Hyperlink': 'first',
        'Access Duration Minutes': 'first',
        'Review Duration Minutes': 'first',
        'Net Review Time': 'first',
        'Reviewed Date': 'first',
        'Patient Nth Queue': 'first',
        'Questionnaire Response': lambda x: ', '.join(x.dropna().astype(str)) if 'Questionnaire Response' in outliers2_df.columns else '',
        'Questionnaire Value': lambda x: ', '.join(x.dropna().astype(str)) if 'Questionnaire Value' in outliers2_df.columns else '',
        'Clinic Name': 'first',
        'Reviewer Name': 'first',
        'Reviewer ID': 'first',
        'N Queues Processed By Same Reviewer': 'first',
        'Review Level': 'first',
        'Queue Type': 'first',
    }).reset_index(drop=True)

    outliers2_sorted_pivot_df_descending = outliers2_pivot_df.sort_values(
        by='Access Duration Minutes', 
        ascending=False
    )

    st.subheader("Outliers (Access Duration >= 1.5 min)")
    st.markdown(styles_html_content, unsafe_allow_html=True)

    outliers2_html = outliers2_sorted_pivot_df_descending.to_html(
        escape=False, 
        index=False
    )
    st.markdown(outliers2_html, unsafe_allow_html=True)

else:
    st.warning("Please upload both CSV files to proceed.")
