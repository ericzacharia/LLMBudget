import re
from datetime import datetime

import openai
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import fuzz, process
from langchain_community.llms import Ollama

from openai_token import openai_api_key


def get_recurring_transactions(df, num_months=2, tolerance=0.20, similarity_threshold=78):
    """Finds recurring transactions in a DataFrame.
    Returns: DataFrame with recurring transactions and a list of recurring transaction descriptions"""

    def group_similar_descriptions(df, similarity_threshold=78):
        """Groups similar descriptions together using the same description in the GroupedDescription column based on a similarity
        threshold"""

        def preprocess_description(description):
            """Uses regular expression to remove numbers (or any unique identifiers) and strip leading/trailing whitespace"""
            return re.sub(r"\d+", "", description).strip()

        def map_to_group(desc):
            processed_desc = preprocess_description(desc)
            for key, values in similar_groups.items():
                if processed_desc in values:
                    return key
            return desc

        df["ProcessedDescription"] = df["Description"].apply(preprocess_description)
        descriptions = df["ProcessedDescription"].unique()

        # Create a mapping of similar descriptions
        similar_groups = {}
        for desc in descriptions:
            matched_desc = process.extractOne(desc, similar_groups.keys(), scorer=fuzz.token_sort_ratio)
            if matched_desc and matched_desc[1] >= similarity_threshold:
                similar_groups[matched_desc[0]].append(desc)
            else:
                similar_groups[desc] = [desc]

        df["GroupedDescription"] = df["Description"].apply(map_to_group)
        return df

    # Convert 'Transaction Date' to datetime and preprocess descriptions
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    df = group_similar_descriptions(df, similarity_threshold=similarity_threshold)

    # Group by description and year/month and count occurrences
    df["YearMonth"] = df["Transaction Date"].dt.to_period("M").astype(str)
    grouped = df.groupby(["GroupedDescription", "YearMonth"]).size().reset_index(name="Count")

    # Count the number of months each description appears in
    recurring = grouped.groupby(["GroupedDescription"]).size().reset_index(name="MonthCount")

    # Filter by the minimum number of months for recurring transactions
    recurring_transaction_descriptions = recurring[recurring["MonthCount"] >= num_months]["GroupedDescription"].tolist()
    # Filter DataFrame by recurring expenses
    df = df[df["GroupedDescription"].isin(recurring_transaction_descriptions)]

    # Identify recurring transactions based on amounts and date differences
    recurring_transactions = []
    grouped = df.groupby("GroupedDescription")

    for name, group in grouped:
        group = group.sort_values(by="Transaction Date")
        amounts = group["Amount"].values
        dates = group["Transaction Date"].values

        for i in range(len(amounts) - 1):
            # Check if the amounts are within the tolerance level
            if abs(amounts[i] - amounts[i + 1]) <= tolerance * abs(amounts[i]):
                # Check for monthly or bi-weekly patterns
                diff = (dates[i + 1] - dates[i]).astype("timedelta64[D]").item().days
                if (28 <= diff <= 32) or (13 <= diff <= 17):
                    recurring_transactions.append(group.iloc[i])
                    recurring_transactions.append(group.iloc[i + 1])
    return pd.DataFrame(recurring_transactions).drop_duplicates(), recurring_transaction_descriptions


def statement_to_recurring_transactions_df(file_path, num_months, tolerance, date_range):
    """
    Standardizes statement file columns, places recurring transactions into a DataFrame,
    and also returns all transactions from the statement.
    """

    def ai_classify_transactions(df):
        """
        Classify transaction descriptions for statements that don't have a category column using our trained SVM model.
        """
        # Convert the descriptions to a list
        descriptions = df["Description"].tolist()
        # Vectorize the descriptions using the loaded TF-IDF vectorizer
        descriptions_tfidf = vectorizer.transform(descriptions)
        # Predict the categories using the SVM model
        categories = svm_clf.predict(descriptions_tfidf)
        # Ensure all categories are valid, fallback to "Transfer" if necessary
        valid_categories = ["Income", "Housing", "Utility", "Transfer", "P2P Expense"]
        categories = [category if category in valid_categories else "Transfer" for category in categories]
        return categories

    # Read in the CSV file
    df = pd.read_csv(file_path)

    # Filter data based on the selected date range
    start_date, end_date = [datetime.combine(d, datetime.min.time()) for d in date_range]
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    df = df[(df["Transaction Date"] >= start_date) & (df["Transaction Date"] <= end_date)]

    # Standardize file format depending on whether it has a 'Debit' column or not
    if "Debit" not in df.columns:
        # Process files without a 'Debit' column by calculating the amount based on the transaction type
        df["Amount"] = df.apply(
            lambda x: (-x["Transaction Amount"] if x["Transaction Type"] == "Debit" else x["Transaction Amount"]),
            axis=1,
        )
        df = df.drop(columns=["Account Number", "Balance", "Transaction Amount", "Transaction Type"])
        df = df.rename(columns={"Transaction Description": "Description"})
    else:
        # Process files with a 'Debit' column by inverting the amounts to reflect expenses as negative
        df = df.dropna(subset=["Debit"])
        df = df.rename(columns={"Debit": "Amount"})
        df = df.drop(columns=["Credit", "Card No.", "Posted Date"])

        # Ensure that "Amount" is negative for expenses
        df["Amount"] = df["Amount"] * -1

    # Ensure the 'Category' column exists, assigning "Transfer" as the default value if it doesn't
    if "Category" not in df.columns:
        # use the AI model to classify the transaction description into a category
        df["Category"] = ai_classify_transactions(df)
        # df["Category"] = "Transfer"
    # Store all transactions in all_transactions_df (no filtering for recurring transactions)
    all_transactions_df = df.copy()

    # Find recurring transactions
    recurring_transactions_df, recurring_transaction_descriptions = get_recurring_transactions(
        df, num_months=num_months, tolerance=tolerance
    )

    # Shorten the name of the recurring transactions DataFrame for clarity
    rt_df = recurring_transactions_df

    # Make a copy of the recurring transactions DataFrame
    rt_df_copy = recurring_transactions_df.copy()

    if len(recurring_transaction_descriptions) == 0:
        return all_transactions_df, None, rt_df_copy

    # Calculate recurring income and expenses
    rt_df["Recurring Income"] = rt_df[rt_df["Amount"] > 0].groupby("YearMonth")["Amount"].transform("sum")
    rt_df["Recurring Expenses"] = rt_df[rt_df["Amount"] < 0].groupby("YearMonth")["Amount"].transform("sum")

    # Aggregate recurring transaction details by month
    monthly_recurring_transactions_df = (
        rt_df.groupby("YearMonth")
        .agg(
            {
                "Description": lambda x: list(x.unique()),
                "Amount": "sum",
                "Recurring Income": "first",
                "Recurring Expenses": "first",
                "Category": "first",
            }
        )
        .reset_index()
    )

    return all_transactions_df, monthly_recurring_transactions_df, rt_df_copy


def combine_statement_dfs(uploaded_statements, date_range, num_months=2, tolerance=0.20):
    """
    Combines the dataframes from each uploaded statement into a single DataFrame.
    """

    def concat_descriptions(descriptions):
        # Concatenate a list of descriptions into a single string
        return ", ".join(sum(descriptions, []))

    # Initialize empty DataFrames
    aggregated_recurring_transactions_df = pd.DataFrame()  # Aggregated recurring transactions by YearMonth
    all_recurring_transactions_df = pd.DataFrame()  # Contains all recurring transactions ungrouped
    all_transactions_df = pd.DataFrame()  # Contains all transactions from all statements

    for uploaded_statement in uploaded_statements:
        # statement_to_recurring_transactions_df should return all transactions and recurring transactions
        all_transactions, monthly_recurring_transactions_df, rt_df_copy = statement_to_recurring_transactions_df(
            uploaded_statement,
            num_months,
            tolerance,
            date_range,
        )

        if all_transactions is not None:
            # Concatenate all transactions into all_transactions_df
            all_transactions_df = pd.concat([all_transactions_df, all_transactions], ignore_index=True)

        if monthly_recurring_transactions_df is not None:
            # Concatenate recurring transactions and their aggregated versions
            aggregated_recurring_transactions_df = pd.concat(
                [aggregated_recurring_transactions_df, monthly_recurring_transactions_df], ignore_index=True
            )
            all_recurring_transactions_df = pd.concat([all_recurring_transactions_df, rt_df_copy], ignore_index=True)

    # Filter recurring transactions based on the selected date range
    start_date, end_date = [datetime.combine(d, datetime.min.time()) for d in date_range]
    all_recurring_transactions_df = all_recurring_transactions_df[
        (all_recurring_transactions_df["Transaction Date"] >= start_date)
        & (all_recurring_transactions_df["Transaction Date"] <= end_date)
    ]

    # Aggregate recurring transactions by year and month
    aggregated_recurring_transactions_df = (
        aggregated_recurring_transactions_df.groupby("YearMonth")
        .agg(
            {
                "Description": concat_descriptions,
                "Amount": "sum",
                "Recurring Income": "sum",
                "Recurring Expenses": "sum",
            }
        )
        .reset_index()
    )
    return aggregated_recurring_transactions_df, all_recurring_transactions_df, all_transactions_df


def plot_cash_flow(data):
    # Convert 'YearMonth' to string if not already
    data["YearMonth"] = data["YearMonth"].astype(str)
    data["Spending Cash"] = data["Amount"]  # Rename 'Amount' to 'Spending Cash'

    # Define custom colors
    income_color = "blue"
    expenses_color = "red"
    cash_color = "green"

    # Round values to 2 decimal places
    data["Recurring Income"] = data["Recurring Income"].round(2)
    data["Recurring Expenses"] = data["Recurring Expenses"].round(2)
    data["Spending Cash"] = data["Spending Cash"].round(2)

    # Create a figure with Plotly
    fig = go.Figure()

    def add_line_to_plot(fig, y, name, color, hover_text):
        """
        Add a line to a Plotly figure with specified x and y values, name, color, and hover text.
        """
        fig.add_trace(
            go.Scatter(
                x=data["YearMonth"],
                y=y,
                mode="lines+markers",
                name=name,
                line=dict(color=color),
                fill="tozeroy",
                hoverinfo="text",
                text=hover_text,
            )
        )

    add_line_to_plot(
        fig,
        data["Recurring Income"],
        "Recurring Income",
        income_color,
        data["Recurring Income"],
    )
    add_line_to_plot(
        fig,
        -data["Recurring Expenses"],
        "Recurring Expenses",
        expenses_color,
        -data["Recurring Expenses"],
    )
    add_line_to_plot(fig, data["Spending Cash"], "Spending Cash", cash_color, data["Spending Cash"])

    unique_months = sorted(data["YearMonth"].unique())  # Get unique months in order

    fig.update_layout(
        title="Monthly Recurring Cash Flow",
        xaxis_title="Month",
        yaxis_title="Amount",
        hovermode="x",
        template="plotly_white",
        xaxis=dict(
            tickmode="array",  # Use specific tick values
            tickvals=unique_months,  # Set ticks to unique months only
            tickangle=-45,  # Rotate labels diagonally
            showgrid=True,  # Show vertical gridlines
        ),
        yaxis=dict(
            showgrid=True,  # Show horizontal gridlines
        ),
    )

    # Show the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)
    data = data.drop(columns=["Description", "Amount"])
    # Format the amounts as currency
    data["Recurring Income"] = data["Recurring Income"].apply(lambda x: "$ {:,.2f}".format(x))
    data["Recurring Expenses"] = data["Recurring Expenses"].apply(lambda x: "$ {:,.2f}".format(x))
    data["Spending Cash"] = data["Spending Cash"].apply(lambda x: "$ {:,.2f}".format(x))
    # Show a collapsible dataframe
    with st.expander("Table: Monthly Recurring Cash Flow", expanded=False):
        st.dataframe(data, width=1000, height=492, hide_index=True)
    return data


def ai_financial_analyzer(all_rt_df, all_transactions_df, model):
    """
    Analyze the recurring transactions using the specified model.
    """
    # Convert the dataframe to a CSV string
    all_rt_csv = all_rt_df.to_csv(index=False)
    all_transactions_csv = all_transactions_df.to_csv(index=False)

    # Define the prompt to be used for both models
    prompt = f"""You are an expert financial advisor. 
Analyze the following table of Monthly Recurring Transactions and the table of All Transactions. 

Monthly Recurring Transactions:
{all_rt_csv}

All Transactions:
{all_transactions_csv}

Do not describe the format of the data you are given.
Identify the top spending categories, and if any: unusual spending behaviors and unnescessary purchases.
Provide a summary of spending trends and budgeting recommendations for the user. 
"""

    if model == "GPT-4o":
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert financial advisor."},
                {"role": "user", "content": prompt},
            ],
        )
        analysis = response["choices"][0]["message"]["content"].strip()
    else:
        if model == "Llama 3.1 8b":
            ollama_model = Ollama(model="llama3.1")
        elif model == "Llama 3.1 70b":
            ollama_model = Ollama(model="llama3.1:70b")
        response = ollama_model.invoke(prompt)
        analysis = response.strip()

    return analysis


def streamlit_app():
    st.title("AI Financial Advisor")
    col1, col2, col3 = st.columns([1.3, 2.1, 4])
    with col1:
        # create toggle for OpenAI vs local/Meta analysis
        model = st.radio("Select a Model", ("GPT-4o", "Llama 3.1 8b", "Llama 3.1 70b"))
    with col2:
        # Set the minimum and maximum selectable dates
        min_date = datetime(1990, 1, 1)
        max_date = datetime.today()

        # Set the default date range from january 1 of this year to today
        default_range = (datetime(2000, 1, 1), datetime.today())

        # Create a date range selector using st.date_input
        date_range = st.date_input(
            "Select a range of dates:",
            value=default_range,
            min_value=min_date,
            max_value=max_date,
            format="MM/DD/YYYY",
        )
    with col3:
        uploaded_statements = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
    # tolerance = st.slider("""Monthly recurring transactions can sometimes vary in amount. Select the percentage the amount can
    #                       differ between months:""", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
    # num_months = st.slider('''Minimum number of months a transaction must have occurred in a year to be considered as a recurring
    #                        transaction:''', min_value=2, max_value=12, value=2)

    if uploaded_statements:
        aggregated_recurring_transactions_df, all_recurring_transactions_df, all_transactions_df = combine_statement_dfs(
            uploaded_statements, date_range
        )  # , num_months, tolerance)
        # shorten name for the aggregated_recurring_transactions_df and all_recurring_transactions_df
        agg_rt_df = aggregated_recurring_transactions_df
        all_rt_df = all_recurring_transactions_df
        # sort the aggregated data by Amount
        if not agg_rt_df.empty:
            # Select a month to view its recurring expenses
            selected_month = st.selectbox(
                "Select a month to view its recurring expenses",
                sorted(agg_rt_df["YearMonth"].astype(str).unique(), reverse=True),
                index=1,  # default to the most recent completed month
            )
            st.write(f"Recurring Transactions for {selected_month}")
            # Filter original data by selected month and format 'Transaction Date'
            all_rt_df = all_rt_df[all_rt_df["YearMonth"].astype(str) == selected_month]
            all_rt_df["Transaction Date"] = all_rt_df["Transaction Date"].astype(str)

            # Prepare data for display purposes by dropping 'YearMonth', reordering columns, and formatting 'Amount'
            all_rt_df = all_rt_df.drop(columns=["YearMonth"])[["Transaction Date", "Description", "Category", "Amount"]]
            all_transactions_df = all_transactions_df[["Transaction Date", "Description", "Category", "Amount"]]
            all_rt_df_formatted = all_rt_df.sort_values(by="Amount", ascending=False)
            all_rt_df_formatted["Amount"] = all_rt_df_formatted["Amount"].apply(lambda x: f"$ {x:,.2f}")

            # Display the formatted data
            st.dataframe(all_rt_df_formatted, width=1000, height=250, hide_index=True)
            plot_cash_flow(agg_rt_df)

            # After the plot_cash_flow function call
            financial_analysis = ai_financial_analyzer(all_rt_df, all_transactions_df, model)
            st.subheader("AI Financial Analysis:")
            st.write(financial_analysis)

            st.write(f"Powered by the {model} model.")
        else:
            st.write("No recurring expenses found in the uploaded files.")
    else:
        st.write("Please upload CSV files to analyze.")


if __name__ == "__main__":
    # Initialize OpenAI API
    openai.api_key = openai_api_key
    # Load the saved SVM model and vectorizer
    svm_clf = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    streamlit_app()
