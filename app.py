import re
from datetime import datetime

import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz, process
import plotly.graph_objects as go
import openai
from langchain_community.llms import Ollama

from openai_token import openai_api_key

# Initialize OpenAI API
openai.api_key = openai_api_key

def preprocess_description(description):
    # Use regular expression to remove numbers (or any unique identifiers) and strip leading/trailing whitespace
    return re.sub(r'\d+', '', description).strip()

def find_recurring_transactions(data, num_months=2, similarity_threshold=78):
    # Convert 'Transaction Date' to datetime
    data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
    # Extract year and month from 'Transaction Date' to create a YearMonth column
    data['YearMonth'] = data['Transaction Date'].dt.to_period('M').astype(str)
    # Preprocess descriptions to remove varying parts
    data['ProcessedDescription'] = data['Description'].apply(preprocess_description)
    # Get unique processed descriptions
    descriptions = data['ProcessedDescription'].unique()
    # Create a mapping of similar descriptions
    similar_groups = {}
    for desc in descriptions:
        # Find the most similar description group based on the threshold
        matched_desc = process.extractOne(desc, similar_groups.keys(), scorer=fuzz.token_sort_ratio)
        if matched_desc and matched_desc[1] >= similarity_threshold:
            # If similar description found, add the current description to that group
            similar_groups[matched_desc[0]].append(desc)
        else:
            # If no similar description found, start a new group
            similar_groups[desc] = [desc]
    
    def map_to_group(desc):
        '''Map descriptions in the dataframe to their groups'''
        processed_desc = preprocess_description(desc)
        for key, values in similar_groups.items():
            if processed_desc in values:
                return key
        return desc  # Fallback to the original description if no match found
    
    data['GroupedDescription'] = data['Description'].apply(map_to_group)
    
    # Group by description and year/month and count occurrences
    grouped = data.groupby(['GroupedDescription', 'YearMonth']).size().reset_index(name='Count')
    # Group by description to count the number of months each appears in
    recurring = grouped.groupby(['GroupedDescription']).size().reset_index(name='MonthCount')
    # Filter to keep only descriptions that appear in the minimum number of months
    recurring_costs = recurring[recurring['MonthCount'] >= num_months]
    recurring_costs = recurring_costs['GroupedDescription'].tolist()
    
    return recurring_costs

def identify_subscriptions(df, tolerance=0.20, similarity_threshold=78):
    # Convert 'Transaction Date' to datetime
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['ProcessedDescription'] = df['Description'].apply(preprocess_description)
    
    # Get unique processed descriptions
    descriptions = df['ProcessedDescription'].unique()
    
    # Create a mapping of similar descriptions
    similar_groups = {}
    
    for desc in descriptions:
        # Find the most similar description group based on the threshold
        matched_desc = process.extractOne(desc, similar_groups.keys(), scorer=fuzz.token_sort_ratio)
        if matched_desc and matched_desc[1] >= similarity_threshold:
            # If similar description found, add the current description to that group
            similar_groups[matched_desc[0]].append(desc)
        else:
            # If no similar description found, start a new group
            similar_groups[desc] = [desc]
    
    def map_to_group(desc):
        '''Map descriptions in the dataframe to their groups'''
        processed_desc = preprocess_description(desc)
        for key, values in similar_groups.items():
            if processed_desc in values:
                return key
        return desc  # Fallback to the original description if no match found
    
    df['GroupedDescription'] = df['Description'].apply(map_to_group)
    
    subscriptions = []
    grouped = df.groupby('GroupedDescription')

    for name, group in grouped:
        group = group.sort_values(by='Transaction Date')  # Ensure transactions are sorted by date
        amounts = group['Amount'].values
        dates = group['Transaction Date'].values

        for i in range(len(amounts) - 1):
            # Check if the amounts are within the tolerance level
            if abs(amounts[i] - amounts[i + 1]) <= tolerance * abs(amounts[i]):
                # Check if the difference in dates suggests a monthly or bi-weekly pattern
                diff = (dates[i + 1] - dates[i]).astype('timedelta64[D]').item().days
                if (28 <= diff <= 32) or (13 <= diff <= 17):
                    subscriptions.append(group.iloc[i])
                    subscriptions.append(group.iloc[i + 1])
    subscriptions_df = pd.DataFrame(subscriptions).drop_duplicates()
    return subscriptions_df

def process_file(file_path, num_months, tolerance, date_range):
    """
    Process a CSV files representing Checking and Credit Card statements to identify recurring transactions and subscriptions.
    
    The files used for this analysis came from Capital One statements. Adapt the code as needed for other banks statement formats.

    This assumes that the CSV file has the following columns from Checking statements:
    Transaction Date,Posted Date,Card No.,Description,Category,Debit,Credit

    And the following columns from Credit Card statements:
    Account Number,Transaction Description,Transaction Date,Transaction Type,Transaction Amount,Balance
    """
    df = pd.read_csv(file_path)
    # filter data based on the selected date range
    start_date, end_date = [datetime.combine(d, datetime.min.time()) for d in date_range]
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df = df[(df['Transaction Date'] >= start_date) & (df['Transaction Date'] <= end_date)]

    if 'Debit' not in df.columns:
        # Process files without a 'Debit' column by calculating the amount based on the transaction type
        df['Amount'] = df.apply(lambda x: -x['Transaction Amount'] if x['Transaction Type'] == 'Debit' else x['Transaction Amount'], axis=1)
        df = df.drop(columns=['Account Number', 'Balance', 'Transaction Amount', 'Transaction Type'])
        df = df.rename(columns={'Transaction Description': 'Description'})
    else:
        # Process files with a 'Debit' column by inverting the amounts to reflect expenses as negative
        df = df.dropna(subset=['Debit'])
        df = df.rename(columns={'Debit': 'Amount'})
        df = df.drop(columns=['Credit', 'Card No.', 'Posted Date', 'Category'])
        df['Amount'] = df['Amount'] * -1

    # Find recurring transactions
    recurring_expenses = find_recurring_transactions(df, num_months)
    df['GroupedDescription'] = df['Description'].apply(lambda x: preprocess_description(x))
    df = df[df['GroupedDescription'].isin(recurring_expenses)]
    df = df[['Amount', 'Transaction Date', 'YearMonth', 'Description', 'GroupedDescription']]
    df = df[['Amount', 'Transaction Date', 'YearMonth', 'Description']]  
    df = identify_subscriptions(df, tolerance)
    
    # Make a copy of the df so we can display the subscriptions for a selected month
    copy_df = df.copy()
    if len(recurring_expenses) == 0:
        return None, copy_df
    
    # Calculate recurring income and expenses
    df['Recurring Income'] = df[df['Amount'] > 0].groupby('YearMonth')['Amount'].transform('sum')
    df['Recurring Expenses'] = df[df['Amount'] < 0].groupby('YearMonth')['Amount'].transform('sum')
    
    # Aggregate details by month
    monthly_recurring_details = df.groupby('YearMonth').agg({
        'Description': lambda x: list(x.unique()),
        'Amount': 'sum',
        'Recurring Income': 'first',
        'Recurring Expenses': 'first'
    }).reset_index()
    
    return monthly_recurring_details, copy_df

def concat_descriptions(descriptions):
    # Concatenate a list of descriptions into a single string
    return ', '.join(sum(descriptions, []))

def aggregate_files(uploaded_files, num_months, tolerance, date_range):
    """
    Aggregate the data from multiple uploaded files into a single DataFrame. 
    Inputs:
    - uploaded_files: List of uploaded CSV files
    - num_months: Minimum number of months a transaction must have occurred in a year to be considered as a recurring transaction
    - tolerance: The percentage the amount can differ between months for a transaction to be considered recurring

    Returns:
    - all_data: DataFrame with aggregated financial data
    - original_data: DataFrame with the original data from the uploaded files for the purpose of later displaying the transactions for a selected month
    """
    original_data = pd.DataFrame()
    all_data = pd.DataFrame()
    for uploaded_file in uploaded_files:
        file_data, copy_df = process_file(uploaded_file, num_months, tolerance, date_range)
        if file_data is not None:
            all_data = pd.concat([all_data, file_data], ignore_index=True)
            original_data = pd.concat([original_data, copy_df], ignore_index=True)
    # Filter data based on the selected date range
    start_date, end_date = [datetime.combine(d, datetime.min.time()) for d in date_range]
    original_data = original_data[(original_data['Transaction Date'] >= start_date) & (original_data['Transaction Date'] <= end_date)]
    # st.dataframe(original_data)

    # Aggregate all data by year and month
    all_data = all_data.groupby('YearMonth').agg({
        'Description': concat_descriptions,
        'Amount': 'sum',
        'Recurring Income': 'sum',
        'Recurring Expenses': 'sum'
    }).reset_index()
    return all_data, original_data

def plot_financial_data(data):
    # Convert 'YearMonth' to string if not already
    data['YearMonth'] = data['YearMonth'].astype(str)
    data['Spending Cash'] = data['Amount']  # Rename 'Amount' to 'Spending Cash'
    
    # Define custom colors
    income_color = 'blue'
    expenses_color = 'red'
    cash_color = 'green'
    
    # Round values to 2 decimal places
    data['Recurring Income'] = data['Recurring Income'].round(2)
    data['Recurring Expenses'] = data['Recurring Expenses'].round(2)
    data['Spending Cash'] = data['Spending Cash'].round(2)

    # Create a figure with Plotly
    fig = go.Figure()
    
    def add_line_to_plot(fig, y, name, color, hover_text):
        """
        Add a line to a Plotly figure with specified x and y values, name, color, and hover text.
        """
        fig.add_trace(go.Scatter(
            x=data['YearMonth'],
            y=y,
            mode='lines+markers',
            name=name,
            line=dict(color=color),
            fill='tozeroy',
            hoverinfo='text',
            text=hover_text
        ))
    add_line_to_plot(fig, data['Recurring Income'], 'Recurring Income', income_color, data['Recurring Income'])
    add_line_to_plot(fig, -data['Recurring Expenses'], 'Recurring Expenses', expenses_color, -data['Recurring Expenses'])
    add_line_to_plot(fig, data['Spending Cash'], 'Spending Cash', cash_color, data['Spending Cash'])

    unique_months = sorted(data['YearMonth'].unique())  # Get unique months in order

    fig.update_layout(
        title='Monthly Financial Overview',
        xaxis_title='Month',
        yaxis_title='Amount',
        hovermode='x',
        template='plotly_white',
        xaxis=dict(
            tickmode='array',  # Use specific tick values
            tickvals=unique_months,  # Set ticks to unique months only
            tickangle=-45,  # Rotate labels diagonally
            showgrid=True,  # Show vertical gridlines
        ),
        yaxis=dict(
            showgrid=True,  # Show horizontal gridlines
        )
    )
    
    # Show the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)
    data = data.drop(columns=['Description', 'Amount'])
    # Format the amounts as currency
    data['Recurring Income'] = data['Recurring Income'].apply(lambda x: '$ {:,.2f}'.format(x))
    data['Recurring Expenses'] = data['Recurring Expenses'].apply(lambda x: '$ {:,.2f}'.format(x))
    data['Spending Cash'] = data['Spending Cash'].apply(lambda x: '$ {:,.2f}'.format(x))
    st.dataframe(data, width=1000, height=495, hide_index=True)

    return data

def analyze_with_llm(df, model):
    """
    Analyze the financial data using the LLM model.
    Inputs:
    - df: DataFrame containing the financial data
    - model: The model to use for analysis (either "OpenAI" or "Llama3.1")
    Returns:
    - analysis: The financial analysis provided by the model
    """
    # Convert the dataframe to a CSV string
    csv_data = df.to_csv(index=False)

    if model == "OpenAI":
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            messages=[
                    {"role": "system", "content": "You are an expert financial advisor."},
                    {"role": "user", "content": f"These are the monthly recurring transactions for a single month: \
                     {csv_data} Please analyze it for any anomalies, providing any suggestions for budgeting and a summary of transactions"}
                ],
            )
        analysis = response['choices'][0]['message']['content'].strip()
    else:
        ollama_model = Ollama(model="llama3.1")
        # Define the prompt for the Ollama model
        prompt = f"You are an expert financial advisor. \
            Analyze the following monthly recurring transactions for anomalies and provide suggestions for budgeting and a summary of transactions:\n\n{csv_data}"
        # Use the Ollama model to analyze the data
        response = ollama_model.invoke(prompt)
        # Extract the analysis from the response
        analysis = response.strip()
    return analysis

def streamlit_app():
    # Streamlit app
    st.title("LLM Budget: Recurring Transactions")
    col1, col2, col3 = st.columns([1.3, 2.1, 4])
    with col1:
        # create toggle for OpenAI vs local analysis
        model = st.radio("Select a Model", ("GPT-4o", "Llama3.1"))
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
            max_value=max_date
        )
    with col3:
        uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
    # Slider to select tolerance for amount variation between months
    # tolerance = st.slider("Monthly recurring transactions can sometimes vary in amount. Select the percentage the amount can differ between months:", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
    tolerance = 0.20


    # Slider to select the minimum number of months for recurring transactions
    # num_months = st.slider('Minimum number of months a transaction must have occurred in a year to be considered as a recurring transaction:', min_value=2, max_value=12, value=2)
    num_months = 2

    if uploaded_files:
        aggregated_data, original_data = aggregate_files(uploaded_files, num_months, tolerance, date_range)
        # sort the aggregated data by Amount    
        if not aggregated_data.empty:
            # Select a month to view its recurring expenses
            selected_month = st.selectbox("Select a month to view its recurring expenses", sorted(aggregated_data['YearMonth'].astype(str).unique(), reverse=True), index=1)
            st.write(f"Recurring Transactions for {selected_month}")
            # Filter original data by selected month
            original_data = original_data[original_data['YearMonth'].astype(str) == selected_month]
            original_data['Transaction Date'] = original_data['Transaction Date'].astype(str)
            display_data = original_data.drop(columns=['YearMonth'])  # Drop "YearMonth' column for display
            display_data = display_data[['Transaction Date', 'Description', 'Amount']]  # Reorder columns
            display_data_formatted = display_data.sort_values(by='Amount', ascending=False) # Sort by amount
            display_data_formatted['Amount'] = display_data_formatted['Amount'].apply(lambda x: '$ {:,.2f}'.format(x))  # Round amounts for display
            st.dataframe(display_data_formatted, width=1000, height=250, hide_index=True)  # Display the data without index
            data = plot_financial_data(aggregated_data)


            # After the plot_financial_data function call
            financial_analysis = analyze_with_llm(display_data, model)
            st.subheader("LLM Financial Analysis:")
            st.write(financial_analysis)
            if model == "OpenAI":
                st.write("Powered by OpenAI's GPT-4o model.")
            else:
                st.write("Powered by the Llama3.1 model.")
        else:
            st.write("No recurring expenses found in the uploaded files.")
    else:
        st.write("Please upload CSV files to analyze.")

if __name__ == '__main__':
    streamlit_app()
