import re

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

def process_file(file_path, num_months, tolerance):
    df = pd.read_csv(file_path)
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

def aggregate_files(uploaded_files, num_months, tolerance):
    original_data = pd.DataFrame()
    all_data = pd.DataFrame()
    for uploaded_file in uploaded_files:
        file_data, copy_df = process_file(uploaded_file, num_months, tolerance)
        if file_data is not None:
            all_data = pd.concat([all_data, file_data], ignore_index=True)
            original_data = pd.concat([original_data, copy_df], ignore_index=True)
    
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
    
    # Add the Recurring Income line with hover text
    fig.add_trace(go.Scatter(
        x=data['YearMonth'], 
        y=data['Recurring Income'], 
        mode='lines+markers',
        name='Recurring Income',
        line=dict(color=income_color),
        fill='tozeroy',
        hoverinfo='text',
        text=data['Recurring Income']
    ))

    # Add the Recurring Expenses line with hover text
    fig.add_trace(go.Scatter(
        x=data['YearMonth'], 
        y=-data['Recurring Expenses'], 
        mode='lines+markers',
        name='Recurring Expenses',
        line=dict(color=expenses_color),
        fill='tozeroy',
        hoverinfo='text',
        text=-data['Recurring Expenses']
    ))

    # Add the Spending Cash line with hover text
    fig.add_trace(go.Scatter(
        x=data['YearMonth'], 
        y=data['Spending Cash'], 
        mode='lines+markers',
        name='Spending Cash',
        line=dict(color=cash_color),
        fill='tozeroy',
        hoverinfo='text',
        text=data['Spending Cash']
    ))

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
    # Convert the dataframe to a CSV string
    csv_data = df.to_csv(index=False)

    if model == "OpenAI":
        # Use OpenAI API to analyze the data
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                    {"role": "system", "content": "You are an expert financial advisor."},
                    {"role": "user", "content": f"These are the monthly recurring transactions for a single month: {csv_data} Please analyze it for any anomalies, providing any suggestions for budgeting and a summary of transactions"}
                ],
            )
        analysis = response['choices'][0]['message']['content'].strip()
    else:
        ollama_model = Ollama(model="llama3.1")
        # Define the prompt for the Ollama model
        prompt = f"You are an expert financial advisor. Analyze the following monthly recurring transactions for anomalies and provide suggestions for budgeting and a summary of transactions:\n\n{csv_data}"
        # Use the Ollama model to analyze the data
        response = ollama_model.invoke(prompt)
        # Extract the analysis from the response
        analysis = response.strip()
    return analysis

# Streamlit app
st.title("Budget: Recurring Transactions")
col1, col2 = st.columns([1, 3])
with col1:
    # create toggle for OpenAI vs local analysis
    model = st.radio("Select a Model", ("GPT-4o", "Llama3.1"))
with col2:
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

# Slider to select tolerance for amount variation between months
tolerance = st.slider("Monthly recurring transactions can sometimes vary in amount. Select the percentage the amount can differ between months:", min_value=0.01, max_value=1.0, value=0.20, step=0.01)

# Slider to select the minimum number of months for recurring transactions
# num_months = st.slider('Minimum number of months a transaction must have occurred in a year to be considered as a recurring transaction:', min_value=2, max_value=12, value=2)
num_months = 2

if uploaded_files:
    aggregated_data, original_data = aggregate_files(uploaded_files, num_months, tolerance)
    if not aggregated_data.empty:
        # Select a month to view its recurring expenses
        selected_month = st.selectbox("Select a month to view its recurring expenses", sorted(aggregated_data['YearMonth'].astype(str).unique(), reverse=True))
        st.write(f"Recurring Transactions for {selected_month}")
        # Filter original data by selected month
        original_data = original_data[original_data['YearMonth'].astype(str) == selected_month]
        original_data['Amount'] = original_data['Amount'].apply(lambda x: '$ {:,.2f}'.format(x))  # Round amounts for display
        original_data['Transaction Date'] = original_data['Transaction Date'].astype(str)
        display_data = original_data.drop(columns=['YearMonth'])  # Drop "YearMonth' column for display
        display_data = display_data[['Transaction Date', 'Description', 'Amount']]  # Reorder columns
        st.dataframe(display_data, width=1000, height=250, hide_index=True)  # Display the data without index
        data = plot_financial_data(aggregated_data)


        # After the plot_financial_data function call
        financial_analysis = analyze_with_llm(display_data, model)
        st.subheader("LLM Financial Analysis:")
        st.write(financial_analysis)
    else:
        st.write("No recurring expenses found in the uploaded files.")
else:
    st.write("Please upload CSV files to analyze.")