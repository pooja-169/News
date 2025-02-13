import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Streamlit page settings
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Load the CSV file directly
file_path = "transformed_sentiment_data.csv"
df = pd.read_csv(file_path)

# Apply custom CSS for a polished look
st.markdown("""
    <style>
        .main {
            background-color: #F5F5F5;
            padding: 20px;
        }
        .card {
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 20px;
            background-color: white;
            box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
        }
        .stDataFrame {
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and introductory text
st.title("Sentiment Analysis Dashboard")
st.write("Explore and understand sentiment analysis through interactive visualizations and data insights.")

# Display the dataframe in a bordered box
#st.subheader("Data Overview")
#st.dataframe(df.style.set_properties(**{'border': '1px solid black'}), height=300)

# Layout the dashboard into two columns
col1, col2 = st.columns([1, 1])

# Sentiment Distribution Plot
with col1:
    st.markdown("### Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=df['sentiment_labels'].value_counts().index, 
                y=df['sentiment_labels'].value_counts().values, 
                palette="viridis", ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment Labels")
    ax.set_ylabel("Count")
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    st.pyplot(fig)

# Intensity Score Distribution
with col2:
    st.markdown("### Intensity Scores Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['intensity_scores'], kde=True, color="royalblue", edgecolor='black', ax=ax)
    ax.set_title("Intensity Scores Distribution")
    ax.set_xlabel("Intensity Scores")
    ax.set_ylabel("Frequency")
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    st.pyplot(fig)

# Sentiment Scores by Cluster Boxplot
st.markdown("### Sentiment Scores by Cluster")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='clusters', y='sentiment_scores', data=df, palette="Set3", ax=ax)
ax.set_title("Sentiment Scores by Cluster")
ax.set_xlabel("Clusters")
ax.set_ylabel("Sentiment Scores")
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
st.pyplot(fig)

# Detailed Data View based on sentiment selection
#st.markdown("### Detailed Sentiment View")
#selected_sentiment = st.selectbox("Select Sentiment Label", options=["positive", "neutral", "negative"])
#detailed_df = df[df['sentiment_labels'] == selected_sentiment]
#st.write(f"Records for selected sentiment: **{selected_sentiment}**")
#st.dataframe(detailed_df.style.set_properties(**{'border': '1px solid black'}))

# Footer with credits
st.markdown("---")
st.write("© 2024 Sentiment Analysis Dashboard | Developed with ❤️ using Streamlit")
