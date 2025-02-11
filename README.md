# Joe-Goldberg: AI-Powered Book Recommendation System 📚

Joe-Goldberg is an intelligent book recommendation system that leverages Large Language Models (LLMs) to provide personalized book suggestions based on user preferences and reading history. The system uses semantic search with Google's Generative AI embeddings to find books similar to user queries, while also considering emotional tones and categories.

## 🚀 Features

- Personalized book recommendations using advanced LLM technology
- Sentiment analysis of book reviews and ratings using DistilRoBERTa
- Zero-shot text classification for genre categorization using BART
- Vector search for finding similar books
- Data exploration and visualization tools
- Interactive dashboard for user interaction
- Emotion-based filtering (Happy, Surprising, Angry, Suspenseful, Sad)
- Category-based filtering across multiple book genres
- Customizable search with adjustable recommendation parameters

## 🛠️ Technical Stack

### Core Technologies
- Python 3.8+
- LangChain
- Google Generative AI
- Chroma DB

### Data Processing & Analysis
- Pandas
- NumPy
- Seaborn
- Matplotlib

### Machine Learning & NLP
- Hugging Face Transformers
  - facebook/bart-large-mnli for zero-shot classification
  - j-hartmann/emotion-english-distilroberta-base for emotion analysis
- Google Generative AI Embeddings
- LangChain Text Splitters
- Chroma Vector Store

### User Interface
- Gradio
- Gradio Glass Theme

### Development Tools
- Python-dotenv
- Jupyter Notebooks

## 📁 Project Structure

```
joe-goldberg/
├── .gitignore
├── README.md
├── books_with_emotions.csv    # Dataset with emotional analysis
├── tagged_description.txt     # Processed book descriptions
├── cover-not-found.jpg       # Default image for missing covers
├── dash.py                   # Main application with Gradio interface
├── data-exploration.ipynb    # Data analysis and visualization
├── sentiment_analysis.ipynb  # Emotional content analysis
├── text_classification.ipynb # Genre classification
└── vector_search.ipynb      # Semantic search implementation
```

## 💡 How It Works

1. **Data Processing**
   - Books data is loaded from `books_with_emotions.csv`
   - Book descriptions are processed and embedded using Google's Generative AI
   - Embeddings are stored in a Chroma vector database for efficient similarity search

2. **Text Classification**
   - Uses BART-large-mnli model for zero-shot classification of books into Fiction/Nonfiction categories
   - Implements flexible classification without need for specific training data
   - Model handles genre categorization through natural language understanding

3. **Sentiment Analysis**
   - Utilizes DistilRoBERTa-based emotion classification model
   - Provides comprehensive emotion analysis across multiple categories
   - Detects emotions: joy, surprise, anger, fear, sadness
   - Used for emotional tone filtering and book recommendation ranking

4. **Recommendation Engine**
   - Uses semantic search to find books similar to user queries
   - Supports filtering by category and emotional tone
   - Returns customizable number of recommendations (default: 16 books)
   - Sorts results based on emotional scores when tone is selected

5. **User Interface**
   - Clean, modern interface built with Gradio
   - Three main input components:
     - Text query box for book description
     - Category dropdown for genre filtering
     - Emotional tone dropdown for mood-based filtering
   - Results displayed in a responsive gallery layout
   - Book thumbnails with author and truncated description

6. **Visualization Features**
   - Data exploration through Jupyter notebooks
   - Statistical analysis of book categories
   - Emotional content visualization
   - Genre distribution analysis
