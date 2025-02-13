import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr

# Keep existing setup code
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCNXJSKZ4ba-gVsnR9oRI7yqWp8vdvgecw"
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(
    documents, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)


# Keep existing recommendation function
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(
            final_top_k
        )
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        return None, gr.Warning("Please enter a description to get recommendations.")

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    details = []

    for _, row in recommendations.iterrows():
        # Thumbnail and caption for gallery
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}"
        results.append((row["large_thumbnail"], caption))

        # Detailed information for markdown
        details.append(
            f"""
### {row['title']}
**Author(s):** {authors_str}  
**Category:** {row['simple_categories']}  
**Description:** {description}  
"""
        )

    details_markdown = "\n---\n".join(details)
    return results, details_markdown


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

css = """
.container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}
.header {
    text-align: center;
    margin-bottom: 30px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as dashboard:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown(
                """
                # 📚 Joe Goldberg
                Discover your next favorite book based on descriptions, categories, and emotional tones.
                """
            )

        with gr.Column():
            with gr.Row():
                with gr.Column(scale=2):
                    user_query = gr.Textbox(
                        label="What kind of book are you looking for?",
                        placeholder="e.g., A heartwarming story about family relationships and personal growth",
                        lines=3,
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        category_dropdown = gr.Dropdown(
                            choices=categories,
                            label="Category",
                            value="All",
                            container=True,
                        )
                        tone_dropdown = gr.Dropdown(
                            choices=tones,
                            label="Emotional Tone",
                            value="All",
                            container=True,
                        )

            submit_button = gr.Button("🔍 Find Books", size="lg", variant="primary")

            gr.Markdown("## 📖 Book Recommendations")
            with gr.Row():
                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="Book Covers",
                        columns=4,
                        rows=4,
                        height="auto",
                        allow_preview=True,
                    )
                with gr.Column(scale=1):
                    details = gr.Markdown(label="Book Details", show_label=True)

        gr.Markdown(
            """
            ---
            ### 💡 Tips for better results:
            - Be specific about themes, settings, or plot elements you're interested in
            - Try different emotional tones to discover varied perspectives
            - Experiment with categories to narrow down your search
            """
        )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[gallery, details],
    )

if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        show_error=True,
        share=False,
    )
