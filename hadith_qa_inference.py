#!/usr/bin/env python
"""
Quran QA Inference Script
- Loads test questions from a CSV file (columns: question_number, english, arabic, arabic_augment)
- For each question, retrieves top 20 answers from the qpc file
- Outputs a CSV with columns: question-number, Q0, passage-id, rank, score, tag
"""


import pandas as pd
import numpy as np
import argparse
import uuid
import os
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import joblib

def load_dense_retriever():
    """
    Loads the Arabic passage corpus, passage embeddings, FAISS index, and SentenceTransformer model for dense retrieval.
    Returns: (qbc_arab, passage_embeddings_arab, faiss_index, embed_model_arab)
    """
    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    hadith_path = os.path.join(data_dir, "hadith_in_arabic_with_english_translation.csv")
    passage_emb_path = os.path.join(data_dir, "hadith_passage_embeddings_arab.npy")
    faiss_index_path = os.path.join(data_dir, "hadith_faiss_index_arab.faiss")
    
    hadith_arabic = pd.read_csv(hadith_path)
    # Arabic
    hadith_arabic = hadith_arabic[["hadith_id", "Arabic_Hadith"]].copy()
    hadith_arabic['text'] = hadith_arabic['Arabic_Hadith']
    hadith_arabic['text'] = hadith_arabic['text'].fillna("")

    # Load precomputed passage embeddings and FAISS index    
    passage_embeddings_arab = np.load(passage_emb_path)
    faiss_index = faiss.read_index(faiss_index_path)
    embed_model_arab = SentenceTransformer("FDSRashid/QulBERT")
    return hadith_arabic, passage_embeddings_arab, faiss_index, embed_model_arab

def load_sparse_retriever():
    """
    Loads the English passage corpus, precomputed TF-IDF vectorizer, and TF-IDF matrix for sparse retrieval.
    Returns: (qbc, tfidf_vectorizer, tfidf_matrix)
    """
    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    hadith_path = os.path.join(data_dir, "hadith_in_arabic_with_english_translation.csv")
    tfidf_vectorizer_path = os.path.join(data_dir, "hadith_tfidf_vectorizer.pkl")
    tfidf_matrix_path = os.path.join(data_dir, "hadith_tfidf_matrix.npz")
    
    hadith_en = pd.read_csv(hadith_path)
    # English
    hadith_en = hadith_en[["hadith_id", "English_Hadith"]].copy()
    hadith_en['text'] = hadith_en['English_Hadith']
    hadith_en['text'] = hadith_en['text'].fillna("")
    # Load precomputed TF-IDF vectorizer and matrix
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    tfidf_matrix = joblib.load(tfidf_matrix_path)
    return hadith_en, tfidf_vectorizer, tfidf_matrix

# Arabic text normalization
_arabic_diacritics = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_punct_like = re.compile(r"[^\w\s\u0600-\u06FF]")
_spaces = re.compile(r"\s+")

def normalize_ar(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = _arabic_diacritics.sub("", text).replace("ـ", "")
    t = (t.replace("أ", "ا")
           .replace("إ", "ا")
           .replace("آ", "ا")
           .replace("ى", "ي")
           .replace("ئ", "ي")
           .replace("ؤ", "و")
           .replace("ة", "ه"))
    t = _punct_like.sub(" ", t)
    t = _spaces.sub(" ", t).strip()
    return t

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text):
    # Dummy lemmatizer for now (no nltk for speed)
    return text

def load_cross_encoder():
    model_dir = os.path.join(os.path.dirname(__file__), "Model/cross_encoder_finetuned")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    return tokenizer, model, device

def tfidf_search(query, tfidf_vectorizer, tfidf_matrix, qbc, top_k=50):
    query = normalize_text(query)
    query = lemmatize_text(query)
    query_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idxs = scores.argsort()[::-1][:top_k]
    return [(qbc.iloc[i]['hadith_id'], float(scores[i]), i) for i in top_idxs]

def dense_search_arab(query, embed_model_arab, faiss_index, qbc_arab, top_k=50):
    query = normalize_ar(query)
    q_emb = embed_model_arab.encode([query])
    faiss.normalize_L2(q_emb)
    scores, idxs = faiss_index.search(q_emb, top_k)
    results = [(qbc_arab.iloc[i]['hadith_id'], float(scores[0][j]), i) for j,i in enumerate(idxs[0])]
    return results


# Add main() wrapper for CLI
def main_cli():
    """Main function to run the inference script from command line.
    example usage:
    python quran_qa_inference.py --test_questions test_questions.csv --qpc qpc.csv --output output.tsv --tag_prefix BayaNet"""
    parser = argparse.ArgumentParser(description="Quran QA Inference Script")
    parser.add_argument("--test_questions", required=True, help="Path to test questions CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--tag_prefix", default="BayaNet", help="Prefix for the tag column")
    args = parser.parse_args()
    main(args.test_questions, args.output, args.tag_prefix)


def main(test_questions_path, output_path, tag_prefix="BayaNet"):
    # Load retrievers and cross-encoder
    qbc_arab, passage_embeddings_arab, faiss_index, embed_model_arab = load_dense_retriever()
    qbc, tfidf_vectorizer, tfidf_matrix = load_sparse_retriever()
    tokenizer, model, device = load_cross_encoder()

    # Load test questions
    test_df = pd.read_csv(test_questions_path)

    results = []
    unique_tag = f"{tag_prefix}_{uuid.uuid4().hex[:8]}"
    for _, row in test_df.iterrows():
        qnum = row["question_number"]
        # You can choose which question text to use (english, arabic, arabic_augment)
        question_text = row["arabic"] + "\n" + row["arabic_augment"]  if pd.notnull(row.get("arabic_augment", None)) and row["arabic_augment"] else row["arabic"]
        # For sparse, use English
        question_text_en = row["english"]

        # Sparse retrieval (top 50)
        sparse_candidates = tfidf_search(question_text_en, tfidf_vectorizer, tfidf_matrix, qbc, top_k=50)
        # Dense retrieval (top 50)
        dense_candidates = dense_search_arab(question_text, embed_model_arab, faiss_index, qbc_arab, top_k=50)

        # Fusion: union by passage_id, keep max score from either, and keep index for text lookup
        fusion_dict = {}
        for pid, score, idx in sparse_candidates:
            fusion_dict[pid] = (score, 'sparse', idx)
        for pid, score, idx in dense_candidates:
            if pid in fusion_dict:
                # Keep the higher score, prefer dense for tie
                if score > fusion_dict[pid][0]:
                    fusion_dict[pid] = (score, 'dense', idx)
            else:
                fusion_dict[pid] = (score, 'dense', idx)
        # Prepare candidates for reranking: get text from qbc
        fusion_candidates = []
        for pid, (score, source, idx) in fusion_dict.items():
            if source == 'dense':
                fusion_candidates.append((pid, score, idx, 'dense'))
            else:
                fusion_candidates.append((pid, score, idx, 'sparse'))
        # Sort by score, take top 50 for reranking
        fusion_candidates = sorted(fusion_candidates, key=lambda x: x[1], reverse=True)[:50]

        # Prepare for cross-encoder: get passage text 
        ce_candidates = []
        for pid, score, idx, source in fusion_candidates:
            ce_candidates.append((pid, score, idx, qbc.iloc[idx]['text']))

        # Cross-encoder rerank (top 20)
        candidate_passages = [x[3] for x in ce_candidates]
        candidate_ids = [x[0] for x in ce_candidates]
        features = tokenizer([question_text_en]*len(candidate_passages), candidate_passages, padding=True, truncation=True, return_tensors="pt")
        features = {k: v.to(device) for k, v in features.items()}
        with torch.no_grad():
            outputs = model(**features)
            if outputs.logits.shape[1] == 2:
                scores = outputs.logits[:, 1].cpu().numpy()
            else:
                scores = outputs.logits[:, 0].cpu().numpy()
        reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)[:5]
        for rank, (pid, score) in enumerate(reranked, 1):
            results.append({
                "question-number": qnum,
                "Q0": "Q0",
                "passage-id": pid,
                "rank": rank,
                "score": float(score),
                "tag": unique_tag
            })
    out_df = pd.DataFrame(results)
    # save dataframe as tsv without header
    if not output_path.endswith('.tsv'):
        output_path += '.tsv'
    out_df.to_csv(output_path, index=False, sep='\t', header=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main_cli()
