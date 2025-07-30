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
    qbc_arab_path = os.path.join(data_dir, "Quran_in_arabic_with_aljalalyn_and_almuyassar_tafseer.csv")
    passage_emb_path = os.path.join(data_dir, "passage_embeddings_arab.npy")
    faiss_index_path = os.path.join(data_dir, "faiss_index_arab.faiss")
    
    qbc_arab = pd.read_csv(qbc_arab_path)
    passage_embeddings_arab = np.load(passage_emb_path)
    faiss_index = faiss.read_index(faiss_index_path)
    embed_model_arab = SentenceTransformer("FDSRashid/QulBERT")
    return qbc_arab, passage_embeddings_arab, faiss_index, embed_model_arab

def load_sparse_retriever():
    """
    Loads the English passage corpus, precomputed TF-IDF vectorizer, and TF-IDF matrix for sparse retrieval.
    Returns: (qbc, tfidf_vectorizer, tfidf_matrix)
    """
    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    qbc_path = os.path.join(data_dir, "qpc_with_quran_translation_and_aljalalyn.csv")
    tfidf_vectorizer_path = os.path.join(data_dir, "tfidf_vectorizer.pkl")
    tfidf_matrix_path = os.path.join(data_dir, "tfidf_matrix.npz")
    
    qbc = pd.read_csv(qbc_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    tfidf_matrix = np.load(tfidf_matrix_path)["arr_0"]
    return qbc, tfidf_vectorizer, tfidf_matrix

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
    return [(qbc.iloc[i]['passage_id'], float(scores[i]), i) for i in top_idxs]

def dense_search_arab(query, embed_model_arab, faiss_index, qbc_arab, top_k=50):
    q_emb = embed_model_arab.encode([query])
    faiss.normalize_L2(q_emb)
    scores, idxs = faiss_index.search(q_emb, top_k)
    results = [(qbc_arab.iloc[i]['passage_id'], float(scores[0][j]), i) for j,i in enumerate(idxs[0])]
    return results

def cross_encode_rerank(query, candidates, qbc_arab, tokenizer, model, device, top_k=20):
    # candidates: list of (passage_id, dense_score, idx)
    candidate_passages = [qbc_arab.iloc[idx]['text'] for _,_,idx in candidates]
    candidate_ids = [pid for pid,_,_ in candidates]
    features = tokenizer([query]*len(candidate_passages), candidate_passages, padding=True, truncation=True, return_tensors="pt")
    features = {k: v.to(device) for k, v in features.items()}
    with torch.no_grad():
        outputs = model(**features)
        # Use the positive class logit (label=1)
        if outputs.logits.shape[1] == 2:
            scores = outputs.logits[:, 1].cpu().numpy()
        else:
            scores = outputs.logits[:, 0].cpu().numpy()
    reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# Add main() wrapper for CLI
def main_cli():
    parser = argparse.ArgumentParser(description="Quran QA Inference Script")
    parser.add_argument("--test_questions", required=True, help="Path to test questions CSV")
    parser.add_argument("--qpc", required=True, help="Path to qpc CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--tag_prefix", default="BayaNet", help="Prefix for the tag column")
    args = parser.parse_args()
    main(args.test_questions, args.qpc, args.output, args.tag_prefix)


def main(test_questions_path, qpc_path, output_path, tag_prefix="BayaNet"):
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
        question_text = row["arabic_augment"] if pd.notnull(row.get("arabic_augment", None)) and row["arabic_augment"] else (row["arabic"] if pd.notnull(row.get("arabic", None)) and row["arabic"] else row["english"])
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
        # Prepare candidates for reranking: get text from qbc_arab if dense, else from qbc
        fusion_candidates = []
        for pid, (score, source, idx) in fusion_dict.items():
            if source == 'dense':
                fusion_candidates.append((pid, score, idx, 'dense'))
            else:
                fusion_candidates.append((pid, score, idx, 'sparse'))
        # Sort by score, take top 50 for reranking
        fusion_candidates = sorted(fusion_candidates, key=lambda x: x[1], reverse=True)[:50]

        # Prepare for cross-encoder: get passage text from correct source
        ce_candidates = []
        for pid, score, idx, source in fusion_candidates:
            if source == 'dense':
                ce_candidates.append((pid, score, idx, qbc_arab.iloc[idx]['text']))
            else:
                ce_candidates.append((pid, score, idx, qbc.iloc[idx]['text']))

        # Cross-encoder rerank (top 20)
        candidate_passages = [x[3] for x in ce_candidates]
        candidate_ids = [x[0] for x in ce_candidates]
        features = tokenizer([question_text]*len(candidate_passages), candidate_passages, padding=True, truncation=True, return_tensors="pt")
        features = {k: v.to(device) for k, v in features.items()}
        with torch.no_grad():
            outputs = model(**features)
            if outputs.logits.shape[1] == 2:
                scores = outputs.logits[:, 1].cpu().numpy()
            else:
                scores = outputs.logits[:, 0].cpu().numpy()
        reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)[:20]
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
    out_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main_cli()
