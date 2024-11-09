import os
import json
import argparse
import numpy as np

from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, key, chunk_size=400, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " "]
    )

    return text_splitter.create_documents([text], metadatas=[{"source": key}])


model = SentenceTransformer('BAAI/bge-m3')


def bert_retrieve(qs, source, corpus_dict, n):
    chunks = []

    for file in source:
        chunk_file = split_text(corpus_dict[int(file)], file)
        chunks.extend(chunk_file)

    query_embedding = model.encode(qs, convert_to_tensor=True)
    chunk_texts = [chunk.page_content for chunk in chunks]  # 獲取每個 chunk 的文字內容
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = torch.topk(similarities, k=n)
    res = [chunks[idx].metadata["source"] for idx in top_results.indices]
    return res

def BM25_retrieve(qs, source, corpus_dict, n):
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    if n == 1 and ans:
        a = ans[0]
        res = [key for key, value in corpus_dict.items() if value == a]
        return res[0]  # 回傳檔案名
    else:
        res = [key for key, value in corpus_dict.items() if value in ans]
        return res[:n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()

    answer_dict = {"answers": []}

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)

    with open("/Users/chiangsssssss/PycharmProjects/Fintech_project/insurance/insurance_formal.json", 'rb') as i_s:
        key_to_source_dict3 = json.load(i_s)  # 讀取參考資料文件
        key_to_source_dict3 = {int(key): value for key, value in key_to_source_dict3.items()}

    for q_dict in qs_ref['questions']:

        if q_dict['category'] == 'insurance':
            corpus_dict_insurance = {key: str(value) for key, value in key_to_source_dict3.items() if
                                     key in q_dict['source']}
            retrieved_1 = bert_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance, 1)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved_1[0]})
        else:
            continue

    # 將答案字典保存為json文件
    with open("answer_bge_formal", 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符..
