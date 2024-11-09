import os
import json
import argparse
import jieba
import time
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.retrievers import KNNRetriever

chunk_size = 400
chunk_overlap = 100
model_name = 'voyage-multilingual-2'
api_key = "pa-DwwWQolVDiERwvz352Ydq7bUm5g42OIn5GfE8JnlLko"
embedding = VoyageAIEmbeddings(
        voyage_api_key=api_key, model=model_name
    )


def chunk(long_text, key):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=len(long_text), chunk_overlap=0, separators=[
        "\n\n",
        "\n",
        " ",
        "。",
        "，",
        ".",
        ","])
    chunks = text_splitter.create_documents([long_text],  metadatas=[{"source": key}])
    return chunks

def voyage_retriever(qs, source, corpus_dict):
    chunks = []
    for file in source:
        file_chunks = chunk(corpus_dict[int(file)], file)
        chunks.extend(file_chunks)
    # query_embd = embedding.embed_query(qs)
    retriever = KNNRetriever.from_documents(chunks, embedding)
    results = retriever.invoke(qs)
    top1_retrieved = results[0].metadata["source"]
    # top2_retrieved = results[1].metadata["source"]
    # top3_retrieved = results[2].metadata["source"]
    # top_3 = [top1_retrieved, top2_retrieved, top3_retrieved]
    return top1_retrieved


def BM25_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    top_docs = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=3)
    result_files = [key for key, value in corpus_dict.items() if value in top_docs]
    return result_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()

    answer_dict = {"answers": []}
    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)

    # insurance問題
    source_path_insurance = os.path.join(args.source_path, 'insurance')
    with open("/Users/chiangsssssss/PycharmProjects/Fintech_project/insurance/insurance_formal.json", 'rb') as f:
        corpus_dict_insurance = json.load(f)
        corpus_dict_insurance = {int(key): value for key, value in corpus_dict_insurance.items()}


    for q_dict in qs_ref['questions']:
        print(q_dict)
        if q_dict['category'] == 'insurance':
            v_retrieved = voyage_retriever(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": v_retrieved})
            time.sleep(1)
        else:
            continue

    # 將答案字典保存為json文件
    with open('answer_formal', 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
