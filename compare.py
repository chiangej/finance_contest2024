import json

with open('dataset/preliminary/pred_retrieve_RE1.json', 'r', encoding='utf-8') as f:
    answer_data = json.load(f)

with open('dataset/preliminary/ground_truths_example.json', 'r', encoding='utf-8') as f:
    reference_data = json.load(f)

answer_dict = {item['qid']: item['retrieve'] for item in answer_data['answers'][50:100]}
reference_dict = {item['qid']: item['retrieve'] for item in reference_data['ground_truths'][50:100]}

correct_count = 0
total_count = len(reference_dict)

for qid, correct_retrieve in reference_dict.items():
    retrieved_value = answer_dict.get(qid)
    if retrieved_value == correct_retrieve:
        correct_count += 1
    else:
        print(f"{qid} correct_ans: {correct_retrieve}, retrieved_ans: {retrieved_value}")

accuracy = correct_count / total_count * 100

print(f'總問題數量: {total_count}')
print(f'正確數量: {correct_count}')
print(f'準確率: {accuracy:.2f}%')
