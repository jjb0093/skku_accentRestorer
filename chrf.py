# import os
# from sacrebleu.metrics import CHRF
# import pandas as pd

# def find_low_chrf_samples(predictions, references, threshold=50.0, save_path=None):
#     chrf = CHRF()
#     low_score_samples = []
#     df = pd.DataFrame(columns=['No.', 'generated_sent', 'target_sent', 'score'])

#     for i, (pred, ref) in enumerate(zip(predictions, references), start=1):
#         if i % 100000 == 0:
#             print(i)
#         score = chrf.sentence_score(pred.strip(), [ref.strip()]).score
#         df.loc[len(df)] = [i, pred.strip(), ref.strip(), score]

#         if score < threshold:
#             low_score_samples.append((i, score, pred.strip(), ref.strip()))

#     if save_path:
#         with open(save_path, "w", encoding="utf-8") as f:
#             for i, score, pred, ref in low_score_samples:
#                 f.write(f"[{i}] chrF: {score:.2f}\n")
#                 f.write(f"PRED: {pred}\n")
#                 f.write(f"REF : {ref}\n\n")

#     return df, low_score_samples

# # 경로 설정
# base_dir = "/home/work/DL_france/model_trials/T5/0613_generated"
# target_path = "/home/work/DL_france/data/OPUS/opus.txt"

# # 디코딩 파일 목록 가져오기
# decoded_files = [f for f in os.listdir(base_dir) if f.endswith("_decoded.txt")]
# # decoded_files = decoded_files[:2]
# decoded_files = decoded_files[2:]
# print(decoded_files)

# # 참조 문장 불러오기
# with open(target_path, 'r', encoding='utf-8') as f:
#     targets = f.readlines()

# # 각 디코딩 파일 처리
# for decoded_file in decoded_files:
#     model_name = decoded_file.replace("_decoded.txt", "")
#     decoded_path = os.path.join(base_dir, decoded_file)

#     with open(decoded_path, 'r', encoding='utf-8') as f:
#         decoded = f.readlines()

#     print(f"Processing {decoded_file}...")
#     output_dir = os.path.join(base_dir, model_name)
#     os.makedirs(output_dir, exist_ok=True)

#     df, low_scores = find_low_chrf_samples(decoded, targets, threshold=50.0,
#                                            save_path=os.path.join(output_dir, "low_chrf_samples.txt"))
#     df.to_csv(os.path.join(output_dir, "chrf_score.csv"), index=False)

# print("All done.")

import os
from sacrebleu.metrics import CHRF
import pandas as pd
from multiprocessing import Pool

def process_file(args):
    decoded_path, target_path, output_dir = args
    model_name = os.path.basename(decoded_path).replace("_decoded.txt", "")

    # 파일 읽기
    with open(decoded_path, 'r', encoding='utf-8') as f:
        decoded = f.readlines()
    with open(target_path, 'r', encoding='utf-8') as f:
        targets = f.readlines()

    # 점수 계산
    chrf = CHRF()
    low_score_samples = []
    df = pd.DataFrame(columns=['No.', 'generated_sent', 'target_sent', 'score'])

    for i, (pred, ref) in enumerate(zip(decoded, targets), start=1):
        score = chrf.sentence_score(pred.strip(), [ref.strip()]).score
        df.loc[len(df)] = [i, pred.strip(), ref.strip(), score]
        if score < 50.0:
            low_score_samples.append((i, score, pred.strip(), ref.strip()))

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "low_chrf_samples.txt"), "w", encoding="utf-8") as f:
        for i, score, pred, ref in low_score_samples:
            f.write(f"[{i}] chrF: {score:.2f}\nPRED: {pred}\nREF : {ref}\n\n")
    df.to_csv(os.path.join(output_dir, "chrf_score.csv"), index=False)

    return f"Finished {model_name}"

if __name__ == "__main__":
    base_dir = "/home/work/DL_france/model_trials/T5/0613_generated"
    target_path = "/home/work/DL_france/data/OPUS/opus.txt"

    decoded_files = [f for f in os.listdir(base_dir) if f.endswith("_decoded.txt")]
    args_list = [
        (os.path.join(base_dir, f), target_path, os.path.join(base_dir, f.replace("_decoded.txt", "")))
        for f in decoded_files
    ]

    print("Starting multiprocessing...")
    with Pool(processes=4) as pool:  # 4개의 프로세스 사용
        results = pool.map(process_file, args_list)

    print("\n".join(results))