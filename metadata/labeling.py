import json
import pandas as pd

# 예: json 파일에서 데이터 로드
with open("TCIA Biobank Pathology Portal.json", "r") as f:
    metadata = json.load(f)

# pandas DataFrame으로 변환
df = pd.DataFrame(metadata)

# 'Tumor' 여부로 label 만들기
df["label"] = df["Tumor_Segment_Acceptable"].apply(lambda x: 1 if x.strip().upper() == "Y" else 0)

# 확인
print(df[["pub_id", "pub_subspec_id", "Tumor_Histologic_Type", "Tumor_Segment_Acceptable", "label"]].head())

# save as csv
result = df[["pub_id", "pub_subspec_id", "Tumor_Histologic_Type", "Tumor_Segment_Acceptable", "label"]]
result.to_csv("label.csv", index=False)
