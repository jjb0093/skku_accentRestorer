import json
import re
from tqdm import tqdm

emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U00002700-\U000027BF"  # Dingbats
    u"\U000024C2-\U0001F251"  # Enclosed characters
    "]+",
    flags=re.UNICODE
)
skip = ['(', ':', '[', '..', '_', '·', '|', '•', '©' , '>', '<']

for i in range(120, 130):
    print(str(i + 1) + "번째 데이터 처리 중..")

    data = []
    content_before = ""
    with open("OSCAR/DATA/fr_meta_part_" + str(i + 1) + ".jsonl", 'r', encoding = 'utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line)
            content = item.get('content')

            if(content[:10] != content_before):
                content = re.sub(r'[\s]+', ' ', content).strip()
                content = emoji_pattern.sub('', content)

                if not any(s in content for s in skip) and len(content.split()) > 10:
                    data.append(content)
                
                content_before = content[:10]

    with open("OSCAR/OUTPUT/oscar_" + str(i + 1) + ".txt", 'w', encoding = 'utf-8') as f:
        for line in data:
            f.write(line)
            f.write('\n')