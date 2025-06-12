import unicodedata

accentMots = ['à', 'â', 'æ', 'Æ', 'ç', 'é', 'è', 'ë', 'ê', 'î', 'ï', 'ô', 'œ', 'Œ', 'ù', 'û', 'ü', 'À', 'É' ,'È' ,'Ç']
ordinalMots = ['a', 'a', 'ae', 'AE', 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'o', 'oe', 'OE', 'u', 'u', 'u', 'A', 'E', 'E', 'C']
count = [0] * len(accentMots)

for j in range(120, 130):
    print(str(j+1) + "번째 작업 수행중")
    result = []
    with open("OSCAR/OUTPUT/oscar_" + str(j + 1) + ".txt", 'r', encoding = 'utf-8') as f:
        for line in f:
            for i in range(len(line) - 1, -1 , -1):
                if(line[i] in accentMots):
                    #print(line[i] + " -> " + ordinalMots[accentMots.index(line[i])])
                    index = i
                    motIndex = accentMots.index(line[i])

                    line = list(line)
                    del line[index]
                    line.insert(index, ordinalMots[motIndex])

                    count[motIndex] += 1
                    line = ''.join(line)

            result.append(line)

    with open("OSCAR/INPUT/oscar_" + str(j + 1) + ".txt", 'w', encoding = 'utf-8') as f:
        print("OSCAR/INPUT/oscar_" + str(j + 1) + ".txt 에 파일을 저장")
        for l in result:
            f.write(l)

for i in range(len(accentMots)):
    print(accentMots[i] + " -> " + str(count[i]))