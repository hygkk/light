import json
# 跳过目录部分
# 去掉页脚
# 不保留(***)之后的 ?


def main():
    data={}
    for i in range(1,4):
        with open(r'en_conversations_with_god_{}.txt'.format(i)) as f:
            book='book{}'.format(i)
            data[book]={}
            n_1=0
            n_chapter=1
            footer=False
            for line in f.readlines():
                this_line=line.strip()
                if not this_line:
                    continue
                if this_line=='cosmic-people.com':
                    footer=False
                    continue
                if footer:
                    continue
                if this_line[:28]=='CONVERSATIONS WITH GOD, Book':
                    footer=True
                    continue
                if this_line=='(1)': # 跳过目录部分....
                    n_1+=1
                if n_1==2:  # 开始提取
                    if this_line=='(***)':
                        break
                    if this_line=='({})'.format(n_chapter):  # 新章节
                        chapter='chapter{}'.format(n_chapter)
                        data[book][chapter]=[{'text':''}]
                        n_chapter+=1
                        continue
                    if this_line!='Chapter {}'.format(n_chapter-1):
                        for item in data[book][chapter]:
                            item['text']+=' '+this_line
                            
    with open(r"/data/huyangge/nlptutorial/task02-get_dataset/text.json",'w',encoding='utf-8') as f:
        json.dump(data,f)



if __name__ == "__main__":
    main()

