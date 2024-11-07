import json
import tqdm
import pandas as pd

# 18 kinds of relations including 6 entity-level and 12 event-level relations
extract_prompts = {
    'MadeOf': 'Extract entity1 and entity2 from the text where entity1 is made of entity2. Text: ',
    'AtLocation': 'Extract entity1 and entity2 from the text where entity1 is located at entity2. Text: ',
    'isA': 'Extract entity1 and entity2 from the text where entity1 is entity2. Text: ',
    'Partof': 'Extract entity1 and entity2 from the text where entity1 is a part of entity2. Text: ',
    'HasA': 'Extract entity1 and entity2 from the text where entity1 has entity2. Text: ',
    'UsedFor': 'Extract entity1 and entity2 from the text where entity1 is used for entity2. Text: ',
    'xNeed': 'Extract event1 and event2 from the text where event2 needs to be true for event1 to take place. Text: ',
    'xAttr': 'Extract event1 and event2 from the text where event2 shows how PersonX is viewed as after event1. Text: ',
    'xReact': 'Extract event1 and event2 from the text where event2 shows how PersonX reacts to event1. Text: ',
    'xEffect': 'Extract event1 and event2 from the text where event2 shows the effect of event1 on PersonX. Text: ',
    'xWant': 'Extract event1 and event2 from the text where event2 shows what PersonX wants after event1 happens. Text: ',
    'xIntent': 'Extract event1 and event2 from the text where event2 shows PersonX\'s intent for event1. Text: ',
    'oEffect': 'Extract event1 and event2 from the text where event2 shows the effect of event1 on PersonY. Text: ',
    'oReact': 'Extract event1 and event2 from the text where event2 shows how PersonY reacts to event1. Text: ',
    'oWant': 'Extract event1 and event2 from the text where event2 shows what PersonY wants after event1 happens. Text: ',
    'isAfter': 'Extract event1 and event2 from the text where event1 happens after event2. Text: ',
    'HasSubEvent': 'Extract event1 and event2 from the text where event1 includes event2. Text: ',
    'HinderedBy': 'Extract event1 and event2 from the text where event1 fails to happen because event2. Text: '
}

link_prompts = {
    'MadeOf': ' is made of ',
    'AtLocation': ' is located at ',
    'isA': ' is  ',
    'Partof': ' is a part of ',
    'HasA': ' has ',
    'UsedFor': ' is used for ',
    'xNeed': ' needs to be true for the occurrence of ',
    'xAttr': ' shows how PersonX is viewed as after ',
    'xReact': ' shows how PersonX reacts to ',
    'xEffect': ' shows the effect of PersonX\'s on ',
    'xWant': ' shows what PersonX wants after the occurrence of ',
    'xIntent': ' shows PersonX\'s intent for ',
    'oEffect': ' shows the effect of PersonY\' ',
    'oReact': ' shows how PersonY reacts to ',
    'oWant': ' shows what PersonY wants after the occurrence of ',
    'isAfter': ' happens after ',
    'HasSubEvent': ' includes ',
    'HinderedBy': ' fails to happen because '
}

extract_prompts_cn = {
    'MadeOf': '从下面的文本中抽取实体1和实体2，其中实体1是由实体2制成的。文本：',
    'AtLocation': '从下面的文本中抽取实体1和实体2，其中实体1位于实体2。文本：',
    'isA': '从下面的文本中抽取实体1和实体2，其中实体1是实体2。文本：',
    'Partof': '从下面的文本中抽取实体1和实体2，其中实体1是实体2的一部分。文本：',
    'HasA': '从下面的文本中抽取实体1和实体2，其中实体1包含实体2。文本：',
    'UsedFor': '从下面的文本中抽取实体1和实体2，其中实体1被用于实体2。文本：',
    'xNeed': '从下面的文本中抽取事件1和事件2，其中事件1发生时事件2才会实现。文本：',
    'xAttr': '从下面的文本中抽取事件1和事件2，其中事件2表示事件1发生后X被看待。文本：',
    'xReact': '从下面的文本中抽取事件1和事件2，其中事件2表示X对事件1发生时的反应。文本：',
    'xEffect': '从下面的文本中抽取事件1和事件2，其中事件2表示事件1对X的影响。文本：',
    'xWant': '从下面的文本中抽取事件1和事件2，其中事件2表示事件1发生后X想要什么。文本：',
    'xIntent': '从下面的文本中抽取事件1和事件2，其中事件2表示X对事件1的意图。文本：',
    'oEffect': '从下面的文本中抽取事件1和事件2，其中事件2表示事件1对Y的影响。文本：',
    'oReact': '从下面的文本中抽取事件1和事件2，其中事件2表示Y对事件1发生时的反应。文本：',
    'oWant': '从下面的文本中抽取事件1和事件2，其中事件2表示事件1发生后Y想要什么。文本：',
    'isAfter': '从下面的文本中抽取事件1和事件2，其中事件1发生在事件2之后。文本：',
    'HasSubEvent': '从下面的文本中抽取事件1和事件2，其中事件1包含事件2。文本：',
    'HinderedBy': '从下面的文本中抽取事件1和事件2，其中事件1因为事件2的发生而失败。文本：'
}

link_prompts_cn = {
    'MadeOf': ' 被制成于 ',
    'AtLocation': ' 位于 ',
    'isA': ' 是 ',
    'Partof': ' 是一部分于 ',
    'HasA': ' 包含 ',
    'UsedFor': ' 被用于 ',
    'xNeed': ' 将实现于发生后于 ',
    'xAttr': ' 表示X将被怎样看待发生后于 ',
    'xReact': ' 表示X如何对待 ',
    'xEffect': ' 表示X如何被影响于 ',
    'xWant': ' 表示X想要什么发生后于 ',
    'xIntent': ' 表示X的意图对于 ',
    'oEffect': ' 表示Y如何被影响于 ',
    'oReact': ' 表示Y如何对待 ',
    'oWant': ' 表示Y想要什么发生后于 ',
    'isAfter': ' 发生后于 ',
    'HasSubEvent': ' 包含 ',
    'HinderedBy': ' 失败由于 '
}

comet_prompts_cn = {
    'MadeOf': '以下实体是由什么制成：',
    'AtLocation': '以下实体位于何处：',
    'isA': '以下实体是什么：',
    'Partof': '以下实体是什么的一部分：',
    'HasA': '以下实体包含什么：',
    'UsedFor': '以下实体被用于什么：',
    'xNeed': '以下事件有哪些必要的先决条件：',
    'xAttr': '以下事件发生后，你将被怎样看待：',
    'xReact': '以下事件发生后，你有什么感觉：',
    'xEffect': '下面的事件发生后，可能会发生什么：',
    'xWant': '下面的事件发生后，你想要什么：',
    'xIntent': '以下事件的动机是什么：',
    'oEffect': '下面的事件发生后，可能会发生什么：',
    'oReact': '以下事件发生后，你有什么感觉：',
    'oWant': '下面的事件发生后，你想要什么：',
    'isAfter': '下面的事件发生在什么事件之后：',
    'HasSubEvent': '以下事件包含什么：',
    'HinderedBy': '以下事件的失败是由于什么：'
}


def dataprocess(path):
    data_list = json.load(open(path, 'r', encoding='utf-8'))
    df_data = []
    df_label = []
    df_year = []
    df_entity = []
    for item in data_list:
        df_data.append(item['content'])
        df_label.append(item['label'])
        df_year.append(item['time'])
        df_entity.append(item['entity_list'])
    return df_data, df_label, df_year, df_entity

