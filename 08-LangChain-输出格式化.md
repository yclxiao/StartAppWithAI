本文介绍 LangChain 的输出解析器`OutputParser`的使用，和基于LangChain的`LCEL`构建`链`。

## 1、输出解析器概述

常规的使用LangChain构建LLM应用的流程是：**Prompt 输入、调用LLM 、LLM输出**。有时候我们期望LLM给到的数据是格式化的数据，方便做后续的处理。

这时就需要在Prompt里设置好格式要求，然后LLM会在输出内容后，再将内容传给输出解析器，输出解析器会解析成我们预期的格式。

![](https://img.mangoant.top/blog/202405262252996.png)



## 2、输出解析器原理

输出解析器原理其实很简单，主要是2个部分：

- **格式化说明**：我们知道，自然语言与LLM交互的关键是提示词，所以你想让LLM做任何事情，都要在提示词中写出要求。那输出解析器中就包含了格式化的要求，这个要求会随着用户的问题 组合成最终的Prompt一起提交给LLM。
- **解析**：将LLM返回的回答，按照解析成指定的格式。

原理图如下：

![](https://img.mangoant.top/blog/202410111835769.png)

## 3、代码实践

实践2个示例：

1. 简易的CommaSeparatedListOutputParser，分隔符形式的解析器。
2. 利用PydanticOutputParser，从信息中提取自己需要的信息，然后返回对象。

### CommaSeparatedListOutputParser

示例1：将调用 LLM 的结果，解析为逗号分隔的列表。比如询问某个城市有N个景点。

```Python
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出{cityName}的{viewPointNum}个著名景点。"),
    ]
)

output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()
print(parser_instructions)

final_prompt = prompt.invoke(
    {"cityName": "南京", "viewPointNum": 3, "parser_instructions": parser_instructions}
)
print(final_prompt)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="sk-XXXXXXXXX",
    openai_api_base="https://api.xiaoai.plus/v1",
)
response = model.invoke(final_prompt)
print(response.content)

ret = output_parser.invoke(response)
print(ret)

```



![](https://img.mangoant.top/blog/202410111757297.png)



### PydanticOutputParser

使用PydanticOutputParser从信息中提取自己需要的字段，解析成自定义对象。使用步骤如下：

- 定义数据结构类，继承`BaseModel` ，定义字段`Field`
- 使用输出解析器`PydanticOutputParser`
- 后续是常规操作：生成prompt、调用LLM执行、将输出按照Parser解析

比如：给LLM一段书籍的介绍，让LLM按照指定的格式总结输出。

```Python
from typing import List
from pydantic import BaseModel,Field

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
#from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI


class BookInfo(BaseModel):
    book_name: str = Field(description="书籍的名字")
    author_name: str = Field(description="书籍的作者")
    genres: List[str] = Field(description="书籍的体裁")


output_parser = PydanticOutputParser(pydantic_object=BookInfo)
#查看输出解析器的内容，会被输出成json格式
print("解析器里的格式说明======", output_parser.get_format_instructions())

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions} 你输出的结果请使用中文。"),
        (
            "human",
            "请你帮我从书籍的概述中，提取书名、作者，以及书籍的体裁。书籍概述会被 ### 符号包围。\n### {book_introduction} ###",
        ),
    ]
)

book_introduction = """
《朝花夕拾》原名《旧事重提》，是现代文学家鲁迅的散文集，收录鲁迅于1926年创作的10篇回忆性散文， [1]1928年由北京未名社出版，现编入《鲁迅全集》第2卷。
此文集作为“回忆的记事”，多侧面地反映了作者鲁迅青少年时期的生活，形象地反映了他的性格和志趣的形成经过。前七篇反映他童年时代在绍兴的家庭和私塾中的生活情景，后三篇叙述他从家乡到南京，又到日本留学，然后回国教书的经历；揭露了半殖民地半封建社会种种丑恶的不合理现象，同时反映了有抱负的青年知识分子在旧中国茫茫黑夜中，不畏艰险，寻找光明的困难历程，以及抒发了作者对往日亲友、师长的怀念之情 [2]。
文集以记事为主，饱含着浓烈的抒情气息，往往又夹以议论，做到了抒情、叙事和议论融为一体，优美和谐，朴实感人。作品富有诗情画意，又不时穿插着幽默和讽喻；形象生动，格调明朗，有强烈的感染力。
"""

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="sk-nPPNdkSwEV74wNDz2615001aE9Fa4a068b568cC9F47cF054",
    openai_api_base="https://api.xiaoai.plus/v1",
)
final_prompt = prompt.invoke(
    {
        "book_introduction": book_introduction,
        "parser_instructions": output_parser.get_format_instructions(),
    }
)
response = model.invoke(final_prompt)
print("回答======",response.content)
result = output_parser.invoke(response)
print("对象======", result)
```

![](https://img.mangoant.top/blog/202410111812090.png)



## 4、总结

本文主要聊了LangChain的输出解析器，希望对你有帮助！

