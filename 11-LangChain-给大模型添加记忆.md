LLM存在一个显著的缺陷：没有记忆。在对话中，无法记住上下文的 LLM 常常会让用户感到困扰。本文聊聊如何利用 LangChain快速为 LLM 添加记忆能力 以及原理。

## 1、LLM 缺陷-没有记忆

### 1.1、没有记忆

当前的 LLM 非常智能，在理解和生成自然语言方面表现优异，但是有一个显著的缺陷：**没有记忆**。

LLM 的本质是基于统计和概率来生成文本，对于每次请求，它们都将上下文视为独立事件。这意味着当你与 LLM 进行对话时，它不会记住你之前说过的话，这就导致了 LLM 有时表现得不够智能。

这种“无记忆”属性使得 LLM 无法在长期对话中有效跟踪上下文，也无法积累历史信息。比如，当你在聊天过程中提到一个人名，后续再次提及该人时，LLM 可能会忘记你之前的描述。

### 1.2、没有记忆的烦恼

当我们与 LLM 聊天时，它们无法记住上下文信息，比如下图的示例：

![](https://img.mangoant.top/blog/202406111642618.png)

## 2、记忆组件的原理

### 2.1、安装记忆的核心步骤

给LLM安装记忆的核心步骤就3个：

1. 在对话之前调取之前的历史消息。
2. 将历史消息填充到Prompt里。
3. 对话结束后，继续将历史消息保存到到memory记忆中。

![](https://img.mangoant.top/blog/202407090806494.png)

### 2.2、示例

如果将已有信息放入到 memory 中，每次跟 LLM 对话时，把已有的信息丢给 LLM，那么 LLM 就能够正确回答，见如下示例：

![](https://img.mangoant.top/blog/202406111645709.png)

目前业内解决 LLM 记忆问题就是采用了类似上图的方案，即：**将每次的对话记录再次丢入到 Prompt 里**，这样 LLM 每次对话时，就拥有了之前的历史对话信息。

但如果每次对话，都需要自己手动将本次对话信息继续加入到`history`信息中，那未免太繁琐。有没有轻松一些的方式呢？**有，LangChain**！LangChain 对记忆组件做了高度封装，开箱即用，下面继续介绍。

### 2.3、长期记忆和短期记忆

在解决 LLM 的记忆问题时，有两种记忆方案，长期记忆和短期记忆。

- **短期记忆**：基于内存的存储，容量有限，用于存储临时对话内容。
- **长期记忆**：基于硬盘或者外部数据库等方式，容量较大，用于存储需要持久的信息。

在LangChain中使用`ConversationBufferMemory`作为短时记忆的组件，实际上就是以键值对的方式将消息存在内存中。

如果碰到较长的对话，一般使用`ConversationSummaryMemory`对上下文进行总结，再交给大模型。或者使用`ConversationTokenBufferMemory`基于固定的token数量进行内存刷新。

如果想对记忆进行长时间的存储，则可以使用向量数据库进行存储（比如FAISS、Chroma等），或者存储到Redis、Elasticsearch中。

下面介绍几个LangChain常用的安装以及的方法。

## 3、给 LLM 安装记忆的使用方法

为了让开发者聚焦于业务实现，LangChain贴心地封装了一套LLM记忆方案。使用方式如下：

### 3.1、单独用 ConversationBufferMemory 做短期记忆

Langchain 提供了 `ConversationBufferMemory` 类，可以用来存储和管理对话。

`ConversationBufferMemory` 包含`input`变量和`output`变量，`input`代表人类输入，`output`代表 AI 输出。

每次往`ConversationBufferMemory`组件里存入对话信息时，都会存储到`history`的变量里。

![](https://img.mangoant.top/blog/202406111811838.png)

### 3.2、利用 MessagesPlaceholder 手动添加 history

```Python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.load_memory_variables({})

memory.save_context({"input": "我的名字叫张三"}, {"output": "你好，张三"})
memory.load_memory_variables({})

memory.save_context({"input": "我是一名 IT 程序员"}, {"output": "好的，我知道了"})
memory.load_memory_variables({})

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)
chain = prompt | model

user_input = "你知道我的名字吗？"
history = memory.load_memory_variables({})["history"]


chain.invoke({"user_input": user_input, "history": history})

user_input = "中国最高的山是什么山？"
res = chain.invoke({"user_input": user_input, "history": history})
memory.save_context({"input": user_input}, {"output": res.content})


res = chain.invoke({"user_input": "我们聊得最后一个问题是什么？", "history": history})

```

执行结果如下：

![](https://img.mangoant.top/blog/202406120656199.png)

### 3.3、利用 ConversationChain 自动添加 history

如果每次都是自己添加history就比较麻烦了，LangChain 的`ConversationChain`对话链，可以用自动添加`history`的方式添加临时记忆，无需手动添加。一个`链`实际上就是将一部分繁琐的小功能做了高度封装，这样多个链就可以组合形成易用的强大功能。这里`链`的优势一下子就体现出来了：

```Python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

memory = ConversationBufferMemory(return_messages=True)
chain = ConversationChain(llm=model, memory=memory)
res = chain.invoke({"input": "你好，我的名字是张三，我是一名程序员。"})
res['response']

res = chain.invoke({"input":"南京是哪个省？"})
res['response']

res = chain.invoke({"input":"我告诉过你我的名字，是什么？，我的职业是什么？"})
res['response']

```

执行结果如下，可以看到利用`ConversationChain`对话链，可以让 LLM 快速拥有记忆：

![](https://img.mangoant.top/blog/202406112300609.png)

### 3.4、对话链结合 PromptTemplate 和 MessagesPlaceholder

在 Langchain 中，`MessagesPlaceholder`是一个占位符，用于在对话模板中动态插入上下文信息。它可以帮助我们灵活地管理对话内容，确保 LLM 能够使用最上下文来生成响应。

采用`ConversationChain`对话链结合`PromptTemplate`和`MessagesPlaceholder`，几行代码就可以轻松让 LLM 拥有短时记忆。

```Python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个爱撒娇的女助手，喜欢用可爱的语气回答问题。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
memory = ConversationBufferMemory(return_messages=True)
chain = ConversationChain(llm=model, memory=memory, prompt=prompt)

res = chain.invoke({"input": "今天你好，我的名字是张三，我是你的老板"})
res['response']

res = chain.invoke({"input": "帮我安排一场今天晚上的高规格的晚饭"})
res['response']

res = chain.invoke({"input": "你还记得我叫什么名字吗？"})
res['response']


```

![](https://img.mangoant.top/blog/202406112311465.png)

### 3.5、使用长期记忆

短期记忆在会话关闭或者服务器重启后，就会丢失。如果想长期记住对话信息，只能采用长期记忆组件。

LangChain 支持多种长期记忆组件，比如`Elasticsearch`、`MongoDB`、`Redis`等，下面以`Redis`为例，演示如何使用长期记忆。

```Python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="sk-xxxxxxxxxxxxxxxxxxx",
    openai_api_base="https://api.aigc369.com/v1",
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个擅长{ability}的助手"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | model

chain_with_history = RunnableWithMessageHistory(
    chain,
    #使用redis存储聊天记录
    lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://10.22.11.110:6379/3"
    ),
    input_messages_key="question",
    history_messages_key="history",
)

#每次调用都会保存聊天记录，需要有对应的session_id
chain_with_history.invoke(
    {"ability": "物理", "question": "地球到月球的距离是多少？"},
    config={"configurable": {"session_id": "baily_question"}},
)

chain_with_history.invoke(
    {"ability": "物理", "question": "地球到太阳的距离是多少？"},
    config={"configurable": {"session_id": "baily_question"}},
)

chain_with_history.invoke(
    {"ability": "物理", "question": "地球到他俩之间谁更近"},
    config={"configurable": {"session_id": "baily_question"}},
)

```

LLM 的回答如下，同时关闭 session 后，直接再次提问最后一个问题，LLM 仍然能给出正确答案。

只要`configurable`配置的`session_id`能对应上，LLM 就能给出正确答案。

![](https://img.mangoant.top/blog/202406112333048.png)

然后，继续查看`redis`存储的数据，可以看到数据在` redis` 中是以 `list`的数据结构存储的。

![](https://img.mangoant.top/blog/202406112336183.png)



## 4、拆解LangChain的大模型记忆方案

以上我们聊过如何使用LangChain给LLM装上记忆，里面提到对话链`ConversationChain`和`MessagesPlaceholder`，可以简化安装记忆的流程。下面来拆解基于LangChain的大模型记忆方案。

### 4.1、给LLM安装记忆 — 非MessagesPlaceholder

#### 4.1.1、ConversationBufferMemory使用示例

使用`ConversationBufferMemory`进行记住上下文：

```Python
memory = ConversationBufferMemory()
memory.save_context(
    {"input": "你好，我的名字是半支烟，我是一个程序员"}, {"output": "你好，半支烟"}
)
memory.load_memory_variables({})

```

#### 4.1.2、LLMChain + ConversationBufferMemory使用示例

```Python
#prompt模板
template = """
你是一个对话机器人，以下<history>标签中是AI与人类的历史对话记录，请你参考历史上下文，回答用户输入的问题。

历史对话:
<history>
{customize_chat_history}
</history>

人类:{human_input}
机器人:

"""

prompt = PromptTemplate(
    template=template,
    input_variables=["customize_chat_history", "human_input"],
)
memory = ConversationBufferMemory(
    memory_key="customize_chat_history",
)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

chain = LLMChain(
    llm=model,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

chain.predict(human_input="你知道我的名字吗？")

#chain.predict(human_input="我叫半支烟，我是一名程序员")

#chain.predict(human_input="你知道我的名字吗？")
```

此时，已经给LLM安装上记忆了，免去了我们写那3步核心的模板代码。

对于`PromptTemplate`可以使用以上方式，但`ChatPromptTemplate`因为有多角色，所以需要使用`MessagesPlaceholder`。具体使用方式如下。

### 4.2、给LLM安装记忆 — MessagesPlaceholder

`MessagesPlaceholder`主要就是用于`ChatPromptTemplate`场景。`ChatPromptTemplate`模式下，需要有固定的格式。

#### 4.2.1、PromptTemplate和ChatPromptTemplate区别

`ChatPromptTemplate`主要用于聊天场景。`ChatPromptTemplate`有多角色，第一个是System角色，后续的是Human与AI角色。因为需要有记忆，所以之前的历史消息要放在最新问题的上方。

![](https://img.mangoant.top/blog/202407082318924.png)

#### 4.2.2、使用MessagesPlaceholder安装记忆

最终的ChatPromptTemplate + MessagesPlaceholder代码如下：

```Python
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手。"),
        MessagesPlaceholder(variable_name="customize_chat_history"),
        ("human", "{human_input}"),
    ]
)

memory = ConversationBufferMemory(
    memory_key="customize_chat_history",
    return_messages=True,
)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

chain = LLMChain(
    llm=model,
    memory=memory,
    prompt=chat_prompt,
    verbose=True,
)

chain.predict(human_input="你好，我叫半支烟，我是一名程序员。")

```

至此，我们使用了`ChatPromptTemplate`简化了构建prompt的过程。

### 4.3、使用对话链ConversationChain

如果连`ChatPromptTemplate`都懒得写了，那直接使用对话链`ConversationChain`，让一切变得更简单。实践代码如下：

```Python
memory = ConversationBufferMemory(
    memory_key="history",  # 此处的占位符必须是history
    return_messages=True,
)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

chain = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True,
)

chain.predict(input="你好，我叫半支烟，我是一名程序员。")  # 此处的变量必须是input
```

ConversationChain提供了包含AI角色和人类角色的对话摘要格式。ConversationChain实际上是对Memory和LLMChain和ChatPrompt进行了封装，简化了初始化Memory和构建ChatPromptTemplate的步骤。

### 4.4、ConversationBufferMemory的注意点

#### 4.4.1、memory_key

`ConversationBufferMemory`有一个入参是`memory_key`，表示内存中存储的本轮对话的`键`，后续可以根据`键`找到对应的值。

#### 4.4.2、使用"chat_history"还是"history"

`ConversationBufferMemory`的`memory_key`，有些资料里是设置是`memory_key="history"`，有些资料里是`"chat_history"`。

这里有2个规则，如下：

- 在使用`MessagesPlaceholder`和`ConversationBufferMemory`时，`MessagesPlaceholder`的`variable_name`和`ConversationBufferMemory`的`memory_key`可以自定义，只要相同就可以。比如这样：

```Python
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手。"),
        MessagesPlaceholder(variable_name="customize_chat_history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(
    memory_key="customize_chat_history",  # 此处的占位符可以是自定义
    return_messages=True,
)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

chain = ConversationChain(
    llm=model,
    memory=memory,
    prompt=chat_prompt,
    verbose=True,
)

chain.predict(input="你好，我叫半支烟，我是一名程序员。")  # 此处的变量必须是input
```
- 如果只是使用`ConversationChain`，***又没有使用***`MessagesPlaceholder`的场景下，ConversationBufferMemory的memory_key，***必须用***`history`。

### 4.5、MessagesPlaceholder的使用场景

`MessagesPlaceholder`其实就是在与AI对话过程中的`Prompt`的一部分，它代表`Prompt`中的历史消息这部分。它提供了一种结构化和可配置的方式来处理这些消息列表，使得在构建复杂`Prompt`时更加灵活和高效。

说白了它就是个占位符，相当于把从memory读取的历史消息插入到这个占位符里了。

比如这样，就可以表示之前的历史对话消息：

```Python
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手。"),
        MessagesPlaceholder(variable_name="customize_chat_history"),
        ("human", "{human_input}"),
    ]
)
```

是否需要使用**MessagesPlaceholder**，记住2个原则：

- `PromptTemplate`类型的模板，无需使用MessagesPlaceholder
- `ChatPromptTemplate` 类型的聊天模板，需要使用MessagesPlaceholder。但是在使用ConversationChain时，可以省去创建ChatPromptTemplate的过程（也可以不省去）。省去和不省去在输出过程中有些区别，如下：

    ![](https://img.mangoant.top/blog/202407082309973.png)

    ![](https://img.mangoant.top/blog/202407082308770.png)



## 5、总结

本文主要聊了LLM 缺乏记忆的固有缺陷和安装记忆的步骤和原理，以还讨论了如何利用 LangChain 给 LLM 装上记忆组件，让 LLM 能够在对话中更好地保持上下文。

重点要熟悉`ConversationBufferMemory`、`MessagesPlaceholder`的使用、对话链`ConversationChain`的使用和原理。