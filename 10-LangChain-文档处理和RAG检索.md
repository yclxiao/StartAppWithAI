我们知道LLM（大语言模型）的底模是基于已经过期的公开数据训练出来的，对于新的知识或者私有化的数据LLM一般无法作答，此时LLM会出现“幻觉”。针对“幻觉”问题，一般的解决方案是采用RAG做检索增强。

本文聊聊如何使用LangChain结合LLM快速做一个私有化的文档搜索工具。之前介绍过，LangChain几乎是LLM应用开发的第一选择，它的野心也比较大，它致力于将自己打造成LLM应用开发的最大社区。自然，它有这方面的成熟解决方案。

## 1、RGA检索流程

文档或者数据分为私有和共有，大模型厂商不可能把所有私有的数据都丢给LLM去学习，比如某个公司积累的某个行业的大量内部知识。所以LLM一般无法回答私有知识的问题，此时就需要一个私有化的文档搜索工具了。

那开发私有化文档搜索工具的第一步就是整理出私有文档，然后将文档喂给大模型，让其学习。所以使用 LangChain 实现私有化文档搜索的主要流程，如下图所示：

```text
文档加载 → 文档分割 → 文档嵌入 → 向量化存储 → 文档检索 → 生成回答
```

![](https://img.mangod.top/blog/202406142215549.png)

![](https://img.mangod.top/blog/202406142229018.png)

## 2、文档处理

### 2.1. 文档加载

首先，我们需要加载文档数据。文档可以是各种格式，比如文本文件、PDF、Word 等。使用 LangChain，可以轻松地加载这些文档。下面以PDF为例：

```Python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./GV2.pdf")
docs = loader.load()

```

### 2.2. 文档分割

加载的文档通常会比较大，为了更高效地处理和检索，我们需要将文档分割成更小的段落或句子。LangChain 提供了便捷的文本分割工具，可以按句子、块长度等方式分割文档。

```Python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=20,
    separators=["\n", "。", "！", "？", "，", "、", ""],
    add_start_index=True,
)
texts = text_splitter.split_documents(docs)

```

分割后的文档内容可以进一步用于生成向量。

### 2.3. 文档嵌入 Embeddings

文档分割后，我们需要将每一段文本转换成向量，这个过程称为文档嵌入。文档嵌入是将文本转换成高维向量，这是相似性搜索的关键。这里我们选择OpenAI的嵌入模型来生成文档的嵌入向量。

```Python
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    openai_api_key="sk-xxxxxxxxxxx",
    openai_api_base="https://api.302.ai/v1",
)

txts = [txt.page_content for txt in texts]

embeddings = embeddings_model.embed_documents(txts)

```

### 2.4. 文档向量化存储

接下来，我们需要将生成的向量化的文档，存入向量数据库中。向量数据库主要用来做相似性搜索，可以高效地存储和检索高维向量。LangChain 支持与多种向量数据库的集成，比如 Pinecone、FAISS、Chroma 等。

本文以FAISS为例，首先需要安装FAISS，直接使用`pip install faiss-cpu`安装。

```Python
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(texts, embeddings_model)
FAISS.save_local(db, "faiss_db2")

```

## 3、文档检索

### 3.1、文档检索

当用户提出问题时，我们需要在向量数据库中检索最相关的文档。检索过程是计算用户问题的向量表示，然后在向量数据库中查找与之最相似的文档。最后将找到的文档内容，拼接成一个大的上下文。

向量数据库的检索支持多种模式，本文先用最简单的，后续再出文章继续介绍别的模式。

```Python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = db.as_retriever()
#retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold":.1,"k":5})
#retriever = db.as_retriever(search_type="mmr")
#retriever = MultiQueryRetriever.from_llm(retriever = db.as_retriever(),llm = model)

context = retriever.get_relevant_documents(query="张学立是谁？")

_content = ""
for i in context:
    _content += i.page_content

```

### 3.2、将检索内容丢给LLM作答

最后，我们需要将检索到的文档内容丢入到 prompt 中，让LLM生成回答。LangChain 可以PromptTemplate模板的方式，将检索到的上下文动态嵌入到 prompt 中，然后丢给LLM，这样可以生成准确的回答。

```Python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

question = "张学立是谁？"
template = [
    (
        "system",
        "你是一个处理文档的助手,你会根据下面提供<context>标签里的上下文内容来继续回答问题.\n 上下文内容\n <context>\n{context} \n</context>\n",
    ),
    ("human", "你好！"),
    ("ai", "你好"),
    ("human", "{question}"),
]
prompt = ChatPromptTemplate.from_messages(template)

messages = prompt.format_messages(context=_content, question=question)
response = model.invoke(messages)

output_parser = StrOutputParser()
output_parser.invoke(response)
```

### 3.3、文档处理和检索的完整代码

最后，将以上所有代码串起来，整合到一起，如下：

```Python
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key="sk-xxxxxxx",
    openai_api_base="https://api.302.ai/v1",
)

loader = PyPDFLoader("./GV2.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=20,
    separators=["\n", "。", "！", "？", "，", "、", ""],
    add_start_index=True,
)
texts = text_splitter.split_documents(docs)

embeddings_model = OpenAIEmbeddings(
    openai_api_key="sk-xxxxxxx",
    openai_api_base="https://api.302.ai/v1",
)
txts = [txt.page_content for txt in texts]
embeddings = embeddings_model.embed_documents(txts)

db = FAISS.from_documents(texts, embeddings_model)
FAISS.save_local(db, "faiss_db2")


retriever = db.as_retriever()

template = [
    (
        "system",
        "你是一个处理文档的助手,你会根据下面提供<context>标签里的上下文内容来继续回答问题.\n 上下文内容\n <context>\n{context} \n</context>\n",
    ),
    ("human", "你好！"),
    ("ai", "你好"),
    ("human", "{question}"),
]
prompt = ChatPromptTemplate.from_messages(template)


question = "张学立是谁？"
context = retriever.get_relevant_documents(query=question)
_content = ""
for i in context:
    _content += i.page_content

messages = prompt.format_messages(context=_content, question=question)
response = model.invoke(messages)

output_parser = StrOutputParser()
output_parser.invoke(response)
```



## 4、实现检索的原理-向量搜索

以上我们在对文档做检索时，用到的是一种向量检索的方案。向量检索让系统越来越“聪明”。

### 4.1、向量搜索是什么鬼？

首先，向量搜索到底是什么呢？简单来说，它是一种“懂你”的搜索技术。

传统搜索引擎一般会根据你输入的关键词，去找那些完全匹配的内容。但是向量搜索不一样，它更聪明，不仅是匹配关键词，而且会试图理解你真正的意图和上下文，然后去找那些最符合你需求的内容。

你可以把它想象成一个特别懂你的“老朋友”，它知道你要的是什么，即使你好像啥也没说清楚。

向量搜索的2个明显应用就是`推荐系统`和`知识库`。无论是购物、音乐推荐 还是 知识库检索，都是向量搜索在背后默默工作。

比如说，你在某个音乐平台听了一首歌，平台不仅会推荐风格相似的歌曲，还会根据歌表达的情绪、歌词的内容给你推荐一些更加相似的歌曲。

### 4.2、向量搜索的核心是向量和维度

那么，向量搜索是怎么做到这些的呢？关键就在“向量”和“维度”。

在数学里，向量是有方向和大小的，而在向量搜索中，文字或数据会被转换成一个“高维向量”。

每个维度代表着数据的不同特性，比如情感、语义或者上下文。想象一下，这些向量在高维空间中变成了一个个点，而搜索的过程就是在这个复杂的空间中找离你需求最近的点。

以上的解释可能有点抽象，可以这样理解下：传统搜索就像在一张平面地图上**精确找点**，而向量搜索则是在一个3D立体空间（**多维空间**）中**找近似点**，而且考虑的因素更多更复杂。

### 4.3、向量搜索改变了搜索方式

向量搜索不仅让搜索变得更智能了，也改变了我们获取信息的方式。信息化社会下，信息是爆发式增长的，数据不仅量大而且非常混乱。

向量搜索能够将这些数据转化为我们可以理解和操作的形式。它不仅能够帮助我们寻找精确的信息，还能够通过多个维度寻找最接近的信息，包括从 同义词、含义、意图和上下文等各个角度。

**向量搜索不仅是对单个词进行搜索，而且还会分析词与词之间的复杂关系**，从而更好地理解每次选择是否更接近或偏离检索句子的含义。

这样一来，我们不仅获取到信息，而且找到了更有意义的结果。

过去，我们需要输入非常精确的关键词才能找到想要的信息，但现在即使描述得比较模糊，向量搜索也能帮我们找到最相关的内容。

这对普通用户来说太方便了，不需要搞懂各种专业术语，只要使用自然语言大致将需求表达清楚，就能得到准确的结果。

## 5、知识库框架

以上已经介绍了如何通过LangChain搭建知识库，但是在搭建AI知识库这件事儿上，有不少成熟的框架，我推荐使用FastGPT。这篇文章笔者就使用过的两款平台做个比较，FastGPT和百度千帆平台。

### 5.1、为什么要搭建知识库

随着企业的运营，企业的私有数据越来越多（结构化、半结构化、非结构化的数据）。这么多数据，我们不可能都记在大脑里，就算老员工能记住，那对于一个新人来说如何快速上手呢？

所以搭建知识库就成了刚需。得益于LLM（大模型）的发展，让AI可以更好的理解自然语言，再加上向量数据库的检索，让许多沉睡的知识可以被唤醒。

企业可以通过搭建知识库提高工作效率，提升服务质量，还可以基于知识做出更好的决策。

### 5.2、技术方案

#### 5.2.1、我推荐的方案

LLM存在幻觉问题，对于它不知道的知识会胡编乱造，可靠性差。所以在搭建知识库的技术方案上，一般有2个争论：

1. 企业自己微调大模型。
2. 使用RAG检索增强技术。

微调大模型，就是将已有的知识喂给LLM，让LLM学习。暂不说这种方式的安全性如何。单单从模型迭代的角度来看就不合适。

不管是部署开源模型 还是 采用闭源模型，随着时间的推移，模型的迭代非常快，模型会越来越聪明。模型每14天都会小迭代一次。迭代之后，之前投喂的数据会失效，而且之前做过的优化，会随着模型能力变强后，会变成无用功。

所以，我更推荐RAG技术。检索增强生成（Retrieval Augmented Generation，RAG）是一种强大的工具，它可以将企业的私有知识 通过LLM + 外接向量数据库的方式整合到一起。

#### 5.2.3、两个主角

方案和步骤有了，下面看看选哪种框架。业内用LLM做知识库的方案较多，比如FastGPT、Dify、自己基于LangChain开发、百度的千帆平台等等。

本篇主要介绍FastGPT和 百度千帆平台。基于以上LangChain搭建知识库检索的流程，可以更好的理解这种开源的完善的知识库框架的使用方式。

- 百度千帆平台是百度推出的基于大模型的一站式应用解决方案平台，提供先进的生成式AI生产及应用全流程开发工具链。**主打一个一站式。**百度深耕AI多年，也是国内最早推出一站式平台的，整体还算方便。

![](https://img.mangod.top/blog/202407241346114.png)

- FastGPT 是一个基于 LLM 大语言模型的知识库问答系统，提供开箱即用的数据处理、模型调用等能力。同时可以通过 Flow 可视化进行工作流编排，从而实现复杂的问答场景！**主打一个知识库问答。**

![](https://img.mangod.top/blog/202407241347492.png)

下面就两个主角在知识库问答领域的效果***做个比较***。

### 5.3、使用比较

#### 5.3.1、操作界面

操作界面上都差不多，从首页，到上传文件到知识库，再到创建AI应用，操作都很简便。

我个人更喜欢FastGPT的页面风格，页面比较清爽，很明显地看到 知识库、创建大模型应用 这2个版块。

**FastGPT**

首页

![](https://img.mangod.top/blog/202407241356634.png)

创建知识库，将处理好的本地文档、网页、QA问答上传，然后利用词嵌入模型处理，再存入向量数据库。

![](https://img.mangod.top/blog/202407241409222.png)

创建AI应用

![](https://img.mangod.top/blog/202407241413773.png)

**百度千帆**

首页：

![](https://img.mangod.top/blog/202407241355376.png)

创建知识库，将处理好的本地文档、网页、QA问答上传，然后利用词嵌入模型处理，再存入向量数据库。

![](https://img.mangod.top/blog/202407241402411.png)

创建AI应用

![](https://img.mangod.top/blog/202407241414590.png)

#### 5.3.2、可选词嵌入模型

在文档嵌入这一步，需要选择词嵌入模型。FastGPT可以选择适合自己的模型，但是百度千帆没有选择项，只能用百度的模型。

**FastGPT**

![](https://img.mangod.top/blog/202407241406979.png)



#### 5.3.3、可选问答模型

在AI回答这一步，也需要用到大模型。FastGPT可以选择适合自己的模型，但是百度千帆没有选择项，只能用百度的模型。

**FastGPT**

![](https://img.mangod.top/blog/202407241427699.png)

**百度千帆**

![](https://img.mangod.top/blog/202407241428375.png)

#### 5.3.4、可以发布到的渠道

发布渠道这一局，算百度完胜了！百度千帆集成了多个发布渠道，使用感受较好。FastGPT相对来说就有点少了，而且还需要额外的搭建和部署。

**FastGPT**

![](https://img.mangod.top/blog/202407241729128.png)

![](https://img.mangod.top/blog/202407241433759.png)

**百度千帆**

![](https://img.mangod.top/blog/202407241728495.png)

#### 5.3.5、最终的回答效果和体验

他俩在正常的问答结果上都差不多，这个结果跟选择的词嵌入模型和问答模型有关。但是在使用的体验上，我有几点要吐槽：

- FastGPT的对话框，历史对话消息不会丢失。但百度千帆的对话框刷新下，历史对话会丢失，这个体验感有点差。
- 百度千帆的对话界面，不知道为什么非要突出老大一个图标和应用标题，几乎占了小半屏，导致对话内容的可见区域被压缩，此处我要@下百度的产品经理，请问你这么设计的目的是啥？

![](https://img.mangod.top/blog/202407241441938.png)

- 对比看下FastGPT就很简洁，对话区域也很大，用起来比较舒适。

![](https://img.mangod.top/blog/202407241727277.png)

- 百度千帆分享出去的网页必须要登录，这个有点恶心，没必要在这里强行拉一波注册吧。而FastGPT分享的网页免登即可使用。



#### 5.3.6、源码开放

**FastGPT**

FastGPT源码开放，可以自己搭建，模型也可以自己搭配。对于企业内部使用非常友好。

自行搭建的话，需要开发者介入。不过FastGPT也有云上的SaaS版本，按需付费使用，无需开发者介入。

**百度千帆**

百度千帆是百度的闭源产品，模型也是闭源，而且模型只能用百度。这个对于企业内部使用不友好。

无需开发者介入，稍微懂些产品，看下文档，即可使用。

#### 5.3.7、数据安全

如果从数据安全方面考虑，只能选择FastGPT了，所有的数据都可以存储在企业自己的服务器上。

### 5.4、方案总结

总之，没有最优的方案，只有最合适的方案。

大部分场景下FastGPT都是比较胜任的。如果只考虑以最小代价快速实现一个知识库问答，我推荐使用百度千帆平台。其余情况，我推荐使用FastGPT。

在操作界面上、词嵌入模型上、问答模型上、对话体验和数据安全上，我觉得FastGPT都是很专业的。FastGPT的发力点就是在构建专业的知识库领域。

## 6、总结

本文主要聊了，使用LangChain轻松实现私有化文档搜索，也介绍了检索背后的原理，还介绍了2个完善的知识库框架。

