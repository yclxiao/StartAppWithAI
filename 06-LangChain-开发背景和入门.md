本篇主要介绍LLM（大模型）应用开发背景和入门相关内容。

## 1、LLM应用开发

AI的发展如火如荼，LLM的发展更是火爆。LLM火爆的原因，其实也简单，主要是满足了2个特性：

- 提高生产力：LLM是一个聪明的大脑，确实可以提高生产力。
- 可以实际落地：LLM确实可以落地，与业务相结合可以满足一些实际需求，并非像元宇宙那样的空中楼阁。

LLM 应用开发目前有两个比较热门的方向，一个是 `RAG 应用开发`，另外一个是 `AI Agent 开发`。

- RAG应用开发：利用LLM对外部知识库做检索的一种AI 应用。它通过检索获取相关的知识并将其融入 Prompt，让大模型能够参考相应的知识库从而给出超出训练集以外的内容，同时也能消除大模型 “幻觉” 的问题。并且这种检索方式可以更好的理解自然语言的提问方式，极大的提升检索效率。
- AI Agent：LLM就是一个超级大脑，有很强的推理决策能力，通过给这颗大脑增加规划、记忆和工具调用的能力，可以构造出一个能够完成给定目标的智能体。

## 2、LangChain简介

LangChain是啥？说白了，LangChain就是一个开发LLM应用程序的框架。

开发LLM应用有多种实现方式，LangChain是比较流行的一种。这两年LangChain发展迅猛，GitHub上的star数飙升，近期更是成功融资2亿美刀，足见其受欢迎程度。

在Java领域，SpringBoot可谓是一把瑞士军刀，是一个非常完善的开发企业级应用的框架。它强大的生态可以帮助开发者节省许多时间，从而把精力聚焦于业务逻辑上。

同样的，LangChain也是类似于SpringBoot，它是一个开发LLM应用程序的框架，它集成了多个LLM和多种外部组件（比如记忆、检索、向量数据库、工具集等等），方便开发者快速开发LLM应用。

后来LangChain还引入了LangSmith用于监控LLM应用，还有LangServe用于部署LLM应用。

LangChain对自己的定位是：**构建LLM App的最大社区，基于LangChain可以开发出可推理的AI应用程序。**

LangChain的架构设计有以下几个核心模块：

- LangChain-Core：抽象LangChain的内核 和 LangChain 表达式语言。
- LangChain-Community：社区的各种部件，比如模型的输入输出、提示词、向量检索、词嵌入、工具等。
- LangChain：构成LLM应用程序需要的 链、代理等。
- LangSmith：开发者平台，可以调试、测试、评估和监控 基于任何 LLM 框架上构建的链，并与 LangChain 无缝集成。
- LangServe：用于将 LangChain 的应用部署为 REST API。

这几点我们从官网的架构图清晰可见。

![](https://img.mangoant.top/blog/202406251501832.png)

## 3、LangChainHub简介

LangChain早期推出的各种组件中`LangChainHub`是其中一个比较有意思的项目。

早期LangChainHub对自己的定位如下：LangChainHub 受 Hugging Face Hub 启发，是一个用于发现和提交常用的 提示词、链、代理等的平台。早期，LangChainHub以Prompt集合为起点，然后很快扩展到 链 和 代理。

这个定位我们从之前的LangChainHub在github仓库上的目录可见一斑。

![](https://img.mangoant.top/blog/202406251345441.png)



此时的`LangChainHub` 可以理解为`LangChain` 工具包 或者说 组件中心，里面提供了高质量的组件方便开发者使用。确确实实是一个分享和探索Prompt、链 和Agent的地方。

比如，我们要基于reAct机制实现一个Agent，如果自己写一堆Prompt（提示词）那就太费劲了。此时，在`LangChainHub`就有许多现成可用的Prompt模板，使用简单又省事，所以`LangChainHub`迅速流行开。

后来，`LangChainHub`发生了一些变化，后续再用别的篇幅讲解。

## 4、LangChain安装

LangChain是一个开发LLM的大型生态框架，从各种大模型、到Hugging Face的模型、到各种组件和工具，LangChain是应有尽有，极大地提升了开发者开发LLM应用的效率。

自从有了LangChain，像我这种非专业AI领域的人士也可以进入到AI开发领域，构建出有意思的LLM应用了。

### 4.1、LangChain版本

LangChain目前使用的版本有`0.1.X`、`0.2.X`、`0.3.X`。

2024 年 1 月，LangChain 推出了 `0.1.X`版本，是稳定版，但是由于是早期设计的版本，不太符合AI的生态发展。

于是很快推出了`0.2.X` 版本，此版本做了大量重构，设计上也更符合AI生态的发展。

近期又推出了`0.3.X` 版本，我建议安装`0.2.X`版本。

### 4.2、Python版本

我建议使用的Python版本是`3.10.X`。

### 4.3、安装

```Bash
pip install langchain==0.2.16
```



## 5、模型选择：LLM模型  vs  Chat模型

大语言模型一般分成了 `LLM 模型` 和 `Chat 模型` 两大类。LLM模型属于通用模型，一般用于文本生成。Chat模型更适用于对话场景，它针对对话任务进行了优化。

### 5.1、LLM模型

LLM模型属于通用模型，一般用于文本生成、词嵌入、向量搜索等。代码示例如下：

```Python
from langchain_openai import OpenAI

model = OpenAI(
    model="gpt-3.5-turbo-instruct",
    openai_api_key="sk-xxxxxx",
    openai_api_base="https://api.xiaoai.plus/v1",
)
res = model.invoke("地球到月球的距离是多少？")

```

结果如下：

```Python
'\n\n月球与地球之间的平均距离约为384,400公里。'
```

### 5.2、Chat模型

Chat模型更适用于对话场景，它引入了角色的概念。一般分为3种角色：

1. **system**：系统消息。用于设定上下文背景，在LangChain中用`SystemMessage`表示。
2. **user**：用户消息。用户输入的的内容都属于用户消息，在LangChain中用`HumanMessage`表示。
3. **assistant**：助手消息。AI回答的消息都属于助手消息，在LangChain中用`AIMessage`表示。

代码示例如下，可与看到ChatOpenAI返回的结果里带有AIMessage对象。

```Python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="sk-xxxxxx",
    openai_api_base="https://api.xiaoai.plus/v1",
)
res = model.invoke("地球到月球的距离是多少？")

```

结果如下：

```Python
AIMessage(content='地球到月球的平均距离约为384,400公里。')
```



## 6、LangChain快速入门

### 6.1、LangChain的几个核心组件

LangChain的核心组件有如下6个，学完这些组件，可以快速开发LLM应用。

![](https://secure2.wostatic.cn/static/rGNCuJQGxzqjETQqyZXpze/image.png?auth_key=1728974298-rQrr9Tzmb2XeGhnBXWBFHT-0-7ae8e823794007443eda45946c09275a)

#### 模型 IO

LLM厂商有多家，每家的API的输入和输出有所不同，LLM内部集成了多家LLM的API，并且在上层提供了统一的输入和输出接口，方便开发者切换底层模型。

LangChain还提供了输出解析器，方便开发者从LLM的输出信息中提取需要的信息，并且可以对提取的信息做格式化。

#### 链

链的概念是LangChain的核心，在LangChain中，处理业务逻辑是通过一个个的组件完成的，只要是有输入和输出的组件都可以称之为`链`。再链条中，上一个组件的输出可以作为下一个组件的输入。多个简单的链，就可以组成复杂的链条。

我们可以将复杂的逻辑，通过一个个简单的链串起来，形成一个复杂链条。同时，还可以通过LCEL表达式简化链条的构建过程。

#### 数据检索

大模型对于底模没有的数据会出现幻觉问题，使用RAG检索增强可以解决此问题，并且是比较合适的技术方案。LangChain对RAG检索增强方案做了很好的支持。

LangChain集成了多种加载文档、转换文档、检索文档的统一接口。在做检索增强时，只需要将检索后的文档和问题传给LLM，LLM即可给出回答。

#### 记忆

LLM大模型是没有记忆功能的，本轮对话时，LLM是不知道上轮对话内容的。

所以，为了优化对话体验，则需要给LLM添加记忆。LangChain提供了多种记忆工具，方便我们在开发LLM应用时，给LLM添加记忆。

#### 代理和工具

整个AI领域都在提倡使用Agent（智能体）结合业务场景去发挥出AI的威力。

LangChain对Agent也提供了非常友好的支持，比如实现了ReAct框架（推理和行动的框架，用于增强LLM的推理和行动能力）。同时，还提供了LLM的工具规范和多种现成的工具箱。

在ReAct框架和工具的加持下，开发者可以更高效地开发出Agent。

#### 回调

LangChain内部提供了回调系统，类似于生命周期函数，或者hook钩子，也就是在应用的各个阶段可以执行给定的回调函数。一般会在回调函数里实现自己的业务逻辑。

这种回调函数常用于日志记录、监控、流式传输和其他任务。



### 6.2、入门示例

下面以OpenAI的LLM接口为例，实现简单问答。利用Python的Web框架`FastAPI`开发应用，利用Web容器`uvicorn`部署应用。

#### 申请OpenAI密钥

我习惯使用OpenAI的模型，OpenAI模型的效果更好一些。国内的通义千问模型、月之暗面模型、智谱清言的模型也还可以，读者可以自行尝试。

现在使用OpenAI模型也很方便，无需搭建梯子，无需购买OpenAI账号。只需要通过代理服务的方式，即可使用。比如我常用的代理服务是这家：[https://xiaoai.plus/](https://xiaoai.plus/)。

在代理服务网站上注册后，创建令牌，即可生成密钥，然后使用时，指定代理服务的host即可。如下：

![](https://img.mangoant.top/blog/202410091552339.png)

```Python
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="sk-xxxxxxxxxxxxxxx",
    openai_api_base="https://api.xiaoai.plus/v1",
)
```

为了密钥的安全性，也可以将密钥写到本机环境变量里。比如：

```Python
export OPENAI_API_KEY = "密钥"
或者
os.environ["OPENAI_API_KEY"] = "密钥"

```



#### 代码示例

使用`FastAPI`开发一个LLM的简单应用，代码如下：

```Python
import os
import sys
from fastapi import FastAPI
from langchain_openai import ChatOpenAI

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

app = FastAPI()

@app.post("/chat")
def chat():
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key="sk-xxxxxx",
        openai_api_base="https://api.xiaoai.plus/v1",
    )
    res = model.invoke("地球到月球的距离是多少？")
    return res.content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8990)

```



#### 部署AI服务

使用`uvicorn`方式将服务部署成REST API的形式。

```Bash
uvicorn server:app --host 0.0.0.0 --port 8990
```

部署后，浏览器访问[http://localhost:8990/docs](http://localhost:8990/docs)，执行chat方法，可以拿到问答结果，效果如下：

![](https://img.mangoant.top/blog/202410091456692.png)



## 7、总结

本篇主要聊了LLM开发相关概念，对LangChain做了简单介绍，同时也演示了一个LangChain的小示例。下一篇开始逐个使用LangChain的核心功能。

