以ChatGPT为代表的AI应用迅速火遍全球，成为打工人的常用工具。紧接着，多模态、AI Agent等各种高大尚的名词也逐渐进入大众视野，吸引了大量关注。那么，到底什么是AI Agent？AI Agent是如何工作和使用工具的呢？

## 1、什么是AI Agent

### 1.1、一句话总结：什么是AI Agent

AI Agent，经常被翻译为：智能体或代理。一句话总结，AI Agent就是一个有着`聪明大脑`而且能够`感知外部环境`并`采取行动`的**智能系统**。

我们可以把它想象成一个能思考和行动的人，而大型语言模型（LLM）就是这个人的“大脑”。通过这个大脑，再加上一些能够感知外部世界和执行任务的部件，AI Agent就变成了一个有“智慧”的机器人。

要让AI Agent充分利用它的“大脑”和各种组件，需要一种协调机制。ReAct机制就是常用的协调机制。通过ReAct机制，AI Agent能够结合外部环境和行动组件，完成复杂的任务。

为什么我们需要AI Agent呢？其实说到底是因为**单一的模型对我们来说作用不大**，我们需要的是一个具备智能的复杂系统。只有复杂系统才能真正的应用到实际生产工作中。

![](https://img.mangoant.top/blog/202406032326549.png)

### 1.2、从单一模型到复合AI系统

要理解AI Agent，我们先看看AI领域的一些变化。

以前的AI系统通常是单一模型，受训练数据的限制，只能解决有限的任务或者固定领域的任务，难以适应新的情况。

而现在，我们有了LLM通用大模型，训练的数据更多，能完成更多领域的任务，比如内容生成、文生图、文生视频等等。同时还可以把大模型和各种外部组件结合起来，构建复合AI系统，这样就能解决更复杂的问题。

举个例子，如果直接让单一模型帮我制定一个去三亚的旅游计划，它无法做到。如果让LLM大模型帮我制定一个去三亚的旅游计划，它可以制定一个鸡肋的计划，几乎不可用，因为它不知道我的个人信息、也不知道航班信息，也不知道天气情况。

但如果我们设计一个复杂AI系统，让系统里的LLM大模型能够通过工具能访问我的个人信息，访问互联网上的天气情况，访问航班信息，再结合航班系统的开放接口，就可以自动帮我预定机票，自动制定行程规划了。

这就是复合AI系统的魅力，它能够结合**工具、记忆、其余各种组件** 来解决复杂问题。

### 1.3、复合AI系统的模块化

复合AI系统是模块化的，就像拼积木一样。你可以选择不同的模型和组件，然后把它们组合在一起，解决不同的问题。

比如，你可以用一个模型来生成文本，用另一个模型来处理图像，还可以用一些编写的程序代码，一起构建出复杂AI系统。

### 1.4、AI Agent的推理与行动能力

AI Agent的核心是让LLM大模型 **掌控** 复杂AI系统的逻辑，说白了就是让LLM主导AI Agent的思维过程。我们向LLM输入复杂问题，它可以将复杂问题分解并一步步的制定解决方案。

这与设计一个程序系统不同，在AI系统里，LLM大模型会一步一步的思考、制定一步一步的计划，然后一个一个的去解决。并不是按照某个指定程序去执行的。

AI Agent的组件包括：大模型的推理能力、行动能力 和 记忆能力。

- 大模型的推理能力是 解决问题的核心。
- 行动能力通过工具（外部程序）实现，模型可以定义何时调用它们以及如何调用它们。工具可以是搜索引擎、计算器、操作数据库等。
- 记忆能力使大模型能够存储内部日志和对话历史，从而使体验更加个性化。记忆可以帮助大模型在解决复杂问题时保持上下文连贯。

## 2、AI Agent的推理能力是如何工作的

我们使用的AI助手，一般是经过了预训练和微调这2个步骤，尽管训练出的模型能回答许多通用类问题，但是在遇到复杂问题时还是束手无策。

直到有人提出了**思维链**方式，才解决了模型在面对复杂问题时的推理能力。

### 2.1、什么是思维链

**思维链**（Chain of Thought, CoT）是用于提高AI模型推理能力的方式。其核心原理就8个字：**化繁为简、逐个击破**。

思维链的工作原理是，模拟人类思考问题的过程，通过将复杂的问题逐步分解，然后逐个向前解决这些简单问题，从而得出最终答案。

### 2.2、实现智能体的方式

智能体（AI Agent）用于更加智能 更加强调推理的场景，思维链便是用于AI Agent的场景，在这种场景下可以发挥它的优势。

在AI Agent领域里，常见的实现思维链的机制有2种，**Plan-and-Executor**机制和**ReAct**机制。

#### 2.2.1、Plan-and-Executor机制

**Plan-and-Executor**机制是分离`规划 和 执行`这2个环节。它将问题解决过程分为两个阶段：**规划和执行**。

**规划阶段：**

在这个阶段，主要是在智能体里分析问题，制定一个详细的解决方案计划。这个阶段通常会涉及到大量的计算过程，用来确定出最优的行动计划。规划的结果是：输出一个具体的**行动计划**。

**执行阶段：**

在这个阶段，智能体按照规划阶段生成的行动计划去**逐步执行**每个步骤。并在执行过程中监控和调整，确保计划的顺利执行。

**优点**：

这种机制的特点就是规划和执行的分离，这种分离可以使每个阶段更加专注于当前任务，从而提高效率。适用于需要复杂度较高，需要提前做复杂规划的任务。

**缺点**：

在执行过程中，可能存在不确定因素，这种方式因为是提前规划好的，所以可能不适应变化，需要频繁调整计划。

**举例说明Plan-and-Executor步骤**

比如，我想知道2024年周杰伦最新的演唱会是时间和地点是什么，通过Plan-and-Executor机制，会被拆解成以下步骤：

计划阶段：

```text
1. 在搜索引擎上查找“2024年周杰伦最新演唱会时间和地点”。
2. 查看官方网站或可信的新闻网站的相关信息。
3. 汇总并记录演唱会的时间和地点。
```

执行阶段：

```text
1. 在搜索引擎上查找“2024年周杰伦最新演唱会时间和地点”。
   - 结果：找到了一些相关的网页链接。
2. 查看官方网站或可信的新闻网站的相关信息。
   - 结果：在周杰伦的官方网站上找到了2024年最新演唱会的时间和地点。
3. 汇总并记录演唱会的时间和地点。
   - 结果：2024年周杰伦最新演唱会将在2024年5月20日于北京举行。
```



#### 2.2.2、ReAct机制

**ReAct**机制是一种将推理（`Reasoning`）和行动（`Action`）结合在一起的实现方式，同时还引入了观察（`Observation`）环节，在每次执行（`Action`）之后，都会先观察（`Observation`）当前现状，然后再进行下一步的推理（`Reason`）。

它强调的是在感知环境变化后，立即做出反应并采取行动，而不是先制定一个详细的计划。

**优点**：

适应性强，能够快速响应环境变化。更适合动态和不确定性高的环境。

**缺点**：

由于没有预先规划，可能在复杂任务中效率较低，每一步都在执行：观察、推理、行动。

**举例说明ReAct步骤**

比如，我要知道2024年周杰伦最新的演唱会是时间和地点是什么，通过ReAct机制，会被拆解成以下步骤：

```text
推理1：用户想知道2024年周杰伦最新的演唱会是时间和地点是什么，需要查找最新的信息。
行动1：调用Google的搜索API进行搜索。
观察1：搜索结束，搜索的结果中出现一些关于《2024年周杰伦最新的演唱会》的网页信息。

推理2：搜索出来的网页较多，大概浏览前6个网页的具体内容。
行动2：点击第一个网页，开始浏览。
观察2：浏览结束，浏览的网页内容提及到了2024年周杰伦最新的演唱会信息。

推理3：针对网页的内容进行，问题的总结。
结果：将最终的答案输出给用户。
```



### 2.3、代码示例

#### 2.3.1、Plan-and-Executor机制

LangChain框架已经实现了Plan-and-Executor机制，三行代码即可调用：

**核心代码：**

```Python
#加载计划
planner = load_chat_planner(model)
#加载执行器
executor = load_agent_executor(model, tools, verbose=True)
#加载代理
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
```

**完整代码：**

```Python
from langchain_openai import ChatOpenAI
from langchain_experimental.plan_and_execute import (
    PlanAndExecute, load_agent_executor, load_chat_planner
)
from langchain.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="sk-xxxxxx",
    openai_api_base="https://api.xiaoai.plus/v1",
)

#定义工具
class SumNumberTool(BaseTool):
    name = "数字相加计算工具"
    description = "当你被要求计算2个数字相加时，使用此工具"

    def _run(self, a, b):
        return a["title"] + b["title"]

#加入到工具合集
tools = [SumNumberTool()]

#加载计划
planner = load_chat_planner(model)
#加载执行器
executor = load_agent_executor(model, tools, verbose=True)
#加载代理
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run("你帮我算下 3.941592623412424 + 4.3434532535353的结果")
```

**执行过程：**

执行过程见下图，从过程中，我们可以看出，Agent确实是先规划了N个`steps`，然后一步步执行`step`。

![](https://img.mangoant.top/blog/202408151045798.png)

![](https://img.mangoant.top/blog/202408151045961.png)

#### 2.3.2、ReAct机制

LangChain框架已经实现了ReAct机制，两行代码即可调用：

**核心代码：**

```Python
#使用reAct的提示词
prompt = hub.pull("hwchase17/structured-chat-agent")
#创建Agent
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)
```

**完整代码：**

```Python
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor, tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="sk-xxxxxx",
    openai_api_base="https://api.xiaoai.plus/v1",
)

#定义工具
class SumNumberTool(BaseTool):
    name = "数字相加计算工具"
    description = "当你被要求计算2个数字相加时，使用此工具"

    def _run(self, a, b):
        return a["title"] + b["title"]

#加入到工具合集
tools = [SumNumberTool()]

#使用reAct的提示词
prompt = hub.pull("hwchase17/structured-chat-agent")

#创建Agent
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

#创建记忆组件
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#创建Agent执行器
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)

agent_executor.invoke({"input": "你帮我算下 3.941592623412424 + 4.3434532535353的结果"})
```

**执行过程：**

ReAct机制的执行过程，读者自行尝试。与Plan-and-Executor相比，ReAct机制少了规划`steps`这个环节。

## 3、基于LangChain的ReAct实现AI Agent

当前，在各个大厂纷纷卷LLM的情况下，各自都借助自己的LLM推出了自己的AI Agent，比如字节的Coze，百度的千帆等，还有开源的Dify。

你是否想知道其中的原理？是否想过自己如何实现一套AI Agent？当然，借助LangChain就可以。

### 3.1、示例

比如，在不使用ReAct机制借助外部工具的情况下，让LLM帮我们计算两个小数相加，则直接出错。

![](https://img.mangoant.top/blog/202406032006313.png)

然后，借助ReAct机制，会让LLM自动使用自定义工具，最终计算正确。

![](https://img.mangoant.top/blog/202406032017436.png)

![](https://img.mangoant.top/blog/202406032018634.png)

然后，继续测试，问别的问题，借助ReAct机制，则不会使用到工具，直接给出答案。

![](https://img.mangoant.top/blog/202406032024192.png)

### 3.2、代码

具体代码如下：

```Python
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

#模型
model = ChatOpenAI(model="gpt-3.5-turbo",
                   openai_api_key="sk-XXXXXXXXXX",
                   openai_api_base="https://api.aigc369.com/v1")
#直接让模型计算数字，模型会算错
model.invoke([HumanMessage(content="你帮我算下，3.941592623412424+4.3434532535353的结果")])


#下面开始使用ReAct机制，定义工具，让LLM使用工具做专业的事情。

#定义工具，要继承自LangChain的BaseTool
class SumNumberTool(BaseTool):
    name = "数字相加计算工具"
    description = "当你被要求计算2个数字相加时，使用此工具"

    def _run(self, a, b):
        return a.value + b.value
        
#工具合集
tools = [SumNumberTool()]
#提示词，直接从langchain hub上下载，因为写这个ReAct机制的prompt比较复杂，直接用现成的。
prompt = hub.pull("hwchase17/structured-chat-agent")
#定义AI Agent
agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)
#使用Memory记录上下文
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)
#定义AgentExecutor，必须使用AgentExecutor，才能执行代理定义的工具
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)
#测试使用到工具的场景
agent_executor.invoke({"input": "你帮我算下3.941592623412424+4.3434532535353的结果"})

#测试不使用工具的场景
agent_executor.invoke({"input": "请你充当稿件审核师，帮我看看'''号里的内容有没有错别字，如果有的话帮我纠正下。'''今天班级里的学生和老实要去哪里玩'''"})        


```



## 4、智能体如何使用工具

只有让LLM（大模型）学会使用工具，才能做出一系列实用的AI Agent，才能发挥出LLM真正的实力。本篇，我们让AI Agent使用更多的工具，比如：外部搜索、分析CSV、文生图、执行代码等。

### 4.1、使用工具的必要性

LLM（大模型）如果没有使用工具的能力，那就相当于一个有着聪明大脑 但四肢僵硬的 **渐冻人**，什么事儿也做不了。人类之所以区别于动物，正是因为学会了使用工具。因此，赋予LLM使用工具的能力至关重要。

我们需要 LLM去帮助执行各种任务。而Tool（工具）就是LLM 在执行任务过程中，能够调用的外部能力。比如：需要检索外部资料时，可以调用检索工具；需要执行一段代码时，可以调用自定义函数去执行。

### 4.2、LangChain的Tool规范

所有的工具肯定要遵守一套规范，才能让LLM随意调用。为此，LangChain 抽象出一个`Tool 层`，只要是遵守这套规范的函数就是 `Tool` 对象，就可以被 LLM调用。

![](https://img.mangoant.top/blog/202406290731770.png)

#### 4.2.1、Tool规范

Tool的规范也简单，只要有三个属性就行：`name`、`description`和`function`。

- name：工具的名称。
- description：对工具的功能描述，后续这个描述文本会添加到Prompt（提示词）中，LLM 将根据description来决定是否调用该工具。
- function：此工具实际运行的函数。 

只要遵守这个规范就行，使用形式可以有多种，下文的实践代码会介绍到。

#### 4.2.2、Agent使用工具的流程

让AI Agent使用工具，需要定义`Agent`和`AgentExecutor`。`AgentExecutor`维护了`Tool.name`到`Tool`的`Map` 结构。

LLM根据Prompt（包含了`Tool`的描述） 和  用户的问题，判断是否需要调用工具，确定某个工具后，在根据`Tool`的名称 和 调用参数，到映射`Map` 中获找`Tool`实例，找到之后调用`Tool`实例的`function`。 

### 4.3、如何使用各种Tool

**自定义Tool**只需要遵守以上规范就可以，下面以几个常用的工具做示例。

下文有些工具用到了`toolkits`。`toolkits`是**LangChain提供的工具包，旨在简化使用工具的成本**，`toolkits`里提供了丰富的工具，还在不断叠加，大部分的工具都可以在里面找到。

#### 4.3.1、外部搜索

使用外部搜索工具。本文使用的是`serpapi`，`serpapi`集成了Google、百度等多家搜索引擎，通过api的形式调用，非常方便。

官网地址：[https://serpapi.com/](https://serpapi.com/)。可以自行注册，有一些免费额度。外部搜索工具定义如下：

```Python
#1. 使用@tool装饰器，定义搜索工具
@tool
def search(query: str) -> str:
    """只有在需要了解实时信息 或 不知道的事情的时候 才会使用这个工具，需要传入要搜索的内容。"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    return result
```

#### 4.3.2、文生图

文生图工具是使用LangChain社区提供的`DallEAPIWrapper`类，本文使用OpenAI的图片生成模型`Dall-E-3`，具体代码如下：

```Python
#2. 使用Tool工具类，定义图片生成工具
dalle_image_generator = Tool(
    name="基于OpenAI Dall-E-3的图片生成器",
    func=DallEAPIWrapper(model="dall-e-3").run,
    description="OpenAI DALL-E API 的包装器。当你需要根据 描述的文本 生成图像时 使用此工具，需要传入 对于图像的描述。",
)
```

这里的`DallEAPIWrapper(model="dall-e-3").run`方法就是个函数，实际是去调用了OpenAI的接口。

![](https://img.mangoant.top/blog/202406290739397.png)

#### 4.3.3、代码执行器

代码执行器工具，可以执行代码 或者 根据自然语言生成代码。主要使用LangChain提供的`PythonREPLTool` 和 LangChain提供的`toolkits`。

比如`create_python_agent`就简化了创建Python解释器工具的过程。代码如下：

```Python
#3.使用toolkit，定义执行Python代码工具
python_agent_executor = create_python_agent(
    llm=model,
    tool=PythonREPLTool(),
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
)
```

#### 4.3.4、分析CSV

CSV工具，用来分析csv文件。依旧是使用`toolkits`工具包里的`create_csv_agent`函数快出创建工具。代码如下：

```Bash
#4. 使用toolkit，定义分析CSV文件工具
csv_agent_executor = create_csv_agent(
    llm=model,
    path="course_price.csv",
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
    allow_dangerous_code=True,
)
```

#### 4.3.5、完整代码

上面介绍了AI Agent的常用工具，定义好工具之后，在把工具放入到工具集中，最后在定义Agent 和 AgentExecutor就算完成了。短短几十行代码，就可以让LLM使用这么多工具了。

完整代码如下：

```Python
import os
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_structured_chat_agent, AgentExecutor, Tool
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_experimental.agents.agent_toolkits import (
    create_python_agent,
    create_csv_agent,
)
from langchain_community.utilities import SerpAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

#需要先安装serpapi, pip install serpapi, 还需要到 https://serpapi.com/ 去注册账号

#SERPAPI_API_KEY 和 OPENAI 相关密钥，注册到环境变量
os.environ["SERPAPI_API_KEY"] = (
    "9dd2b2ee429ed996c75c1daf7412df16336axxxxxxxxxxxxxxx"
)
os.environ["OPENAI_API_KEY"] = "sk-a3rrW46OOxLBv9hdfQPBKFZtY7xxxxxxxxxxxxxxxx"
os.environ["OPENAI_API_BASE"] = "https://api.302.ai/v1"

model = ChatOpenAI(model_name="gpt-3.5-turbo")


#基于reAct机制的Prompt模板
prompt = hub.pull("hwchase17/structured-chat-agent")



#各种方式定义工具

#1. 使用@tool装饰器，定义搜索工具
@tool
def search(query: str) -> str:
    """只有在需要了解实时信息 或 不知道的事情的时候 才会使用这个工具，需要传入要搜索的内容。"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    return result


#2. 使用Tool工具类，定义图片生成工具
dalle_image_generator = Tool(
    name="基于OpenAI Dall-E-3的图片生成器",
    func=DallEAPIWrapper(model="dall-e-3").run,
    description="OpenAI DALL-E API 的包装器。当你需要根据 描述的文本 生成图像时 使用此工具，需要传入 对于图像的描述。",
)

#3. 使用toolkit，定义执行Python代码工具
python_agent_executor = create_python_agent(
    llm=model,
    tool=PythonREPLTool(),
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
)

#4. 使用toolkit，定义分析CSV文件工具
csv_agent_executor = create_csv_agent(
    llm=model,
    path="course_price.csv",
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
    allow_dangerous_code=True,
)

#定义工具集合
tool_list = [
    search,
    dalle_image_generator,
    Tool(
        name="Python代码工具",
        description="""
        当你需要借助Python解释器时，使用这个工具。
        比如当你需要执行python代码时，
        或者，当你想根据自然语言的描述生成对应的代码时，让它生成Python代码，并返回代码执行的结果。
        """,
        func=python_agent_executor.invoke,
    ),
    Tool(
        name="CSV分析工具",
        description="""
        当你需要回答有关course_price.csv文件的问题时，使用这个工具。
        它接受完整的问题作为输入，在使用Pandas库计算后，返回答案。
        """,
        func=csv_agent_executor.invoke,
    ),
]


#将工具丢给Agent
agent = create_structured_chat_agent(
    llm=model,
    tools=tool_list,
    prompt=prompt
)

#定义AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tool_list, 
    verbose=True, # 打印详细的 选择工具的过程 和 reAct的分析过程
    handle_parsing_errors=True
)


#不会使用工具
agent_executor.invoke({"input": "你是谁？"})

```

一起看下使用工具后，reAct的整个过程。

![](https://img.mangoant.top/blog/202406281831022.png)

# 5、总结

本文主要聊了AI Agent的概念、思维链的概念以及LangChain中如何实现AI Agent和AI Agent如何使用工具。AI Agent只有借助工具才能发挥威力。

