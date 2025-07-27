本篇主要聊聊LangChain里`链`的概念，以及LangChain的`LECL`表达式用法。

## 1、链

LangChain几乎是LLM应用开发的第一选择，它的野心也比较大，它致力于将自己打造成LLM应用开发的最大社区。而LangChain最核心的部分非 **Chain** 莫属。那**Chain**到底是个啥，概念比较模糊，像雾像雨又像风。

### 1.1、Chain是核心

在LangChain里只要实现了`Runnable`接口，并且有`invoke`方法，都可以成为`链`。实现了`Runnable`接口的类，可以拿上一个链的输出作为自己的输入。

![](https://img.mangoant.top/blog/202405262304395.png)

比如`ChatPromptTemplate` 、`ChatOpenAI` 、`PydanticOutputParser`等，都实现了`Runnable`接口，且都有`invoke`方法。

LangChain的Chain到底是什么？一句话总结：***Chain是指对 LangChain 多个组件的一系列调用。***说白了就是，将多个组件转起来就是链。当然啦，这个组件得有输入和输出，上一个组件的输出可以作为下一个组件的输入。

再看看官网的解释：***Chain是指调用的序列 - 无论是调用 LLM、工具还是数据预处理步骤，主要支持的方法是使用 LCEL。***

官网里还提到了LCEL，***LCEL是LangChain 表达式语言，是一种更加高效简介的链接 LangChain 组件的方式，也是官网推荐的方式。***本文第二部分继续介绍LCEL。

从下图官网的描述，也可以看到，Chain可以是从最简单的“prompt + LLM”链 到 最复杂的链（运行了包含 100 多个步骤的链）。

![](https://img.mangoant.top/blog/202406152238501.png)



![](https://img.mangoant.top/blog/202406152240311.png)



### 1.2、为什么需要Chain

我们所期待的LLM是能处理许多复杂任务，而非简单的一问一答，也不是简单的处理单一任务。所以，最终我期待的LLM处理任务的流程应该是这样，它中间的复杂过程对用户来说是一个黑盒：

![](https://img.mangoant.top/blog/202406161031962.png)

既然定位是完成复杂任务，那自然就需要通过某个机制将多个单一任务串起来，形成一个大的链条，多个步骤共同完成某个复杂任务。

***Chain可以将多个步骤连接到一起，最终完成各种复杂繁琐的任务。***这就是Chain存在的必要性了。我很喜欢LangChain的Logo，很形象地表达了这一思想。

![](https://img.mangoant.top/blog/202410131014013.png)

Chain需要串联一系列的组件，才能完成复杂任务。当然，我们也可以把 Chain 看作是流水线。通过使用 Chain，你可以将各个步骤定义为独立的模块，然后按顺序串联起来。这样不仅大大简化了代码逻辑，也使得整个流程更加直观和易于管理。

而LCEL的存在，也只是为了让构建链的过程更简单，让链的表达力更清晰更简单。

接下来，我将通过一个示例展示`没有 Chain` 和`有Chain`的2种实现方式，以便更清晰地理解 Chain 的价值。

### 1.3、如果没有Chain

这里举个例子，比如：我们给LLM输入一段**项目描述**，让LLM给这个项目起一个**名称**和**Slogan**。

如果不使用Chain的话，我们可以这样实现。

```Python
#本次需求：我们给LLM输入一段项目描述，让LLM给这个项目起一个名称和Slogan
#以下是实现：

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

proj_desc = """
    我们本次的项目是去森林里探险救援，我们有一个10人小队，
    我们要到达一个叫做“蝴蝶谷”的目的地，去那里解救一位被困的科学家。
    这期间我们可能会遇到许多危险，我们需要共同合作，互相帮助，历经磨难，才能到达目的地。
    我们的任务是要在5天内到达目的地并且救出探险家，才算完成这次探险，否则任务失败，我们将受到惩罚。
    出发前我们要各自准备好自己的装备和干粮，加油！
"""

def name_slogan_by_desc(project_desc):
    """
    根据项目描述，生成项目名称和slogan
    """
    str_parser = StrOutputParser()

    promt_template_project_name = "请你根据<desc>标签里的关于某个项目的描述，生成一个项目名称，只需要返回项目名称。<desc>{project_desc}</desc>"
    promt_project_name = PromptTemplate.from_template(promt_template_project_name)
    final_promt_project_name = promt_project_name.invoke({"project_desc": project_desc})
    res_project_name = model.invoke(final_promt_project_name)
    parsed_res_project_name = str_parser.invoke(res_project_name)

    promt_template_slogan = "请你根据<desc>标签里的关于某个项目的描述，和这个项目的名称{project_name}，给这个项目起一个slogan，slogan要求干脆简洁积极向上，只返回slogan。<desc>{project_desc}</desc>"
    promt_slogan = PromptTemplate.from_template(promt_template_slogan)
    final_promt_slogan = promt_slogan.invoke(
        {"project_desc": project_desc, "project_name": parsed_res_project_name}
    )
    response_slogan = model.invoke(final_promt_slogan)
    parsed_response_slogan = str_parser.invoke(response_slogan)

    final_result = {
        "project_name": parsed_res_project_name,
        "slogan": parsed_response_slogan,
    }
    return final_result

#输入项目描述，输出项目名称和slogan
result = name_slogan_by_desc(proj_desc)
print(result)

```

执行结果如下：

```JSON
{'project_name': '蝴蝶谷救援行动', 'slogan': '拯救科学家，共同合作，蝴蝶谷等你来！'}
```

可以看到，实现过程比较繁琐，变量和代码也多，不够直观，很容易出错。这还只是简单场景，如果碰到复杂场景就更麻烦了。



### 1.4、如果使用了Chain

接下来，我们使用 LangChain 的 Chain 功能，来实现相同的功能。代码如下：

```Python
#本次需求：我们给LLM输入一段项目描述，让LLM给这个项目起一个名称和Slogan
#以下是实现：

from operator import itemgetter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SequentialChain

proj_desc = """
    我们本次的项目是去森林里探险救援，我们有一个10人小队，
    我们要到达一个叫做“蝴蝶谷”的目的地，去那里解救一位被困的科学家。
    这期间我们可能会遇到许多危险，我们需要共同合作，互相帮助，历经磨难，才能到达目的地。
    我们的任务是要在5天内到达目的地并且救出探险家，才算完成这次探险，否则任务失败，我们将受到惩罚。
    出发前我们要各自准备好自己的装备和干粮，加油！
"""

def name_slogan_by_desc(project_desc):
    """
    根据项目描述，生成项目名称和slogan
    """

    #第1条链
    promt_template_project_name = "请你根据<desc>标签里的关于某个项目的描述，生成一个项目名称，只需要返回项目名称。<desc>{project_desc}</desc>"
    chain_one = LLMChain(
        llm=model,
        prompt=PromptTemplate.from_template(promt_template_project_name),
        output_parser=StrOutputParser(),
        output_key="project_name",
    )

    #第2条链
    promt_template_slogan = "请你根据<desc>标签里的关于某个项目的描述，和这个项目的名称{project_name}，给这个项目起一个slogan，slogan要求干脆简洁积极向上，只返回slogan。<desc>{project_desc}</desc>"
    chain_two = LLMChain(
        llm=model,
        prompt=PromptTemplate.from_template(promt_template_slogan),
        output_parser=StrOutputParser(),
        output_key="slogan",
    )

    #串联两条链
    sequential_chain = SequentialChain(
        chains=[chain_one, chain_two],
        input_variables=["project_desc"],
        output_variables=["project_name", "slogan"],
    )
    final_res = sequential_chain(project_desc)

    final_result = {
        "project_name": final_res["project_name"],
        "slogan": final_res["slogan"],
    }
    return final_result

#输入项目描述，输出项目名称和slogan
result = name_slogan_by_desc(proj_desc)
print(result)
```

执行结果如下：

```JSON
{'project_name': '蝴蝶谷救援行动', 'slogan': '团结合作，共赴蝴蝶谷'}
```

可以看到代码更简洁，也很直观。接下来继续使用LCEL让整个链条的表达更加清晰简洁。



## 2、LCEL

### 2.1、LCEL是啥

LCEL是LangChain 表达式语言（LangChain Expression Language）的简称。使用LCEL可以快速将各种`链`组合到一起。

LCEL提供了多种方式将链组合起来，比如使用`管道符` `|`，这种方式既方便书写，表达力也很强劲。下图就是使用LCEL表达式的示例：

![](https://img.mangoant.top/blog/202405262259479.png)

### 2.2、使用区别

**不使用LCEL**

不使用LCEL时，代码写起来是，各种`invoke`满天飞，而且整个执行的流水线流程也不清晰。比如这样：

```Python
final_prompt = prompt.invoke({"book_introduction": book_introduction,
                              "parser_instructions": output_parser.get_format_instructions()})
response = model.invoke(final_prompt)
result = output_parser.invoke(response)
```

**使用LCEL**

使用LCEL时，代码简洁，并且表达力强许多，比如这样：

```Python
chain = prompt | model | output_parser
ret = chain.invoke({"book_introduction": book_introduction,
                    "parser_instructions": output_parser.get_format_instructions()})
```



### 2.3、LCEL表达式的复杂示例

继续使用上面的例子，这次使用LCEL方式来实现：

```Python
#本次需求：我们给LLM输入一段项目描述，让LLM给这个项目起一个名称和Slogan
#以下是实现：

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

proj_desc = """
    我们本次的项目是去森林里探险救援，我们有一个10人小队，
    我们要到达一个叫做“蝴蝶谷”的目的地，去那里解救一位被困的科学家。
    这期间我们可能会遇到许多危险，我们需要共同合作，互相帮助，历经磨难，才能到达目的地。
    我们的任务是要在5天内到达目的地并且救出探险家，才算完成这次探险，否则任务失败，我们将受到惩罚。
    出发前我们要各自准备好自己的装备和干粮，加油！
"""

def name_slogan_by_desc(project_desc):
    """
    根据项目描述，生成项目名称和slogan
    """

    #第1条链
    promt_template_project_name = "请你根据<desc>标签里的关于某个项目的描述，生成一个项目名称，只需要返回项目名称。<desc>{project_desc}</desc>"
    chain_one = (
        PromptTemplate.from_template(promt_template_project_name)
        | model
        | {"project_name": StrOutputParser(), "project_desc": lambda x: project_desc}
    )

    #第2条链
    promt_template_slogan = "请你根据<desc>标签里的关于某个项目的描述，和这个项目的名称{project_name}，给这个项目起一个slogan，slogan要求干脆简洁积极向上，只返回slogan。<desc>{project_desc}</desc>"
    chain_two = (
        PromptTemplate.from_template(promt_template_slogan)
        | model
        | {"slogan": StrOutputParser(), "project_info": lambda x: chain_one}
    )

    #串联两条链
    final_chain = chain_one | chain_two
    final_res = final_chain.invoke({"project_desc": project_desc})

    final_result = {
        "project_name": final_res["project_info"]["project_name"],
        "slogan": final_res["slogan"],
    }

    return final_result

#输入项目描述，输出项目名称和slogan
result = name_slogan_by_desc(proj_desc)
print(result)
```

普通方式和LCEL方式的核心代码对比：

- **普通方式**：

![](https://img.mangoant.top/blog/202406161848187.png)

- **LCEL方式**：

![](https://img.mangoant.top/blog/202406161849832.png)



### 2.4、LCEL的原理

***LangChain的核心是Chain，即对多个组件的一系列调用。***

***LCEL是LangChain 定义的表达式语言，是一种更加高效简洁的调用一系列组件的方式。***

***LCEL***使用方式就是：以一堆`管道符（"|"）`串联所有实现了`Runnable接口`的组件。

比如这样：

```Python
prompt_tpl = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出{cityName}的{viewPointNum}个著名景点。"),
    ]
)

output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()

model = ChatOpenAI(model="gpt-3.5-turbo")

chain = prompt_tpl | model | output_parser

response = chain.invoke(
    {"cityName": "南京", "viewPointNum": 3, "parser_instructions": parser_instructions}
)

```

以上是概念和使用方式，那LangChain的原理是什么呢？

LangChain为了让组件能以LCEL的方式快速简洁的被调用，计划将所有组件都实现Runnable接口。比如我们常用的`PromptTemplate` 、`LLMChain` 、`StructuredOutputParser` 等等。

`管道符（"|"）`在Python里就类似`or`运算（或运算），比如`A|B`，就是`A.or(B)`。

那对应到***LangChain***的Runnable接口里，这个`or`运算是怎么实现的呢？一起看到源码：

![](https://img.mangoant.top/blog/202407291716086.png)

LangChain通过`or`将所有的Runnable串联起来，在通过`invoke`去一个个执行，上一个组件的输出，作为下一个组件的输入。

![](https://img.mangoant.top/blog/202408011937155.png)

LangChain这风格怎么有点像神经网络呀，不得不说，这个世界到处都是相似的草台班子。嗨！

总结起来讲就是：LangChain的每个组件都实现了Runnable，通过LCEL方式，将多个组件串联到一起，最后一个个执行每个组件的invoke方法。上一个组件的输出是下一个组件的输入。

![](https://img.mangoant.top/blog/202406161031962.png)

### 2.5、Runnable的含义和应用场景

在使用LCEL表达式时，会碰到一些没见过的用法。比如：RunnablePassthrough、RunnableParallel、RunnableBranch、RunnableLambda。它们又是什么意思？什么场景下用呢？

#### 2.5.1、RunnablePassthrough

**定义**

`RunnablePassthrough` 主要用在链中传递数据。`RunnablePassthrough`一般用在链的第一个位置，用于接收用户的输入。如果处在中间位置，则用于接收上一步的输出。

**应用场景**

比如，依旧使用上面的例子，接受用户输入的城市，如果输入城市是南京，则替换成北京，其余不变。代码如下。此处的`{}`和`RunnablePassthrough.assign()`是同一个语义。

```Python
chain = (
    {
        "cityName": lambda x: '北京' if x["cityName"] == '南京' else x["cityName"],
        "viewPointNum": lambda x: x["viewPointNum"],
        "parser_instructions": lambda x: x["parser_instructions"],
    }
    | prompt_tpl
    | model
    | output_parser
)
```



#### 2.5.2、RunnableParallel

**定义**

`RunnableParallel`看名字里的`Parallel`就猜到一二，用于并行执行多个组件。通过`RunnableParallel`，可以实现部分组件或所有组件并发执行的需求。

**应用场景**

比如，同时要执行两个任务，一个列出城市著名景点，一个列出城市著名书籍。

```Python
prompt_tpl_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出{cityName}的{viewPointNum}个著名景点。"),
    ]
)
prompt_tpl_2 = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出关于{cityName}历史的{viewPointNum}个著名书籍。"),
    ]
)

output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()

model = ChatOpenAI(model="gpt-3.5-turbo")

chain_1 = prompt_tpl_1 | model | output_parser
chain_2 = prompt_tpl_2 | model | output_parser
chain_parallel = RunnableParallel(view_point=chain_1, book=chain_2)

response = chain_parallel.invoke(
    {"cityName": "南京", "viewPointNum": 3, "parser_instructions": parser_instructions}
)

```



#### 2.5.3、RunnableLambda

**定义**

要说牛批还得是`RunnableLambda`，它可以将Python 函数转换为 `Runnable`对象。这种转换使得任何函数都可以被看作 LCEL 链的一部分，我们把自己需要的功能通过自定义函数 + `RunnableLambda`的方式包装一下，集成到 LCEL 链中，这样算是***可以跟任何外部系统打通***了。

**应用场景**

比如，在执行过程中，想在中间插入一段自定义功能（如 打印日志 等），可以通过自定义函数 + RunnableLambda的方式实现。

```Python
def print_info(info: str):
    print(f"info: {info}")
    return info

prompt_tpl_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出{cityName}的{viewPointNum}个著名景点。"),
    ]
)

output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()

model = ChatOpenAI(model="gpt-3.5-turbo")

chain_1 = prompt_tpl_1 | model | RunnableLambda(print_info) | output_parser


response = chain_1.invoke(
    {"cityName": "南京", "viewPointNum": 3, "parser_instructions": parser_instructions}
)
```



#### 2.5.4、RunnableBranch

**定义**

`RunnableBranch`主要用于多分支子链的场景，为链的调用提供了路由功能，这个有点类似于LangChain的`路由链`。我们可以创建多个子链，然后根据条件选择执行某一个子链。

**应用场景**

比如，有多个回答问题的链，先根据问题找到分类，然后在使用具体的链回答问题。

```Python
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

#准备2条目的链：一条物理链，一条数学链
#1. 物理链
physics_template = """
你是一位物理学家，擅长回答物理相关的问题，当你不知道问题的答案时，你就回答不知道。
具体问题如下：
{input}
"""
physics_chain = PromptTemplate.from_template(physics_template) | model | output_parser

#2. 数学链
math_template = """
你是一个数学家，擅长回答数学相关的问题，当你不知道问题的答案时，你就回答不知道。
具体问题如下：
{input}
"""
math_chain = PromptTemplate.from_template(math_template) | model | output_parser

#4. 其他链
other_template = """
你是一个AI助手，你会回答一下问题。
具体问题如下：
{input}
"""
other_chain = PromptTemplate.from_template(other_template) | model | output_parser


classify_prompt_template = """
请你对以下问题进行分类，将问题分类为"数学"、"物理"、"其它"，不需要返回多个分类，返回一个即可。
具体问题如下：
{input}

分类结果：
"""
classify_chain = PromptTemplate.from_template(classify_prompt_template) | model | output_parser

answer_chain = RunnableBranch(
    (lambda x: "数学" in x["topic"], math_chain),
    (lambda x: "物理" in x["topic"], physics_chain),
    other_chain
)

final_chain =  {"topic": classify_chain, "input": itemgetter("input")} | RunnableLambda(print_info) | answer_chain
#final_chain.invoke({"input":"地球的半径是多少？"})
final_chain.invoke({"input":"对y=x求导的结果是多少？"})

```



下面介绍2个常用的链：路由链、转换链，还有其余的链，读者可以自行尝试。

## 3、路由链

### 3.1、路由链概念

**路由链（RouterChain）**是由LLM根据输入的Prompt去选择具体的某个链。路由链中一般会存在多个Prompt，Prompt结合LLM决定下一步选择哪个链。

![](https://img.mangoant.top/blog/202406202304175.png)

### 3.2、路由链的使用场景

路由链一般涉及到2个核心类，`LLMRouterChain`和`MultiPromptChain`，一起看看官网介绍：

![](https://img.mangoant.top/blog/202406200900004.png)

- **LLMRouterChain**：使用LLM路由到可能的选项中。
- **MultiPromptChain**：该链可用于在多个提示词之间路由输入，当你有多个提示词并且只想路由到其中一个时，可以用这个链。

一般使用路由链时，有固定的几个步骤：

1. 准备多个链的Prompt提示词，然后各自封装成链。
2. 将可能路由到的链封装到destination_chains里。
3. 构建多提示词和RouterChain ，负责选择下一个要调用的链。
4. 构建默认链。
5. 使用MultiPromptChain选择某个链，然后再去执行此链。

### 3.3、使用路由链的案例

假设我们有一个常见的场景，根据用户的输入内容选择不同的处理路径，如果没有选到合适的链，则使用默认链。比如：根据用户的输入问题，选择不同的链去处理，如果没选到合适的，则走默认链。

具体代码如下：

```Python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key="sk-xxxx",
    openai_api_base="https://api.302.ai/v1",
)


from langchain.chains.router import LLMRouterChain, MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate

#准备2条目的链：一条物理链，一条数学链
#1. 物理链
physics_template = """
你是一位物理学家，擅长回答物理相关的问题，当你不知道问题的答案时，你就回答不知道。
具体问题如下：
{input}
"""
physics_prompt = PromptTemplate.from_template(physics_template)
physics_chain = LLMChain(llm=model, prompt=physics_prompt)

#2. 数学链
math_template = """
你是一个数学家，擅长回答数学相关的问题，当你不知道问题的答案时，你就回答不知道。
具体问题如下：
{input}
"""
math_prompt = PromptTemplate.from_template(math_template)
math_chain = LLMChain(llm=model, prompt=math_prompt)

#3. 英语链
english_template = """
你是一个非常厉害的英语老师，擅长回答英语相关的问题，当你不知道问题的答案时，你就回答不知道。
具体问题如下：
{input}
"""
english_prompt = PromptTemplate.from_template(english_template)
english_chain = LLMChain(llm=model, prompt=english_prompt)


#所有可能的目的链
destination_chains = {}
destination_chains["physics"] = physics_chain
destination_chains["math"] = math_chain
destination_chains["english"] = english_chain


#默认链
default_chain = ConversationChain(llm=model, output_key="text")

#让多路由模板 能找到合适的 提示词模板
destinations_template_str = """
physics:擅长回答物理问题
math:擅长回答数学问题
english:擅长回答英语问题
"""
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_template_str
)

#通过路由提示词模板，构建路由提示词
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

#路由链
router_chain = LLMRouterChain.from_llm(llm=model, prompt=router_prompt)

#最终的链
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

#multi_prompt_chain.invoke({"input": "重力加速度是多少？"})
#multi_prompt_chain.invoke("y=x^2+2x+1的导数是多少？")
multi_prompt_chain.invoke("将以下英文翻译成中文，只输出中文翻译结果：\n The largest community building the future of LLM apps.")
#multi_prompt_chain.invoke("你是怎么理解java的面向对象的思想的？")

```

执行结果跟我们预想的一致，执行结果如下：

![](https://img.mangoant.top/blog/202406201040966.png)

![](https://img.mangoant.top/blog/202406201040824.png)

![](https://img.mangoant.top/blog/202406201041851.png)



## 4、转换链

### 4.1、转换链的概念

在开发AI Agent（智能体）时，我们经常需要对输入数据进行预处理，这样可以更好地利用LLM。LangChain提供了一个强大的工具——转换链（`TransformChain`），它可以帮我们轻松实现这一任务。

**转换链（TransformChain）**主要是将  给定的数据  按照某个`函数`进行转换，再将  转换后的结果  输出给LLM。 所以**转换链**的核心是：根据业务逻辑编写合适的转换函数。

其实，转换链的设计也很精妙，从源码可以看出，它只是做了一条链，然后具体的任务完全丢给了外部的函数来实现。在LangChain里只要是链，就可以随处链接。

![](https://img.mangoant.top/blog/202406231708340.png)

### 4.2、转换链的使用场景

转换链只有1个核心类，`TransformChain`。

有时，我们在将数据发送给LLM之前，希望对其做一些操作时（比如替换一些字符串、截取部分文本等等），就会用到`转换链`。`TransformChain` 在 NLP 中很重要，有些场景还很实用。

一般使用`转换链`有几个固定步骤：

1. 根据需求定义转换函数`transform_func`，入参和出参都是字典。
2. 实例化转换链`TransformChain`。
3. 因为转换链只能做内容转换的事情，后续的操作还需要LLM介入，所以需要实例化`LLMChain`。
4. 最终通过顺序连`SimpleSequentialChain`将`TransformChain`和`LLMChain`串起来完成任务。



### 4.3、使用转换链的案例

比如，给定LLM一篇很长的文章，但是我只想让LLM帮我总结文章前3自然段的内容，同时，总结之前，我还需要将自然段里的 部分字段 替换成 给定字段。

具体代码如下：

```Python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, TransformChain, SimpleSequentialChain
from langchain_openai import OpenAI, ChatOpenAI

file_content = ""
with open("./file_data.txt", "r") as file:
    file_content = file.read()


#定义转换函数，截取文章前8段，再替换部分字符串
def transform_func(data):
    text = data["input_text"]
    shortened_text = "\n".join(text.split("\n")[:7])
    transform_shortened_text: str = shortened_text.replace(
        "PVC", "PersistentVolumeClaim"
    ).replace("PV", "PersistentVolume")
    return {"output_text": transform_shortened_text}


#定义转换链
transform_chain = TransformChain(
    input_variables=["input_text"],
    output_variables=["output_text"],
    transform=transform_func,
)

#定义LLM
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key="sk-xxxxxx",
    openai_api_base="https://api.302.ai/v1",
)

#定义提示词模板 和 LLM链
prompt_template = """
请你对下面的文字进行总结:
{output_text}

总结:
"""

prompt = PromptTemplate(input_variables=["output_text"], template=prompt_template)
llm_chain = LLMChain(
    llm=model,
    prompt=prompt,
)


#使用顺序链连接起来
final_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])
res = final_chain.run(file_content)
print(res)
```

代码执行结果符合预期。总结的结果很精通，同时也是按照给定的字符串返回的。

# 5、总结

本文主要聊了 LangChain 中的 Chain 概念。Chain 是 LangChain 中的核心组件，我们对多个组件的一系列调用就是Chain。***使用Chain可以让构建复杂的任务，更加清晰简洁。***

还聊了LangChain的LCEL表达式，以及原理，以及常用的几个Runnable的定义和应用场景。另外还介绍了LangChain中的**路由链（RouterChain）和 转换链（TransformChain）**的使用。

