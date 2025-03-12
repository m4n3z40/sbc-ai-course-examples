import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { tool } from "@langchain/core/tools";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { ChatOpenAI } from "@langchain/openai";
import * as d3 from "d3";
import { createCanvas } from "canvas";
import { z } from "zod";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

/**
 * Create an agent that can run a set of tools.
 * @param {Object} options
 * @param {ChatOpenAI} options.llm - The language model to use.
 * @param {Array} options.tools - The tools to use.
 * @param {string} options.systemMessage - The system message to display.
 */
async function createAgent({ llm, tools, systemMessage }) {
  const toolNames = tools.map((tool) => tool.name).join(", ");
  const formattedTools = tools.map((t) => convertToOpenAITool(t));

  let prompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      "You are a helpful AI assistant, collaborating with other assistants." +
      " Use the provided tools to progress towards answering the question." +
      " If you are unable to fully answer, that's OK, another assistant with different tools " +
      " will help where you left off. Execute what you can to make progress." +
      " If you or any of the other assistants have the final answer or deliverable," +
      " prefix your response with `FINAL ANSWER` so the team knows to stop." +
      " You have access to the following tools: {tool_names}.\n{system_message}",
    ],
    new MessagesPlaceholder("messages"),
  ]);

  prompt = await prompt.partial({
    system_message: systemMessage,
    tool_names: toolNames,
  });

  return prompt.pipe(llm.bind({ tools: formattedTools }));
}

// This defines the object that is passed between each node
// in the graph. We will create different nodes for each agent and tool
const AgentState = Annotation.Root({
  messages: Annotation({
    reducer: (x, y) => x.concat(y),
  }),
  sender: Annotation({
    reducer: (x, y) => y ?? x ?? "user",
    default: () => "user",
  }),
})

const chartTool = tool(
  ({ data }) => {
    const width = 500;
    const height = 500;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");

    const x = d3
      .scaleBand()
      .domain(data.map((d) => d.label))
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.value) ?? 0])
      .nice()
      .range([height - margin.bottom, margin.top]);

    const colorPalette = [
      "#e6194B",
      "#3cb44b",
      "#ffe119",
      "#4363d8",
      "#f58231",
      "#911eb4",
      "#42d4f4",
      "#f032e6",
      "#bfef45",
      "#fabebe",
    ];

    data.forEach((d, idx) => {
      ctx.fillStyle = colorPalette[idx % colorPalette.length];
      ctx.fillRect(
        x(d.label) ?? 0,
        y(d.value),
        x.bandwidth(),
        height - margin.bottom - y(d.value),
      );
    });

    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.moveTo(margin.left, height - margin.bottom);
    ctx.lineTo(width - margin.right, height - margin.bottom);
    ctx.stroke();

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    x.domain().forEach((d) => {
      const xCoord = (x(d) ?? 0) + x.bandwidth() / 2;
      ctx.fillText(d, xCoord, height - margin.bottom + 6);
    });

    ctx.beginPath();
    ctx.moveTo(margin.left, height - margin.top);
    ctx.lineTo(margin.left, height - margin.bottom);
    ctx.stroke();

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const ticks = y.ticks();
    ticks.forEach((d) => {
      const yCoord = y(d); // height - margin.bottom - y(d);
      ctx.moveTo(margin.left, yCoord);
      ctx.lineTo(margin.left - 6, yCoord);
      ctx.stroke();
      ctx.fillText(d.toString(), margin.left - 8, yCoord);
    });

    const out = fs.createWriteStream(__dirname + '/chart.jpeg');
    
    // Disable 2x2 chromaSubsampling for deeper colors and use a higher quality
    canvas.createJPEGStream({
        quality: 0.95,
        chromaSubsampling: false
    }).pipe(out);

    out.on('finish', () =>  console.log('The JPEG file was created.'));
    
    return "Chart has been generated and displayed to the user!";
  },
  {
    name: "generate_bar_chart",
    description:
      "Generates a bar chart from an array of data points using D3.js and displays it for the user.",
    schema: z.object({
      data: z
        .object({
          label: z.string(),
          value: z.number(),
        })
        .array(),
    }),
  }
)

const tavilyTool = new TavilySearchResults();

/** 
 * This function runs an agent node in the graph.
 * @param {Object} props
 * @param {Object} props.state - The current state of the graph.
 * @param {Object} props.agent - The agent to run.
 * @param {string} props.name - The name of the agent.
 * @param {Object} props.config - The configuration object to pass to the agent.
*/
async function runAgentNode(props) {
  const { state, agent, name, config } = props;

  let result = await agent.invoke(state, config);
  // We convert the agent output into a format that is suitable
  // to append to the global state
  if (!result?.tool_calls || result.tool_calls.length === 0) {
    // If the agent is NOT calling a tool, we want it to
    // look like a human message.
    result = new HumanMessage({ ...result, name: name });
  }

  return {
    messages: [result],
    // Since we have a strict workflow, we can
    // track the sender so we know who to pass to next.
    sender: name,
  };
}

const llm = new ChatOpenAI({ modelName: "gpt-4o-mini" });

// Research agent and node
const researchAgent = await createAgent({
  llm,
  tools: [tavilyTool],
  systemMessage:
    "You should provide accurate data for the chart generator to use.",
});

async function researchNode(
  state,
  config,
) {
  return runAgentNode({
    state: state,
    agent: researchAgent,
    name: "Researcher",
    config,
  });
}

// Chart Generator
const chartAgent = await createAgent({
  llm,
  tools: [chartTool],
  systemMessage: "Any charts you display will be visible by the user.",
});

async function chartNode(state) {
  return runAgentNode({
    state: state,
    agent: chartAgent,
    name: "ChartGenerator",
  });
}

const tools = [tavilyTool, chartTool];
// This runs tools in the graph
const toolNode = new ToolNode(tools);

// Either agent can decide to end
function router(state) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1];
  if (lastMessage?.tool_calls && lastMessage.tool_calls.length > 0) {
    // The previous agent is invoking a tool
    return "call_tool";
  }
  if (
    typeof lastMessage.content === "string" &&
    lastMessage.content.includes("FINAL ANSWER")
  ) {
    // Any agent decided the work is done
    return "end";
  }
  return "continue";
}

// 1. Create the graph
const workflow = new StateGraph(AgentState)
   // 2. Add the nodes; these will do the work
  .addNode("Researcher", researchNode)
  .addNode("ChartGenerator", chartNode)
  .addNode("call_tool", toolNode);

// 3. Define the edges. We will define both regular and conditional ones
// After a worker completes, report to supervisor
workflow.addConditionalEdges("Researcher", router, {
  // We will transition to the other agent
  continue: "ChartGenerator",
  call_tool: "call_tool",
  end: END,
});

workflow.addConditionalEdges("ChartGenerator", router, {
  // We will transition to the other agent
  continue: "Researcher",
  call_tool: "call_tool",
  end: END,
});

workflow.addConditionalEdges(
  "call_tool",
  // Each agent node updates the 'sender' field
  // the tool calling node does not, meaning
  // this edge will route back to the original agent
  // who invoked the tool
  (x) => x.sender,
  {
    Researcher: "Researcher",
    ChartGenerator: "ChartGenerator",
  },
);

workflow.addEdge(START, "Researcher");
const graph = workflow.compile();

const streamResults = await graph.stream(
  {
    messages: [
      new HumanMessage({
        content: "Generate a bar chart of the US gdp over the past 3 years.",
      }),
    ],
  },
  { recursionLimit: 150 },
);

const prettifyOutput = (output) => {
  const keys = Object.keys(output);
  const firstItem = output[keys[0]];

  if ("messages" in firstItem && Array.isArray(firstItem.messages)) {
    const lastMessage = firstItem.messages[firstItem.messages.length - 1];
    console.dir({
      type: lastMessage._getType(),
      content: lastMessage.content,
      tool_calls: lastMessage.tool_calls,
    }, { depth: null });
  }

  if ("sender" in firstItem) {
    console.log({
      sender: firstItem.sender,
    })
  }
}

for await (const output of await streamResults) {
  if (!output?.__end__) {
    prettifyOutput(output);
    console.log("----");
  }
}