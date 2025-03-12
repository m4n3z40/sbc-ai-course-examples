import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { chartTool, tavilyTool } from "./tools.mjs";
import { createAgent, runAgentNode, prettifyOutput } from "./utils.mjs";

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

const llm = new ChatOpenAI({ modelName: "gpt-4o-mini" });

// Research agent and node
const researchAgent = await createAgent({
  llm,
  tools: [tavilyTool],
  systemMessage: "You should provide accurate data for the chart generator to use.",
});

const researchNode = async (state, config) => runAgentNode({
  state,
  agent: researchAgent,
  name: "Researcher",
  config,
});

// Chart Generator
const chartAgent = await createAgent({
  llm,
  tools: [chartTool],
  systemMessage: "Any charts you display will be visible by the user.",
});

const chartNode = async (state) => runAgentNode({
  state: state,
  agent: chartAgent,
  name: "ChartGenerator",
});

// This runs tools in the graph
const toolNode = new ToolNode([tavilyTool, chartTool]);

/**
 * Router function to determine the next agent to call
 * @param {typeof Annotation} state 
 * @returns {'call_tool' | 'continue' | 'end'}
 */
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
        content: "Generate a bar chart of the Brazillian gdp over the past 5 years.",
      }),
    ],
  },
  { recursionLimit: 150 },
);

for await (const output of await streamResults) {
  if (!output?.__end__) {
    prettifyOutput(output);
    console.log("----");
  }
}