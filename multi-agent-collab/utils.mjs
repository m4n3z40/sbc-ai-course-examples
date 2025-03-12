import { HumanMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";

/**
 * Create an agent that can run a set of tools.
 * @param {Object} options
 * @param {ChatOpenAI} options.llm - The language model to use.
 * @param {Array} options.tools - The tools to use.
 * @param {string} options.systemMessage - The system message to display.
 * @returns {ChatPrompt} - The agent.
 */
export async function createAgent({ llm, tools, systemMessage }) {
  const toolNames = tools.map((tool) => tool.name).join(", ");
  const formattedTools = tools.map((t) => convertToOpenAITool(t));

  let prompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      [
        "You are a helpful AI assistant, collaborating with other assistants.",
        " Use the provided tools to progress towards answering the question.",
        " If you are unable to fully answer, that's OK, another assistant with different tools ",
        " will help where you left off. Execute what you can to make progress.",
        " If you or any of the other assistants have the final answer or deliverable,",
        " prefix your response with FINAL ANSWER so the team knows to stop.",
        " You have access to the following tools: {tool_names}.\n{system_message}",
      ].join(""),
    ],
    new MessagesPlaceholder("messages"),
  ]);

  prompt = await prompt.partial({
    system_message: systemMessage,
    tool_names: toolNames,
  });

  return prompt.pipe(llm.bind({ tools: formattedTools }));
}

/** 
 * This function runs an agent node in the graph.
 * @param {Object} props
 * @param {Object} props.state - The current state of the graph.
 * @param {Object} props.agent - The agent to run.
 * @param {string} props.name - The name of the agent.
 * @param {Object} props.config - The configuration object to pass to the agent.
 * @returns {Object} - The output of the agent.
*/
export async function runAgentNode(props) {
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

/**
 * Pretty print the output of an agent.
 * @param {*} output
 * @returns {void}
 */
export const prettifyOutput = (output) => {
    const keys = Object.keys(output);
    const firstItem = output[keys[0]];

    if ("sender" in firstItem) {
        console.log({ sender: firstItem.sender });
    }

    if ("messages" in firstItem && Array.isArray(firstItem.messages)) {
        const lastMessage = firstItem.messages[firstItem.messages.length - 1];

        console.dir({
            type: lastMessage._getType(),
            content: lastMessage.content,
            tool_calls: lastMessage.tool_calls,
        }, { depth: null });
    }
}
