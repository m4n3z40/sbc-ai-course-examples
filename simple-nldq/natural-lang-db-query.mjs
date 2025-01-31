import path from "node:path";
import url from "node:url";
import readline from 'node:readline';
import util from 'node:util';
import { SqlDatabase } from "langchain/sql_db";
import { QuerySqlTool } from "langchain/tools/sql";
import { pull } from "langchain/hub";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation, StateGraph, MemorySaver } from "@langchain/langgraph";
import { DataSource } from "typeorm";
import { z } from "zod";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

const dataSource = new DataSource({
    type: "sqlite",
    database: path.resolve(__dirname, "Chinook.db"),
});

const db = await SqlDatabase.fromDataSourceParams({
    appDataSource: dataSource,
});

const llm = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: "gpt-4o-mini",
    temperature: 0,
});

const queryOutupt = z.object({
    query: z.string().describe("Syntactically valid SQL query."),
});

const structuredLlm = llm.withStructuredOutput(queryOutupt);

const queryPromptTemplate = await pull("langchain-ai/sql-query-system-prompt");

const writeQuery = async (state) => {
    const promptValue = await queryPromptTemplate.invoke({
        dialect: db.appDataSourceOptions.type,
        top_k: 10,
        table_info: await db.getTableInfo(),
        input: state.question,
    });

    const result = await structuredLlm.invoke(promptValue);
    return { query: result.query };
};

const executeQuery = async (state) => {
    const executeQueryTool = new QuerySqlTool(db);

    return { result: await executeQueryTool.invoke(state.query) };
};

const generateAnswer = async (state) => {
    const promptValue =
        "Given the following user question, corresponding SQL query, " +
        "and SQL result, answer the user question.\n\n" +
        `Question: ${state.question}\n` +
        `SQL Query: ${state.query}\n` +
        `SQL Result: ${state.result}\n`;

  const response = await llm.invoke(promptValue);

  return { answer: response.content };
};

const graphBuilder = new StateGraph({
    stateSchema: Annotation.Root({
        question: Annotation,
        query: Annotation,
        result: Annotation,
        answer: Annotation,
    })
})
    .addNode("writeQuery", writeQuery)
    .addNode("executeQuery", executeQuery)
    .addNode("generateAnswer", generateAnswer)
    .addEdge("__start__", "writeQuery")
    .addEdge("writeQuery", "executeQuery")
    .addEdge("executeQuery", "generateAnswer")
    .addEdge("generateAnswer", "__end__");

const checkpointer = new MemorySaver();

const graph = graphBuilder.compile({
    checkpointer,
});

const threadConfig = {
    configurable: { thread_id: "1" },
    streamMode: "updates",
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const inquiry = util.promisify(rl.question).bind(rl);

const input = { question: await inquiry("What would you like to know about the Chinook db?\n> ") };

rl.close();

console.log(input);
console.log("\n====\n");

for await (const state of await graph.stream(input, threadConfig)) {
    console.log(state);
    console.log("\n====\n");
}