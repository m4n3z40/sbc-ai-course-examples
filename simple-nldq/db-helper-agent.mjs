import path from "node:path";
import url from "node:url";
import readline from 'node:readline';
import util from 'node:util';
import { SqlDatabase } from "langchain/sql_db";
import { SqlToolkit } from "langchain/agents/toolkits/sql";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { isAIMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { DataSource } from "typeorm";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

const dataSource = new DataSource({
    type: "sqlite",
    database: path.resolve(__dirname, "Chinook.db"),
});

const db = await SqlDatabase.fromDataSourceParams({
    appDataSource: dataSource,
});

async function queryAsList(database, query) {
    const res = JSON.parse(
      await database.run(query)
    )
      .flat()
      .filter((el) => el != null);

    const justValues = res.map((item) =>
      Object.values(item)[0]
        .replace(/\b\d+\b/g, "")
        .trim()
    );

    return justValues;
}

let artists = await queryAsList(db, "SELECT Name FROM Artist");
let albums = await queryAsList(db, "SELECT Title FROM Album");
let properNouns = artists.concat(albums);

const llm = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: "gpt-4o-mini",
    temperature: 0,
});

const toolkit = new SqlToolkit(db, llm);

const tools = toolkit.getTools();

const embbedings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
});

const vectorstore = new MemoryVectorStore(embbedings);

const documents = properNouns.map((noun) => new Document({ pageContent: noun }));

await vectorstore.addDocuments(documents);

const retriever = vectorstore.asRetriever(5);

const retrieverTool = createRetrieverTool(retriever, {
    name: "search-proper-nouns",
    description:
        "Use to look up values to filter on. Input is an approximate spelling " +
        "of the proper noun, output is valid proper nouns. Use the noun most " +
        "similar to the search.",
});

/**
 * @type {ChatPromptTemplate}
 */
const systemPromptTemplate = await pull("langchain-ai/sql-agent-system-prompt");

const systemMessage = await systemPromptTemplate.format({
    dialect: db.appDataSourceOptions.type,
    top_k: 10,
}) + 
    "If you need to filter on a proper noun like a Name, you must ALWAYS first look up " +
    "the filter value using the 'search-proper-nouns' tool! Do not try to " +
    "guess at the proper name - use this function to find similar ones.";

const agent = createReactAgent({
    llm,
    tools: tools.concat([retrieverTool]),
    stateModifier: systemMessage,
});

const prettyPrint = (message) => {
    let txt = `[${message._getType()}]: ${message.content}`;
    if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
      const tool_calls = message?.tool_calls
        ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
        .join("\n");
      txt += ` \nTools: \n${tool_calls}`;
    }
    console.log(txt);
};

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const inquiry = util.promisify(rl.question).bind(rl);

let input = {
    messages: [
        { role: "user", content: await inquiry("What would you like to know about the Chinook db?\n> ") },
    ],
};

rl.close();

for await (const step of await agent.stream(input, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
    console.log("-----\n");
}