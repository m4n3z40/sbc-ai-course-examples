import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { HtmlToTextTransformer } from "@langchain/community/document_transformers/html_to_text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";

const llm = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: "gpt-4o-mini",
    temperature: 0
});

const loader = new CheerioWebBaseLoader('https://www.sbcschool.com.br/blog/o-que-se-espera-de-um-product-manager', {
    selector: 'h1, h2, h3, h4, h5, h6, p, li',
});

const splitter = RecursiveCharacterTextSplitter.fromLanguage("html", { 
    chunkSize: 2000, 
    chunkOverlap: 100,
});
const transformer = new HtmlToTextTransformer();

const docs = await loader.load();
const splitDocs = await splitter.pipe(transformer).invoke(docs);

// Define prompt
const prompt = PromptTemplate.fromTemplate("Summarize the main themes in these retrieved docs: {context}");
  
// Instantiate
const chain = await createStuffDocumentsChain({
    llm: llm,
    outputParser: new StringOutputParser(),
    prompt,
});

// Invoke
const result = await chain.invoke({ context: splitDocs });

console.log(result);
