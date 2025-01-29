import path from "node:path";
import url from "node:url";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

async function createRAG() {
    // 1. Carregar documento PDF
    const loader = new PDFLoader(path.resolve(__dirname, "sbc-arquitetura-vcs.pdf"));
    const docs = await loader.load();

    // 2. Instanciar modelo OpenAI
    const model = new ChatOpenAI({
        apiKey: process.env.OPENAI_API_KEY,
        model: "gpt-4o-mini",
        temperature: 0.3,
    });

    // 3. Dividir em chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    const splits = await textSplitter.splitDocuments(docs);

    // 4. Criar embeddings e armazenar
    const vectorstore = await MemoryVectorStore.fromDocuments(
        splits,
        new OpenAIEmbeddings()
    );
    
    const retriever = vectorstore.asRetriever();

    // 5. Criar prompt
    const systemTemplate = [
        `Você é um assistente para tarefas de perguntas e respostas. `,
        `Use os seguintes trechos do contexto recuperado para responder à pergunta. `,
        `Se você não souber a resposta, diga que não sabe.`,
        `Use no máximo cinco frases e mantenha a resposta concisa.`,
        `\n\n`,
        `{context}`,
    ].join("");

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", systemTemplate],
        ["human", "{input}"],
    ]);

    // 6. Criar chain de pergunta e resposta
    const questionAnswerChain = await createStuffDocumentsChain({ llm: model, prompt });

    const ragChain = await createRetrievalChain({
        retriever,
        combineDocsChain: questionAnswerChain,
    });


    // 7. Fazer uma pergunta e obter resposta
    const results = await ragChain.invoke({
        input: "Quais sao os aprendizados do case do Mercado Livre?",
    });
    
    console.log(results.answer);
}

createRAG().catch(console.error);
