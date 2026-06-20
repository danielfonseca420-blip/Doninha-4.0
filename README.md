# Doninha IA - Middleware Neuro-Simbolico Hibrido

> Doninha IA transforma um LLM base em um middleware de raciocinio neuro-simbolico com camadas epistemologicas, logica paraconsistente, RAG hibrido, verificacao, auditoria e sintese final.

A IA Doninha e uma criacao de Daniel Barros Fonseca. O uso e permitido somente para usuarios privados. Para uso comercial ou governamental, contate o desenvolvedor pelo e-mail `danielfonseca420@proton.me` para informacoes sobre licenciamento de uso mediante pagamento de royalties.

## Visao Geral

O Doninha nao e apenas um chamador de LLM. Ele e um middleware em camadas que recebe um prompt, transforma esse prompt em conceitos, proposicoes, pesos logicos e sinteses verificadas, e so entao entrega uma resposta textual.

O objetivo do projeto e combinar:

- geracao estatistica de linguagem;
- estrutura simbolica inspirada em Aristoteles, Kant, Russell, Popper, Hempel e logica paraconsistente;
- recuperacao de contexto via KB local e RAG;
- auditoria de confianca, contradicao, valor-verdade e fontes;
- Chain of Thought auditavel por camada, exposto como resumo estruturado e nao como raciocinio interno bruto.

## Fluxo Principal L1-L7

```text
Prompt do usuario
  |
  v
L1 - Tabua de Conceitos
  |
  v
L2 - Juizos Kantianos e classificacao epistemica
  |
  v
Silogismo cientifico + Hempel + Popper
  |
  v
L3 - Logica Paraconsistente + Weighted Dynamic Ensemble
  |
  v
L4 - Sintese Russelliana + Chain of Verification
  |
  v
L5 - Geracao textual com provider base
  |
  v
L6 - Refinamento final
  |
  v
L7 - Texto definitivo e auditavel
```

O ponto de entrada principal e [pipeline.py](pipeline.py), por meio da classe `HybridLLMPipeline`.

## Camadas do Middleware

### L1 - Tabua de Conceitos

Arquivo principal: [l1_concept_table.py](l1_concept_table.py)

Ferramentas principais:

- `ConceptNode`: estrutura de conceito com definicao, sinonimos, antonimos, hiponimos, hiperonimos, dominio, contexto aplicacional e fonte canonica.
- `ConceptTable`: tabela semantica que extrai conceitos conhecidos do prompt.
- `SpacySemanticTermExtractor`: extrator opcional baseado em spaCy para termos e sintagmas.
- `LogicLMSymbolicSolver`: enriquecedor simbolico que avalia compatibilidade de contexto e gera alertas canonicos.

Funcao da camada:

- extrair conceitos do prompt;
- identificar relacoes semanticas;
- inferir dominio;
- enriquecer conceitos com contexto de aplicacao;
- evitar uso incompatível de conceitos canonicos;
- preparar material semantico para L2.

### L2 - Juizos Kantianos

Arquivo principal: [l2_kantian_judgments.py](l2_kantian_judgments.py)

Ferramentas principais:

- `KantianJudgment`: representa uma proposicao refinada com quantidade, qualidade, relacao, modalidade, prioridade e classificacao epistemica.
- `SyntaxProfile`: perfil sintatico minimo do prompt.
- `BERTAssertionClassifier`: classificador opcional baseado em `transformers.pipeline` para T/I/F.
- `KantianJudgmentEngine`: motor que transforma conceitos em juizos priorizados.

Funcao da camada:

- converter conceitos em proposicoes;
- classificar as proposicoes pela tabua kantiana;
- gerar variacoes afirmativas, negativas, hipoteticas e apoditicas;
- atribuir prioridade epistemologica;
- fornecer hipoteses para os filtros cientificos e para L3.

### Etapa Intermediaria - Silogismo, Hempel e Popper

Arquivo principal: [syllogism_module.py](syllogism_module.py)

Ferramentas principais:

- `Syllogism`: estrutura de premissas e conclusao.
- `AristotelianSyllogismValidator`: valida relacoes silogisticas.
- `HempelFilter`: remove hipoteses espurias por baixa relacao semantica.
- `PopperFalsifiability`: avalia falsificabilidade.
- `ScientificSyllogismPipeline`: integra as tres rotinas antes de L3.

Funcao da etapa:

- filtrar hipoteses muito soltas;
- preservar proposicoes mais testaveis;
- reduzir ruido antes do calculo paraconsistente.

### L3 - Logica Paraconsistente

Arquivos principais:

- [l3_paraconsistent.py](l3_paraconsistent.py)
- [paraconsistent_rules.py](paraconsistent_rules.py)
- [neural_truth_model.py](neural_truth_model.py)

Ferramentas principais:

- `ParaconsistentValue`: valor logico com `mu`, `lambda`, certeza, contradicao, estado, valor-verdade e metadados do ensemble.
- `ManyValuedRouter`: roteia pares de proposicoes para contradicao real, incerteza estatistica, ambiguidade ou nao classificado.
- `ParaconsistentEngine`: calcula anotacoes fuzzy e estados logicos.
- `ParaconsistentRules`: regras de 12 estados derivadas de `data/Fuzzy.txt`.
- `TruthScoringModel`: modelo neural baseado em Transformer para estado paraconsistente e valor-verdade.
- `neural_annotations`: converte a saida neural em `mu/lambda`.

Funcao da camada:

- calcular `mu` como evidencia favoravel;
- calcular `lambda` como evidencia contraria;
- derivar certeza `Gc = mu - lambda`;
- derivar contradicao `Gct = mu + lambda - 1`;
- classificar o estado logico;
- combinar heuristica e modelo neural por Weighted Dynamic Ensemble.

### Weighted Dynamic Ensemble em L3

A L3 combina duas fontes:

- heuristica baseada em KB, prioridade L2 e contradicoes locais;
- anotacao neural do `TruthScoringModel`, quando disponivel.

O fluxo e:

1. calcula `h_mu/h_lam` pela heuristica;
2. calcula `n_mu/n_lam` pelo modelo neural;
3. mede concordancia fuzzy entre as duas fontes;
4. define pesos dinamicos:
   - `heuristic_weight = 0.65 + 0.25 * agreement`
   - `neural_weight = 1.0 - heuristic_weight`
5. combina `mu/lambda`;
6. aplica regularizacao paraconsistente em `Gc/Gct`;
7. registra `confidence`, `ensemble_agreement`, `neural_state` e `neural_truth`.

Quando o modelo neural nao esta disponivel, a camada continua funcionando em modo heuristico, com `heuristic_weight=1.0` e `neural_weight=0.0`.

### L4 - Sintese Russelliana + CoVe

Arquivos principais:

- [l4_synthesis.py](l4_synthesis.py)
- [l4_russell_equivalence.py](l4_russell_equivalence.py)
- [l4_chain_verification.py](l4_chain_verification.py)

Ferramentas principais:

- `SynthesisResult`: resultado estruturado da sintese.
- `RussellianSynthesisEngine`: combina proposicoes L3 com prioridades L2 e KB.
- `RussellConceptBase`: base conceitual extraida de `data/russell.txt`.
- `score_proposition_by_concepts`: calcula correspondencia conceitual proposicao-fato.
- `ChainOfVerificationAgent`: aplica Chain of Verification no resultado.

Funcao da camada:

- selecionar e ponderar a melhor hipotese;
- calcular sintese baseada em valor-verdade, prioridade e correspondencia;
- verificar a resposta por CoVe;
- produzir resposta estruturada antes da geracao textual.

### L5 - Geracao Textual

Arquivo principal: [l5_generation.py](l5_generation.py)

Ferramentas principais:

- `build_context_for_generation`: monta o contexto L1-L4 para o provider.
- `generate_with_ollama_l5`: gera texto com Ollama.
- `generate_with_custom_lm`: usa o modelo customizado local.
- `generate_response`: roteia a geracao para Ollama, provider remoto, custom LM ou fallback template.

Funcao da camada:

- transformar a sintese L4 em linguagem natural;
- preservar contexto epistemologico;
- usar provider configurado sem perder o fallback local.

### L6 - Refinamento Final

Arquivo principal: [l6_final_response.py](l6_final_response.py)

Ferramentas principais:

- `EpistemicContext`: contexto agregado com estados L3, rotas paraconsistentes e classificacoes BERT.
- `FinalResponseEngine`: gera e reescreve a resposta fluida.

Funcao da camada:

- melhorar clareza e coesao;
- ajustar tom ao grau de confianca;
- mencionar incertezas e contradicoes quando necessario;
- preservar os dados epistemicos.

### L7 - Texto Final Definitivo

Arquivos principais:

- [l7_final_text.py](l7_final_text.py)
- [agente_sintese_final.py](agente_sintese_final.py)

Ferramentas principais:

- `FinalTextEngine`: cria o prompt final e chama provider/template.
- `synthesize_final_text`: agente de sintese final reutilizavel via CLI ou pipeline.

Funcao da camada:

- integrar L1-L6 em um texto final;
- classificar audiencia (`leigo`, `tecnico`, `academico`);
- ajustar tom conforme confianca, contradicao e estado;
- usar o rastro CoT auditavel como contexto de sintese.

## Chain of Thought Auditavel

Arquivos principais:

- [cot_hierarchical.py](cot_hierarchical.py)
- [prompt_engineering.py](prompt_engineering.py)

Ferramentas principais:

- `CoTStep`: registra camada, titulo, resumo do raciocinio, decisoes-chave, saida e duracao.
- `HierarchicalCoTTrace`: agrega os passos L1-L7 e exporta `dict` ou Markdown.
- `HierarchicalCoTOrchestrator`: wrapper para executar o pipeline com retorno de trace.
- `get_layer_prompt`: gera prompts especificos para L1-L7.

Funcao:

- gerar uma trilha de auditoria por camada;
- registrar decisoes e resumos sem expor raciocinio interno bruto;
- permitir `return_cot=True` no pipeline.

Uso:

```python
from pipeline import HybridLLMPipeline

pipeline = HybridLLMPipeline(verbose=False)
result = pipeline.process("O que e conhecimento?", return_cot=True)

print(result.response)
print(result.cot_markdown)
```

## Providers de LLM

Arquivo principal: [llm_provider_client.py](llm_provider_client.py)

Providers suportados:

- `ollama`
- `openai`
- `anthropic`
- `gemini`
- `grok`
- `groq`
- `meta`
- `template`
- `custom_lm`

Funcoes:

- normalizar providers;
- definir modelos padrao;
- gerar texto local com Ollama;
- chamar APIs remotas via `requests`;
- extrair texto de formatos OpenAI-like, Anthropic e Gemini;
- fornecer fallback quando provider nao esta configurado.

## RAG, KB e Busca

### Knowledge Base

Arquivo principal: [knowledge_base.py](knowledge_base.py)

Ferramentas:

- `SEED_KNOWLEDGE_BASE`: base minima de fallback.
- `load_kb_from_file`: carrega JSON, JSONL/NDJSON ou documentos.
- `merge_kb`: mescla bases.
- `enrich_kb_from_chroma`: enriquece KB via ChromaDB.
- `get_domain_knowledge_base`: recupera KB por dominio.
- `get_knowledge_base`: ponto de entrada geral da KB.

Funcao:

- fornecer termos e pesos para L1, L3 e L4;
- permitir KB generica, por dominio e enriquecida por RAG.

### RAG Hibrido

Arquivos principais:

- [rag_hybrid_context_injection.py](rag_hybrid_context_injection.py)
- [l1_l2_rag_integration.py](l1_l2_rag_integration.py)
- [pipeline_with_rag_integration.py](pipeline_with_rag_integration.py)

Ferramentas:

- `RetrievalStrategy`: `direct_injection`, `semantic_retrieval`, `hybrid`, `domain_aware`.
- `DomainContext`: configura dominio, KB, Chroma, prompt e pesos.
- `RetrievedDocument`: documento recuperado.
- `RAGContext`: contexto compilado.
- `HybridRAGContextInjectionEngine`: motor de retrieval + injection.
- `IntegratedL1L2RAGPipeline`: integra RAG com L1 e L2.
- `HybridLLMPipelineWithRAG`: pipeline alternativa com RAG embutido.

Funcao:

- detectar dominio;
- injetar contexto direto de KB;
- recuperar documentos em ChromaDB;
- compilar contexto para L1/L2 e resposta final.

### Agente de Busca

Arquivo principal: [agente_busca_web.py](agente_busca_web.py)

Ferramentas:

- `get_retriever_tool`: ferramenta de busca local em ChromaDB.
- `get_duckduckgo_tool`: ferramenta DuckDuckGo quando instalada.
- `build_agent`: monta agente ReAct via LangChain.
- `run_search_for_context`: retorna contexto textual para o pipeline.

Funcao:

- buscar contexto local e/ou web;
- enriquecer respostas quando a L3 indica incerteza, indeterminacao ou fallback heuristico.

## Interfaces

### CLI

Arquivo principal: [pipeline.py](pipeline.py)

Comandos:

```bash
python pipeline.py --demo
python pipeline.py --prompt "Explique logica paraconsistente em 5 linhas"
python pipeline.py --repl
```

### API REST

Arquivo principal: [api.py](api.py)

Tecnologias:

- FastAPI
- Pydantic
- Uvicorn

Endpoints:

- `GET /health`: status da API.
- `POST /process`: processa prompt unico.
- `POST /chat`: processa mensagem com sessao.
- `POST /agent`: executa apenas o agente de busca.

### Chainlit

Arquivo principal: [app.py](app.py)

Tecnologias:

- Chainlit
- Ollama

Funcao:

- interface simples de chat;
- streaming via `ollama.chat`;
- historico basico em memoria.

### Standalone

Arquivo principal: [doninha_standalone.py](doninha_standalone.py)

Funcao:

- arquivo consolidado com a maior parte dos modulos embutidos;
- util para distribuicao ou execucao sem separar muitos arquivos;
- inclui suporte ao trace CoT L1-L7 e ao Weighted Dynamic Ensemble.

## Treinamento e Modelos

### TruthScoringModel

Arquivos:

- [neural_truth_model.py](neural_truth_model.py)
- [train_truth_model.py](train_truth_model.py)

Ferramentas:

- PyTorch
- Transformers (`AutoModel`, `AutoTokenizer`)
- Dataset customizado de proposicoes
- Regras de `data/Fuzzy.txt`

Funcao:

- treinar um classificador/regressor para estado paraconsistente e valor-verdade;
- gerar `truth_scoring_model.pt`;
- alimentar a L3 neural.

### Modelo de Linguagem Customizado

Arquivos:

- [custom_tokenizer.py](custom_tokenizer.py)
- [custom_lm_model.py](custom_lm_model.py)
- [pretrain_custom_lm.py](pretrain_custom_lm.py)
- [run_pretrain.py](run_pretrain.py)

Ferramentas:

- SentencePiece
- PyTorch
- Transformer customizado
- Corpus filosofico local

Funcao:

- treinar tokenizer proprio;
- treinar LM pequeno local;
- permitir provider `custom_lm`.

### Base Russelliana

Arquivos:

- [l4_russell_equivalence.py](l4_russell_equivalence.py)
- [train_l4_russell.py](train_l4_russell.py)

Funcao:

- extrair conceitos de equivalencia/correspondencia de `data/russell.txt`;
- gerar `l4_russell_concepts.json`;
- ponderar L4 por correspondencia conceitual.

## Metricas e Avaliacao

Arquivos principais:

- [metrics.py](metrics.py)
- [eval_pipeline.py](eval_pipeline.py)
- [test_epistemic_classification.py](test_epistemic_classification.py)
- [test_l7_tone_guidance.py](test_l7_tone_guidance.py)
- [test_provider_config_resolution.py](test_provider_config_resolution.py)
- [test_citation_behavior.py](test_citation_behavior.py)
- [test_rag_hybrid.py](test_rag_hybrid.py)

Ferramentas:

- `coherence_l3`: mede coerencia entre verdade, estado e contradicao.
- BLEU simples.
- ROUGE-L simples.
- similaridade semantica via SentenceTransformers.
- suites de testes para L2, L7, providers, citacao e RAG.

## Configuracao

Arquivos:

- [config.yaml](config.yaml)
- [config_loader.py](config_loader.py)
- [config_rag.yaml](config_rag.yaml)

`config_loader.py` resolve:

- paths relativos;
- providers;
- modelos;
- KB e Chroma;
- configuracao de agente;
- API;
- chat;
- L1 spaCy.

Exemplo:

```yaml
generation:
  provider: "ollama"
  ollama_model: "doninha8:latest"
  ollama_host: "http://localhost:11434"

finalization:
  provider: "ollama"
  ollama_model: "doninha8:latest"

l7:
  provider: "ollama"
  model: "doninha8:latest"
```

Variaveis uteis:

- `OLLAMA_MODEL`
- `OLLAMA_HOST`
- `GENERATION_PROVIDER`
- `FINALIZATION_PROVIDER`
- `L7_PROVIDER`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `GROK_API_KEY`
- `GROQ_API_KEY`
- `META_API_KEY`
- `VECTOR_DB_PATH`

## Ferramentas do Middleware IA Doninha

### Ferramentas internas de raciocinio

| Ferramenta | Arquivo | Funcao |
|---|---|---|
| `ConceptTable` | `l1_concept_table.py` | Extrai conceitos e relacoes semanticas. |
| `LogicLMSymbolicSolver` | `l1_concept_table.py` | Enriquece conceitos com contexto e valida compatibilidade canonica. |
| `KantianJudgmentEngine` | `l2_kantian_judgments.py` | Gera juizos kantianos priorizados. |
| `BERTAssertionClassifier` | `l2_kantian_judgments.py` | Classifica proposicoes assertricas em verdade, indeterminacao e falsidade. |
| `ScientificSyllogismPipeline` | `syllogism_module.py` | Aplica silogismo, filtro de Hempel e falsificabilidade popperiana. |
| `ParaconsistentEngine` | `l3_paraconsistent.py` | Calcula `mu/lambda`, estado, verdade, certeza e contradicao. |
| `ManyValuedRouter` | `l3_paraconsistent.py` | Classifica pares de proposicoes como contradicao, incerteza ou ambiguidade. |
| `ParaconsistentRules` | `paraconsistent_rules.py` | Implementa regras fuzzy de 12 estados. |
| `TruthScoringModel` | `neural_truth_model.py` | Modelo neural para scoring paraconsistente. |
| `RussellianSynthesisEngine` | `l4_synthesis.py` | Sintetiza proposicoes por equivalencia/correspondencia. |
| `ChainOfVerificationAgent` | `l4_chain_verification.py` | Verifica e revisa a resposta por CoVe. |
| `FinalResponseEngine` | `l6_final_response.py` | Refina a resposta final e monta contexto epistemico. |
| `FinalTextEngine` | `l7_final_text.py` | Produz o texto final definitivo. |
| `HierarchicalCoTTrace` | `cot_hierarchical.py` | Registra Chain of Thought auditavel L1-L7. |
| `get_layer_prompt` | `prompt_engineering.py` | Gera prompts especificos por camada. |

### Ferramentas de contexto, busca e RAG

| Ferramenta | Arquivo | Funcao |
|---|---|---|
| `get_knowledge_base` | `knowledge_base.py` | Carrega KB geral ou por dominio. |
| `enrich_kb_from_chroma` | `knowledge_base.py` | Enriquece KB com trechos ChromaDB. |
| `HybridRAGContextInjectionEngine` | `rag_hybrid_context_injection.py` | Combina injeção direta e retrieval semantico. |
| `IntegratedL1L2RAGPipeline` | `l1_l2_rag_integration.py` | Usa RAG para enriquecer L1/L2. |
| `run_search_for_context` | `agente_busca_web.py` | Executa busca local/web e retorna contexto. |
| `DuckDuckGoSearchRun` | `agente_busca_web.py` | Busca web via DuckDuckGo quando disponivel. |
| `Chroma` | varios | Base vetorial local para retrieval. |
| `HuggingFaceEmbeddings` | varios | Embeddings para ChromaDB. |

### Ferramentas de geracao

| Ferramenta | Arquivo | Funcao |
|---|---|---|
| `generate_text` | `llm_provider_client.py` | Roteia chamadas para providers locais/remotos. |
| Ollama | `llm_provider_client.py`, `l5_generation.py`, `l7_final_text.py` | Provider local padrao. |
| OpenAI | `llm_provider_client.py` | Provider remoto OpenAI-compatible. |
| Anthropic | `llm_provider_client.py` | Provider remoto Claude. |
| Gemini | `llm_provider_client.py` | Provider remoto Google Gemini. |
| Grok | `llm_provider_client.py` | Provider remoto xAI. |
| Groq | `llm_provider_client.py` | Provider remoto OpenAI-compatible da Groq. |
| Meta/Llama API | `llm_provider_client.py` | Provider remoto para modelos Llama. |
| `custom_lm` | `custom_lm_model.py` | Modelo local customizado treinavel. |
| `template` | varios | Fallback sem chamada externa. |

### Ferramentas de treinamento e dados

| Ferramenta | Arquivo | Funcao |
|---|---|---|
| `train_truth_model.py` | treinamento L3 | Treina o `TruthScoringModel`. |
| `train_l4_russell.py` | treinamento L4 | Gera base de conceitos russellianos. |
| `pretrain_custom_lm.py` | LM customizado | Treina modelo de linguagem proprio. |
| `custom_tokenizer.py` | tokenizacao | Treina/carrega SentencePiece. |
| `corpus_utils.py` | corpus | Leitura e preparacao de corpus. |
| `build_concepts_from_english_dict.py` | conceitos | Extrai conceitos de dicionario em ingles. |
| `philosophy-corpus/encode_corpus.py` | corpus | Codifica corpus para treino. |

### Ferramentas de interface e operacao

| Ferramenta | Arquivo | Funcao |
|---|---|---|
| `HybridLLMPipeline` | `pipeline.py` | Orquestra L1-L7. |
| `HybridLLMPipelineWithRAG` | `pipeline_with_rag_integration.py` | Variante com RAG hibrido. |
| FastAPI | `api.py` | API HTTP. |
| Chainlit | `app.py` | Interface de chat local. |
| `ChatSession` | `chat_session.py` | Historico de conversa. |
| `doninha_standalone.py` | standalone | Distribuicao consolidada. |
| scripts `consolidate_*` | varios | Geram arquivos consolidados. |

## Como Executar

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

Rodar Ollama:

```bash
ollama serve
ollama pull doninha8:latest
```

Executar pipeline:

```bash
python pipeline.py --prompt "Explique logica paraconsistente em 5 linhas"
python pipeline.py --demo
python pipeline.py --repl
```

Executar API:

```bash
python api.py
```

Executar Chainlit:

```bash
chainlit run app.py
```

Executar RAG hibrido:

```bash
python pipeline_with_rag_integration.py
python example_rag_hybrid_usage.py
```

Treinar L3 neural:

```bash
python train_truth_model.py
```

Treinar base L4 Russell:

```bash
python train_l4_russell.py
```

Treinar LM customizado:

```bash
python run_pretrain.py
```

## Auditoria da Saida

O pipeline adiciona blocos de auditoria como:

- `[AUDIT L4]`
- `[AUDIT L5]`
- `[AUDIT L6]`
- `[AUDIT L7]`

Eles podem incluir:

- provider usado;
- modelo usado;
- valor-verdade;
- certeza;
- contradicao;
- estado logico;
- fontes locais/canonicas;
- resumo L2;
- resumo L3;
- pesos do ensemble L3.

Com `return_cot=True`, o resultado tambem recebe:

- `result.cot_trace`
- `result.cot_markdown`

## Estrutura de Arquivos

```text
pipeline.py                     Orquestrador principal L1-L7
doninha_standalone.py            Versao consolidada
api.py                           API FastAPI
app.py                           Interface Chainlit/Ollama
l1_concept_table.py              L1 - conceitos
l2_kantian_judgments.py          L2 - juizos
syllogism_module.py              Silogismo/Hempel/Popper
l3_paraconsistent.py             L3 - paraconsistente
paraconsistent_rules.py          Regras fuzzy
neural_truth_model.py            Modelo neural de truth scoring
l4_synthesis.py                  L4 - sintese
l4_chain_verification.py         CoVe
l4_russell_equivalence.py        Base Russelliana
l5_generation.py                 L5 - geracao
l6_final_response.py             L6 - refinamento
l7_final_text.py                 L7 - texto final
prompt_engineering.py            Prompts por camada
cot_hierarchical.py              Trace CoT auditavel
knowledge_base.py                KB
rag_hybrid_context_injection.py  RAG hibrido
llm_provider_client.py           Providers
```

## Observacoes Importantes

- O Doninha funciona mesmo sem modelo neural L3, usando heuristica paraconsistente.
- O provider `template` permite rodar partes do fluxo sem LLM externo.
- O provider local recomendado e Ollama.
- O RAG depende de ChromaDB e embeddings quando configurado.
- O arquivo standalone e grande por conter muitos modulos consolidados.
- Os arquivos em `data/` e `philosophy-corpus/` sao usados como corpus, KB e fontes de treinamento.

## Referencia

Daniel Fonseca - criador da Inteligencia Artificial Doninha, middleware que transforma LLMs em um modelo neuro-simbolico hibrido com ferramentas de auditoria.

A fundamentacao teorica esta no artigo "Uma verdadeira Epistemologia para a Inteligencia Artificial".

Todos os direitos reservados a Daniel Barros Fonseca, sujeitos a licenciamento para uso de terceiros. A tecnologia IA Doninha encontra-se em processo corrente de registro de patente. Plagios ou apropriacoes indevidas desta tecnologia proprietaria estarao sujeitos a indenizacao, reparacao de danos, reparacao por lucros perdidos e obrigatoriedade de publicacao de carta de retratacao por espionagem industrial.
