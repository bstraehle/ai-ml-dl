import os, wandb

from wandb.sdk.data_types.trace_tree import Trace

WANDB_API_KEY = os.environ["WANDB_API_KEY"]

RAG_LANGCHAIN  = "LangChain"
RAG_LLAMAINDEX = "LlamaIndex"

def trace_wandb(config,
                rag_option,
                prompt,
                completion,
                result,
                callback,
                err_msg,
                start_time_ms,
                end_time_ms):
    wandb.init(project = "openai-llm-rag")

    if (rag_option == RAG_LANGCHAIN):
        prompt_template = os.environ["LANGCHAIN_TEMPLATE"]
    elif (rag_option == RAG_LLAMAINDEX):
        prompt_template = os.environ["LLAMAINDEX_TEMPLATE"]
    else:
        prompt_template = os.environ["TEMPLATE"]
                    
    trace = Trace(
        kind = "LLM",
        name = "Context-Aware Reasoning Application",
        status_code = "success" if (str(err_msg) == "") else "error",
        status_message = str(err_msg),
        inputs = {"prompt": prompt,
                  "prompt_template": prompt_template,
                  "rag_option": rag_option,
                  "config": str(config)
                 } if (str(err_msg) == "") else {},
        outputs = {"result": str(result),
                   "callback": str(callback),
                   "completion": str(completion)
                  } if (str(err_msg) == "") else {},
        start_time_ms = start_time_ms,
        end_time_ms = end_time_ms
    )
    
    trace.log("evaluation")
                    
    wandb.finish()
