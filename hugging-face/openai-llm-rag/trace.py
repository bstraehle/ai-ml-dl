import os, wandb

from wandb.sdk.data_types.trace_tree import Trace

WANDB_API_KEY = os.environ["WANDB_API_KEY"]

def wandb_trace(config,
                is_rag_off, 
                prompt, 
                completion, 
                result, 
                chain, 
                cb, 
                err_msg, 
                start_time_ms, 
                end_time_ms):
    wandb.init(project = "openai-llm-rag")
    
    trace = Trace(
        kind = "chain",
        name = "" if (chain == None) else type(chain).__name__,
        status_code = "success" if (str(err_msg) == "") else "error",
        status_message = str(err_msg),
        metadata = {"chunk_overlap": "" if (is_rag_off) else config["chunk_overlap"],
                    "chunk_size": "" if (is_rag_off) else config["chunk_size"],
                   } if (str(err_msg) == "") else {},
        inputs = {"is_rag": not is_rag_off,
                  "prompt": prompt,
                  "chain_prompt": (str(chain.prompt) if (is_rag_off) else 
                                   str(chain.combine_documents_chain.llm_chain.prompt)),
                  "source_documents": "" if (is_rag_off) else str([doc.metadata["source"] for doc in completion["source_documents"]]),
                 } if (str(err_msg) == "") else {},
        outputs = {"result": result,
                   "cb": str(cb),
                   "completion": str(completion),
                  } if (str(err_msg) == "") else {},
        model_dict = {"client": (str(chain.llm.client) if (is_rag_off) else
                                 str(chain.combine_documents_chain.llm_chain.llm.client)),
                      "model_name": (str(chain.llm.model_name) if (is_rag_off) else
                                     str(chain.combine_documents_chain.llm_chain.llm.model_name)),
                      "temperature": (str(chain.llm.temperature) if (is_rag_off) else
                                      str(chain.combine_documents_chain.llm_chain.llm.temperature)),
                      "retriever": ("" if (is_rag_off) else str(chain.retriever)),
                     } if (str(err_msg) == "") else {},
        start_time_ms = start_time_ms,
        end_time_ms = end_time_ms
    )
    
    trace.log("evaluation")
                    
    wandb.finish()
