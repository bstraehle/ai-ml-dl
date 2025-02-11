import os, wandb

from wandb.sdk.data_types.trace_tree import Trace

WANDB_API_KEY = os.environ["WANDB_API_KEY"]

def trace_wandb(config,
                agent_option,
                prompt,
                completion,
                result,
                err_msg,
                start_time_ms,
                end_time_ms):
    wandb.init(project = "openai-llm-agent")

    trace = Trace(
        kind = "LLM",
        name = "Real-Time Reasoning Application",
        status_code = "success" if (str(err_msg) == "") else "error",
        status_message = str(err_msg),
        inputs = {"prompt": prompt,
                  "agent_option": agent_option,
                  "config": str(config)
                 } if (str(err_msg) == "") else {},
        outputs = {"result": str(result),
                   "completion": str(completion)
                  } if (str(err_msg) == "") else {},
        start_time_ms = start_time_ms,
        end_time_ms = end_time_ms
    )
    
    trace.log("evaluation")
                    
    wandb.finish()
