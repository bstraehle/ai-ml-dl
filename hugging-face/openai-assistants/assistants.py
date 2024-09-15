Hugging Face's logo
Hugging Face
Search models, datasets, users...
Models
Datasets
Spaces
Posts
Docs
Solutions
Pricing



Spaces:

bstraehle
/
openai-assistants


like
0

Logs
App
Files
Community
Settings
openai-assistants/
assistants.py

125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
import gradio as gr
    show_json("run", run)

    if hasattr(run, "last_error") and run.last_error:
        raise gr.Error(run.last_error)

    return run

def get_run_steps(thread, run):
    run_steps = openai_client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id,
        order="asc",
    )

    show_json("run_steps", run_steps)
    return run_steps

def execute_tool_call(tool_call):
    name = tool_call.function.name
    args = {}
    
    if len(tool_call.function.arguments) > 10:
        args_json = ""

        try:
            args_json = tool_call.function.arguments
            args = json.loads(args_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing function name '{name}' function args '{args_json}': {e}")

    return tools[name](**args)

def execute_tool_calls(run_steps):
    run_step_details = []
    tool_call_ids = []
    tool_call_results = []
    
    for step in run_steps.data:
        step_details = step.step_details
        run_step_details.append(step_details)
        show_json("step_details", step_details)
        
        if hasattr(step_details, "tool_calls"):
            for tool_call in step_details.tool_calls:
                show_json("tool_call", tool_call)

                if hasattr(tool_call, "function"):
                    gr.Info(f"Custom tool call: {tool_call}")
                    tool_call_ids.append(tool_call.id)
                    tool_call_results.append(execute_tool_call(tool_call))
                else:
                    gr.Info(f"Built-in tool call: {tool_call}")
    
    return tool_call_ids, tool_call_results

def recurse_execute_tool_calls(thread, run, run_steps, iteration):
    tool_call_ids, tool_call_results = execute_tool_calls(run_steps)
    
    if len(tool_call_ids) > iteration:
        tool_output = {}
        
        try:
            tool_output = {
                "tool_call_id": tool_call_ids[iteration],
                "output": tool_call_results[iteration].to_json()
            }
        except AttributeError:
            tool_output = {
                "tool_call_id": tool_call_ids[iteration],
                "output": tool_call_results[iteration]
            }
      
        # https://platform.openai.com/docs/api-reference/runs/submitToolOutputs
        run = openai_client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=[tool_output]
        )
    
        run = wait_on_run(thread, run)
        run_steps = get_run_steps(thread, run)
        recurse_execute_tool_calls(thread, run, run_steps, iteration + 1)
    else:
        return

def get_messages(thread):
    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id
    )
    
    show_json("messages", messages)
    return messages
                        
def extract_content_values(data):
Commit directly to the
main
branch
Open as a pull request to the
main
branch
Commit changes
Update assistants.py
Add an extended description...
Upload images, audio, and videos by dragging in the text input, pasting, or clicking here.
