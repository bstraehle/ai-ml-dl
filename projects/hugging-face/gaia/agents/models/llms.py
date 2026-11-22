# Pricing:

# https://platform.openai.com/docs/pricing
# https://ai.google.dev/gemini-api/docs/pricing
# https://claude.com/pricing#api

# CrewAI crew

LLM_CREW_PLANNING = "gpt-6.2-sol"

# CrewAI manager

LLM_MANAGER = "gpt-6.2-sol"

# CrewAI agents

LLM_AGENT = "gpt-6.2-sol"

# crewAI assumes 8,192 tokens for models it doesn't know (like gpt-6.2-sol), which
# makes it defer large tool results and repeatedly compact the conversation.
# 272K is OpenAI's standard-price input limit; above it input is billed at 2x.
CONTEXT_WINDOW_SIZE = 272_000

# CrewAI tools

LLM_WEB_SEARCH         = "gemini-3.1-pro-preview"
LLM_WEB_BROWSER        = "claude-sonnet-5-5"
LLM_IMAGE_ANALYSIS     = "gemini-3.1-pro-preview"
LLM_AUDIO_ANALYSIS     = "gemini-3.1-pro-preview"
LLM_VIDEO_ANALYSIS     = "gemini-3.1-pro-preview"
LLM_YOUTUBE_ANALYSIS   = "gemini-3.1-pro-preview"
LLM_DOCUMENT_ANALYSIS  = "gemini-3.1-pro-preview"
LLM_CODE_GENERATION    = "gemini-3.1-pro-preview"
LLM_CODE_EXECUTION     = "gemini-3.1-pro-preview"
LLM_IMAGE_TO_FEN       = "gemini-3.1-pro-preview"
LLM_ALGEBRAIC_NOTATION = "gemini-3.1-pro-preview"

THINKING_LEVEL_WEB_SEARCH         = "high"
THINKING_LEVEL_MEDIA_ANALYSIS     = "high"
THINKING_LEVEL_YOUTUBE_ANALYSIS   = "high"
THINKING_LEVEL_DOCUMENT_ANALYSIS  = "high"
THINKING_LEVEL_CODE_GENERATION    = "high"
THINKING_LEVEL_CODE_EXECUTION     = "low"
THINKING_LEVEL_IMAGE_TO_FEN       = "high"
THINKING_LEVEL_ALGEBRAIC_NOTATION = "high"

# Other tools

LLM_FINAL_ANSWER = "gemini-3.1-pro-preview"

THINKING_LEVEL_FINAL_ANSWER = "low"

# Fallback model (used when the primary Gemini model hits its daily rate limit)

LLM_FALLBACK = "gemini-3.8-flash"
