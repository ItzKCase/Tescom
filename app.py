import gradio as gr
from agent import chat_with_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TESCOM_LOGO_URL = os.getenv("TESCOM_LOGO_URL", "")
TESCOM_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "tescom-logo.png")

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment variables.")
    print("Please set your OpenAI API key in a .env file or environment variable.")

def format_message(message):
    """Format a message for display in the chat interface."""
    if isinstance(message, HumanMessage):
        return f"**You:** {message.content}"
    elif isinstance(message, AIMessage):
        return f"**Agent:** {message.content}"
    else:
        return str(message)

def _truncate(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _convert_messages_to_gradio(messages):
    """Convert LangChain messages to Gradio Chatbot 'messages' format, hiding tool/system messages."""
    converted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            converted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            converted.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            # Hide tool results in user-facing chat
            continue
        elif isinstance(msg, SystemMessage):
            # Do not show full system prompt; optional minimal indicator
            continue
        else:
            converted.append({"role": "assistant", "content": _truncate(str(msg))})
    return converted


def chat_interface(message, history):
    """Handle chat interactions with the agent."""
    if not message.strip():
        return "", history
    
    try:
        # Convert Gradio history to LangChain message format
        conversation_history = []
        for msg in history:
            if isinstance(msg, dict):
                # New message format
                if msg.get("role") == "user":
                    conversation_history.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    conversation_history.append(AIMessage(content=msg["content"]))
            else:
                # Old tuple format (fallback)
                human_msg, ai_msg = msg
                if human_msg:
                    conversation_history.append(HumanMessage(content=human_msg))
                if ai_msg:
                    conversation_history.append(AIMessage(content=ai_msg))
        
        # Get response from agent
        response, updated_conversation = chat_with_agent(message, conversation_history)

        # Build full history with tool-result echoing
        new_history = _convert_messages_to_gradio(updated_conversation)

        return "", new_history
    
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        return "", new_history

def clear_chat():
    """Clear the chat history."""
    return [], ""

# Create the Gradio interface
with gr.Blocks(
    title="Tescom Capabilities Agent",
    theme=gr.themes.Soft(primary_hue="teal"),
    css="""
        :root {
            --tescom-teal: #00a3a3;
            --tescom-teal-600: #008f8f;
            --tescom-navy: #0f253e;
            --bg: #ffffff;
            --bg-soft: #f7f9fc;
            --text: #0f172a;
            --user-bubble: #e6f7f6; /* light teal */
            --assistant-bubble: var(--bg-soft);
        }
        html, body { height: 100%; background: var(--bg); color: var(--text); }
        .gradio-container { min-height: 100vh; max-width: 1000px; margin: 0 auto; display: flex; flex-direction: column; }
        .gradio-container a { color: var(--tescom-teal); }
        /* Stretch main column */
        .gradio-container > .gradio-block { display: flex; flex-direction: column; flex: 1 1 auto; }
        /* Dynamic chatbot height */
        #chatbot { height: calc(100vh - 240px) !important; }
        /* Brand header styling */
        .brand-header { display: flex; align-items: center; gap: 12px; padding: 8px 0 4px; }
        .brand-header img { height: 28px; }
        /* Buttons */
        button.svelte-1ipelgc, .gr-button-primary { background: var(--tescom-teal) !important; border-color: var(--tescom-teal) !important; }
        button.svelte-1ipelgc:hover, .gr-button-primary:hover { background: var(--tescom-teal-600) !important; border-color: var(--tescom-teal-600) !important; }
        /* Chat bubbles */
        #chatbot .message.user { background: var(--user-bubble) !important; color: var(--text) !important; }
        #chatbot .message.assistant { background: var(--assistant-bubble) !important; color: var(--text) !important; }
        /* Inputs */
        input, textarea { border-color: var(--tescom-teal) !important; }
    """
) as demo:
    
    # Brand header with local image support (assets/tescom-logo.png). Falls back to URL or text.
    has_local_logo = os.path.exists(TESCOM_LOGO_PATH)
    if has_local_logo:
        import base64
        with open(TESCOM_LOGO_PATH, "rb") as lf:
            logo_b64 = base64.b64encode(lf.read()).decode("ascii")
        with gr.Row(elem_classes=["brand-header"]):
            gr.HTML(f"<img src='data:image/png;base64,{logo_b64}' alt='Tescom Logo' style='height:40px;' />")
            gr.Markdown("## Tescom Capabilities Agent")
    elif TESCOM_LOGO_URL:
        gr.Markdown(f"""
        <div class="brand-header">
            <img src="{TESCOM_LOGO_URL}" alt="Tescom Logo" style="height:40px;" />
            <h2 style="margin:0; display:inline-block; padding-left:12px;">Tescom Capabilities Agent</h2>
        </div>
        """)
    else:
        gr.Markdown("""
        ## Tescom Capabilities Agent
        """)
    
    chatbot = gr.Chatbot(
        label="Chat History",
        show_label=False,
        container=True,
        type="messages",
        autoscroll=True,
        elem_id="chatbot",
    )
    
    msg = gr.Textbox(
        label="Your Message",
        placeholder="Type your message here...",
        show_label=False
    )
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", variant="secondary", scale=1)
    
    # Event handlers
    submit_btn.click(
        chat_interface,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        chat_interface,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    # Enable request queuing to avoid dropped/cancelled events for longer operations
    # Note: On Gradio 5.x, queue() may not accept concurrency args; use defaults for compatibility.
    demo.queue()
    # Launch the app
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
