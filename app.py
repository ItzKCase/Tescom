import gradio as gr
from agent import chat_with_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    title="LangGraph Agent Chat",
    theme=gr.themes.Soft(primary_hue="slate").set(
        body_background_fill_dark="#0f0f23",
        background_fill_primary_dark="#1a1a3e",
        background_fill_secondary_dark="#252547",
        border_color_primary_dark="#4a4a6a",
        input_background_fill_dark="#1a1a3e",
        button_primary_background_fill_dark="#4338ca",
        button_primary_background_fill_hover_dark="#5b51d6"
    ),
    css="""
        html, body { height: 100%; }
        .gradio-container { min-height: 100vh; max-width: 900px; margin: 0 auto; display: flex; flex-direction: column; }
        /* Make the main app column stretch */
        .gradio-container > .gradio-block { display: flex; flex-direction: column; flex: 1 1 auto; }
        /* Dynamically size the chatbot to fill viewport minus header and input areas */
        #chatbot { height: calc(100vh - 260px) !important; }
        .dark { color-scheme: dark; }
    """
) as demo:
    
    gr.Markdown("""
    # 🤖 LangGraph Agent Chat
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
