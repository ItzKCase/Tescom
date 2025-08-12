import gradio as gr
import logging
from agent import chat_with_agent, cleanup_resources
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import os
import atexit
from dotenv import load_dotenv

# Configure logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register cleanup function to run on exit
atexit.register(cleanup_resources)

# Load environment variables
load_dotenv()
TESCOM_LOGO_URL = os.getenv("TESCOM_LOGO_URL", "")
TESCOM_LOGO_PATH = os.getenv("TESCOM_LOGO_PATH", os.path.join(os.path.dirname(__file__), "assets", "tescom-logo.png"))

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
        content = ""
        if isinstance(msg, HumanMessage):
            content = msg.content.strip() if msg.content else ""
            if content:  # Only add non-empty messages
                converted.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            content = msg.content.strip() if msg.content else ""
            if content:  # Only add non-empty messages
                converted.append({"role": "assistant", "content": content})
        elif isinstance(msg, ToolMessage):
            # Hide tool results in user-facing chat
            continue
        elif isinstance(msg, SystemMessage):
            # Do not show full system prompt; optional minimal indicator
            continue
        else:
            content = _truncate(str(msg)).strip()
            if content:  # Only add non-empty messages
                converted.append({"role": "assistant", "content": content})
    return converted


def chat_interface(message, history):
    """Handle chat interactions with the agent."""
    if not message.strip():
        logger.warning("Empty message received")
        return "", history
    
    logger.info(f"Processing chat message: {message[:50]}...")
    
    try:
        # Convert Gradio history to LangChain message format
        conversation_history = []
        for msg in history:
            if isinstance(msg, dict):
                # New message format
                if msg.get("role") == "user" and msg.get("content", "").strip():
                    conversation_history.append(HumanMessage(content=msg["content"].strip()))
                elif msg.get("role") == "assistant" and msg.get("content", "").strip():
                    conversation_history.append(AIMessage(content=msg["content"].strip()))
            else:
                # Old tuple format (fallback)
                human_msg, ai_msg = msg
                if human_msg and human_msg.strip():
                    conversation_history.append(HumanMessage(content=human_msg.strip()))
                if ai_msg and ai_msg.strip():
                    conversation_history.append(AIMessage(content=ai_msg.strip()))
        
        # Get response from agent
        response, updated_conversation = chat_with_agent(message, conversation_history)

        # Build full history, filtering out empty messages
        new_history = _convert_messages_to_gradio(updated_conversation)

        logger.info(f"Chat response generated successfully")
        return "", new_history
    
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        error_msg = f"❌ An error occurred while processing your request: {str(e)}"
        new_history = history + [
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": error_msg}
        ]
        return "", new_history

def clear_chat():
    """Clear the chat history."""
    logger.info("Chat history cleared")
    return [], ""

def get_cache_status():
    """Get current cache status for monitoring."""
    try:
        from agent import equipment_data_cache, search_results_cache, search_circuit_breaker, openai_circuit_breaker
        equipment_size = equipment_data_cache.size()
        search_size = search_results_cache.size()
        
        # Circuit breaker status
        search_status = "🟢" if search_circuit_breaker.state == "CLOSED" else "🔴" if search_circuit_breaker.state == "OPEN" else "🟡"
        openai_status = "🟢" if openai_circuit_breaker.state == "CLOSED" else "🔴" if openai_circuit_breaker.state == "OPEN" else "🟡"
        
        return f"📊 Cache: Equipment ({equipment_size}), Search ({search_size}) | Circuit Breakers: Search {search_status}, OpenAI {openai_status} | Phase 2 ✅"
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return "📊 Status: Unable to retrieve system information"

# Create the Gradio interface
with gr.Blocks(
    title="Tescom Capabilities Agent",
    theme=gr.themes.Soft(primary_hue="teal"),
    css="""
        /* Import professional fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --tescom-teal: #00a3a3;
            --tescom-teal-600: #008f8f;
            --tescom-navy: #0f253e;
            --bg: #ffffff;
            --bg-soft: #f7f9fc;
            --text: #0f172a;
            --text-secondary: #64748b;
            --user-bubble: #e6f7f6; /* light teal */
            --assistant-bubble: var(--bg-soft);
            --border-light: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        
        /* Global typography */
        html, body { 
            height: 100%; 
            background: var(--bg); 
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
            line-height: 1.5;
        }
        
        .gradio-container { 
            min-height: 100vh; 
            max-width: 1000px; 
            margin: 0 auto; 
            display: flex; 
            flex-direction: column;
            font-family: 'Inter', sans-serif !important;
        }
        
        .gradio-container a { color: var(--tescom-teal); }
        
        /* Stretch main column */
        .gradio-container > .gradio-block { display: flex; flex-direction: column; flex: 1 1 auto; }
        
        /* Dynamic chatbot height */
        #chatbot { height: calc(100vh - 240px) !important; }
        
        /* Brand header styling */
        .brand-header { 
            display: flex; 
            align-items: center; 
            gap: 12px; 
            padding: 8px 0 4px;
        }
        .brand-header img { height: 28px; }
        .brand-header h2 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600;
            font-size: 1.5rem;
            color: var(--tescom-navy);
            margin: 0;
        }
        
        /* Typography improvements */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600;
            color: var(--tescom-navy);
            line-height: 1.25;
        }
        
        p, span, div {
            font-family: 'Inter', sans-serif !important;
            line-height: 1.6;
        }
        
        /* Chat message styling */
        #chatbot .message {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.95rem;
            line-height: 1.6;
            border-radius: 12px;
            border: 1px solid var(--border-light);
            box-shadow: var(--shadow-sm);
            margin: 8px 0;
        }
        
        #chatbot .message.user { 
            background: var(--user-bubble) !important; 
            color: var(--text) !important;
            border-color: var(--tescom-teal);
        }
        
        #chatbot .message.assistant { 
            background: var(--assistant-bubble) !important; 
            color: var(--text) !important;
        }
        
        /* Code blocks in messages */
        #chatbot .message code {
            font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace !important;
            background: rgba(0, 0, 0, 0.05);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        
        #chatbot .message pre {
            font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace !important;
            background: rgba(0, 0, 0, 0.05);
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85rem;
            line-height: 1.4;
        }
        
        /* Buttons */
        button.svelte-1ipelgc, .gr-button-primary { 
            background: var(--tescom-teal) !important; 
            border-color: var(--tescom-teal) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500;
            font-size: 0.875rem;
            border-radius: 8px;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }
        
        button.svelte-1ipelgc:hover, .gr-button-primary:hover { 
            background: var(--tescom-teal-600) !important; 
            border-color: var(--tescom-teal-600) !important;
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }
        
        /* Secondary buttons */
        .gr-button-secondary {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500;
            font-size: 0.875rem;
            border-radius: 8px;
            border: 1px solid var(--border-light);
            background: white;
            color: var(--text);
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }
        
        .gr-button-secondary:hover {
            background: var(--bg-soft);
            border-color: var(--tescom-teal);
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }
        
        /* Input fields */
        input, textarea { 
            border-color: var(--border-light) !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.925rem;
            border-radius: 8px;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-sm);
        }
        
        input:focus, textarea:focus {
            border-color: var(--tescom-teal) !important;
            box-shadow: 0 0 0 3px rgba(0, 163, 163, 0.1);
        }
        
        /* Labels */
        label {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        /* Status and system info */
        .gr-textbox[readonly] {
            background: var(--bg-soft);
            border-color: var(--border-light);
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        /* Improved spacing and layout */
        .gr-row {
            gap: 8px;
        }
        
        .gr-column {
            gap: 12px;
        }
        
        /* Loading states */
        .loading {
            opacity: 0.7;
            pointer-events: none;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .gradio-container {
                max-width: 100%;
                padding: 0 16px;
            }
            
            #chatbot {
                height: calc(100vh - 280px) !important;
            }
            
            .brand-header h2 {
                font-size: 1.25rem;
            }
        }
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
            gr.Markdown("## Tescom Capabilities AI Agent")
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
    
    # Cache status display
    cache_status = gr.Textbox(
        value=get_cache_status(),
        label="System Status",
        interactive=False,
        max_lines=1
    )
    
    # Function to handle chat and update cache status
    def chat_with_status_update(message, history):
        result_msg, result_history = chat_interface(message, history)
        status = get_cache_status()
        return result_msg, result_history, status
    
    def clear_with_status_update():
        history, msg = clear_chat()
        status = get_cache_status()
        return history, msg, status
    
    # Event handlers
    submit_btn.click(
        chat_with_status_update,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, cache_status]
    )
    
    msg.submit(
        chat_with_status_update,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, cache_status]
    )
    
    clear_btn.click(
        clear_with_status_update,
        outputs=[chatbot, msg, cache_status]
    )

if __name__ == "__main__":
    # Enable request queuing to avoid dropped/cancelled events for longer operations
    # Note: On Gradio 5.x, queue() may not accept concurrency args; use defaults for compatibility.
    demo.queue()
    # Launch the app - Docker-friendly configuration
    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),  # Allow external connections in Docker
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        share=False,
        show_error=True
    )
