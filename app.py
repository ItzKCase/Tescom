import gradio as gr
import logging
from agent import chat_with_agent, cleanup_resources
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import os
import atexit
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from equipment_processor import process_equipment_list, get_data_preview, add_temp_file, cleanup_all_temp_files

# Configure logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register cleanup function to run on exit
atexit.register(cleanup_resources)
atexit.register(cleanup_all_temp_files)

# Load environment variables
load_dotenv()
TESCOM_LOGO_URL = os.getenv("TESCOM_LOGO_URL", "")
TESCOM_LOGO_PATH = os.getenv("TESCOM_LOGO_PATH", os.path.join(os.path.dirname(__file__), "assets", "tescom-logo.png"))

# Check Gradio version for compatibility
try:
    import gradio as gr
    gradio_version = gr.__version__
    print(f"📦 Gradio version: {gradio_version}")
    
    # Check if we have access to newer components
    has_box = hasattr(gr, 'Box')
    has_column = hasattr(gr, 'Column')
    
    if not has_box:
        print("⚠️  gr.Box not available, using gr.Column as fallback")
    if not has_column:
        print("⚠️  gr.Column not available, using basic layout")
        
except ImportError as e:
    print(f"❌ Error importing Gradio: {e}")
    raise

def ensure_database_ready():
    """Ensure the equipment database is available before starting the app."""
    db_path = os.getenv('DB_PATH', 'equipment.db')
    excel_file = 'Tescom_new_list.xlsx'
    
    print(f"🔍 Checking database at: {db_path}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(db_path)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        print(f"📁 Ensured directory exists: {data_dir}")
    
    # Check if database exists and has data
    if os.path.exists(db_path):
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM equipment;")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                print(f"✅ Database ready: {count} equipment records")
                return True
            else:
                print("⚠️  Database exists but is empty")
        except Exception as e:
            print(f"⚠️  Database error: {e}")
    else:
        print("❌ Database not found")
    
    # Try to build database from Excel
    if os.path.exists(excel_file):
        print(f"🔨 Building database from {excel_file}...")
        try:
            result = subprocess.run([
                sys.executable, 'build_equipment_db.py',
                '--excel', excel_file,
                '--db', db_path,
                '--rebuild',
                '--report'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Database built successfully!")
                return True
            else:
                print(f"❌ Database build failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Database build error: {e}")
            return False
    else:
        print(f"❌ Excel file not found: {excel_file}")
        print("💡 The application will start but equipment lookups may not work.")
        return False

# Ensure database is ready before starting
print("🚀 Starting Tescom Capabilities Agent...")
ensure_database_ready()

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
        
        /* Equipment upload section styling */
        .equipment-upload {
            background: var(--bg-soft);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .equipment-upload h3 {
            margin-top: 0;
            margin-bottom: 16px;
            color: var(--tescom-navy);
        }
        
        /* Preview section styling */
        .preview-section {
            background: var(--bg-soft);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: var(--shadow-sm);
            animation: fadeIn 0.5s ease-in-out;
            min-height: 0;
        }
        
        .preview-section:empty {
            display: none;
        }
        
        .preview-section h4 {
            color: var(--tescom-navy);
            margin-bottom: 16px;
        }
        
        .preview-section table {
            font-size: 0.875rem;
        }
        
        /* Download section styling */
        .download-section {
            background: linear-gradient(135deg, #e6f7f6 0%, #f0fdfa 100%);
            border: 2px solid var(--tescom-teal);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: var(--shadow-md);
            animation: fadeIn 0.5s ease-in-out;
        }
        
        .download-section h3 {
            color: var(--tescom-teal);
            margin-bottom: 12px;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .gradio-container {
                max-width: 100%;
                padding: 0 16px;
            }
            
            #chatbot {
                height: calc(100vh - 440px) !important;
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
    
    # Equipment List Upload Section
    try:
        # Try to use gr.Column for better layout
        with gr.Column(elem_classes=["equipment-upload"]):
            gr.Markdown("### 📋 Equipment List Accreditation Assessment")
            gr.Markdown("Upload your equipment list (Excel or CSV) to get a comprehensive accreditation assessment. We'll analyze each item against our database and provide detailed results.")
            
            with gr.Row():
                file_input = gr.File(
                    label="Upload Equipment List",
                    file_types=[".xlsx", ".csv"],
                    file_count="single"
                )
            
            with gr.Row():
                preview_btn = gr.Button("👁️ Preview Data", variant="secondary", scale=1)
                process_btn = gr.Button("🔍 Process Equipment List", variant="primary", scale=1)
            
            status_display = gr.Textbox(
                label="Processing Status",
                value="Ready to process equipment list. Please upload a file.",
                interactive=False,
                max_lines=3
            )
            
            # Data preview section
            preview_section = gr.HTML(
                value="",
                visible=True,
                elem_classes=["preview-section"]
            )
            
            # Download section - will be populated after processing
            download_section = gr.Markdown(
                value="",
                visible=False,
                elem_classes=["download-section"]
            )
            download_btn = gr.File(
                label="📄 Download Assessment Results (Excel)", 
                visible=True,
                scale=1,
                interactive=False
            )
    except AttributeError:
        # Fallback to basic layout if gr.Column is not available
        gr.Markdown("### 📋 Equipment List Accreditation Assessment")
        gr.Markdown("Upload your equipment list (Excel or CSV) to get a comprehensive accreditation assessment. We'll analyze each item against our database and provide detailed results.")
        
        file_input = gr.File(
            label="Upload Equipment List",
            file_types=[".xlsx", ".csv"],
            file_count="single"
        )
        
        preview_btn = gr.Button("👁️ Preview Data", variant="secondary")
        process_btn = gr.Button("🔍 Process Equipment List", variant="primary")
        
        status_display = gr.Textbox(
            label="Processing Status",
            value="Ready to process equipment list. Please upload a file.",
            interactive=False,
            max_lines=3
        )
        
        # Data preview section
        preview_section = gr.HTML(
            value="",
            visible=True,
            elem_classes=["preview-section"]
        )
        
        # Download section - will be populated after processing
        download_section = gr.Markdown(
            value="",
            visible=False,
            elem_classes=["download-section"]
        )
        download_btn = gr.File(
            label="📄 Download Assessment Results (Excel)", 
            visible=True,
            interactive=False
        )
    
    # Chat Interface
    gr.Markdown("### 💬 Chat with AI Agent")
    gr.Markdown("Ask questions about specific equipment, accreditation capabilities, or get help with your equipment list.")
    
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
    
    # Function to handle file preview
    def preview_equipment_file(file):
        if file is None:
            return "", "❌ Please select a file to upload."
        
        try:
            file_path = file.name
            preview_html, error_message = get_data_preview(file_path)
            
            if preview_html:
                return preview_html, "✅ File preview generated successfully. Review your data above, then click 'Process Equipment List' to continue."
            else:
                return "", error_message
        except Exception as e:
            logger.error(f"Error previewing equipment file: {e}")
            return "", f"❌ Error previewing file: {str(e)}"
    
    # Function to handle equipment list processing
    def process_equipment_file(file):
        if file is None:
            return None, "", "❌ Please select a file to upload."
        
        try:
            file_path = file.name
            result_file, status_message = process_equipment_list(file_path)
            
            if result_file:
                # Add to cleanup list
                add_temp_file(result_file)
                
                # Show download section and return the file
                download_html = """
                ### 📥 Assessment Results Ready!
                
                Your equipment accreditation assessment is complete. Click the download button below to get your results.
                
                **Results Summary:**
                - ✅ Processing completed successfully
                - 📊 Equipment items analyzed
                - 📄 Excel report generated
                """
                return result_file, download_html, status_message
            else:
                return None, "", status_message
        except Exception as e:
            logger.error(f"Error processing equipment file: {e}")
            return None, "", f"❌ Error processing file: {str(e)}"
    
    # Event handlers
    try:
        preview_btn.click(
            preview_equipment_file,
            inputs=[file_input],
            outputs=[preview_section, status_display]
        )
        
        process_btn.click(
            process_equipment_file,
            inputs=[file_input],
            outputs=[download_btn, download_section, status_display]
        )
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
    except Exception as e:
        logger.error(f"Error setting up event handlers: {e}")
        print(f"⚠️  Some event handlers may not work: {e}")

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
