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
from equipment_processor import (process_equipment_list, get_data_preview, add_temp_file, 
                                cleanup_all_temp_files)
import time
import hashlib

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

# Generate cache-busting version identifier
CACHE_BUST_VERSION = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

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

# Ensure database is ready and initialize performance optimizations
print("🚀 Starting Tescom Capabilities Agent...")
ensure_database_ready()

# Initialize performance optimizations
def initialize_performance_optimizations():
    """Initialize performance optimizations and caches."""
    try:
        from agent import warm_essential_caches, connection_pool
        
        # Warm up caches for better initial performance
        print("🔥 Warming up caches...")
        warm_essential_caches()
        
        print("✅ Performance optimizations initialized")
        
    except Exception as e:
        print(f"⚠️ Warning: Some performance optimizations failed to initialize: {e}")

# Initialize optimizations
initialize_performance_optimizations()

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
    """Get current cache and system status for monitoring."""
    try:
        from agent import (equipment_data_cache, search_results_cache, search_circuit_breaker, 
                          openai_circuit_breaker, memory_monitor, connection_pool)
        
        equipment_size = equipment_data_cache.size()
        search_size = search_results_cache.size()
        
        # Circuit breaker status
        search_status = "🟢" if search_circuit_breaker.state == "CLOSED" else "🔴" if search_circuit_breaker.state == "OPEN" else "🟡"
        openai_status = "🟢" if openai_circuit_breaker.state == "CLOSED" else "🔴" if openai_circuit_breaker.state == "OPEN" else "🟡"
        
        # Memory status
        memory_status = memory_monitor.check_memory()
        memory_icon = "🟢" if memory_status["status"] == "normal" else "🟡" if memory_status["status"] == "warning" else "🔴"
        memory_mb = memory_status["usage"]["rss_mb"]
        
        # Connection pool status
        pool_status = "🟢" if connection_pool and hasattr(connection_pool, 'connections') else "🔴"
        pool_size = len(connection_pool.connections) if connection_pool and hasattr(connection_pool, 'connections') else 0
        
        return f"📊 Cache: Eq({equipment_size}), Search({search_size}) | Memory: {memory_icon}{memory_mb:.0f}MB | Pool: {pool_status}({pool_size}) | CB: Search{search_status}, OpenAI{openai_status} | Enhanced ⚡"
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return "📊 Status: System monitoring active"

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
        
        /* Progress section styling */
        .progress-section {
            background: linear-gradient(135deg, #e6f7ff 0%, #f0f9ff 100%);
            border: 2px solid var(--tescom-teal);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: var(--shadow-md);
            animation: pulseProgress 2s ease-in-out infinite;
        }
        
        .progress-section h3 {
            color: var(--tescom-teal);
            margin-bottom: 12px;
            font-size: 1.1rem;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(0, 163, 163, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--tescom-teal) 0%, var(--tescom-teal-600) 100%);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--tescom-navy);
            margin: 5px 0;
        }
        
        @keyframes pulseProgress {
            0%, 100% { box-shadow: var(--shadow-md); }
            50% { box-shadow: 0 6px 12px -2px rgba(0, 163, 163, 0.3), 0 4px 6px -3px rgba(0, 163, 163, 0.2); }
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
        
        /* Enhanced download button styling */
        .download-button {
            background: linear-gradient(135deg, var(--tescom-teal) 0%, var(--tescom-teal-600) 100%) !important;
            border: 3px solid var(--tescom-teal) !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 25px rgba(0, 163, 163, 0.3), 0 4px 10px rgba(0, 0, 0, 0.1) !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 16px 24px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .download-button:hover {
            background: linear-gradient(135deg, var(--tescom-teal-600) 0%, #007a7a 100%) !important;
            border-color: #007a7a !important;
            box-shadow: 0 12px 35px rgba(0, 163, 163, 0.4), 0 6px 15px rgba(0, 0, 0, 0.15) !important;
            transform: translateY(-3px) !important;
        }
        
        .download-button:active {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(0, 163, 163, 0.3) !important;
        }
        
        /* Download button glow effect */
        .download-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .download-button:hover::before {
            left: 100%;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Side-by-side layout adjustments */
        .gr-row {
            gap: 16px;
        }
        
        .gr-column {
            gap: 16px;
        }
        
        /* Chat interface height adjustment for side-by-side layout */
        #chatbot {
            height: 500px !important;
            min-height: 400px;
        }
        
        /* Equipment upload section height matching */
        .equipment-upload {
            min-height: 500px;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .gradio-container {
                max-width: 100%;
                padding: 0 16px;
            }
            
            /* Stack vertically on mobile */
            .gr-row {
                flex-direction: column;
            }
            
            #chatbot {
                height: 300px !important;
            }
            
            .equipment-upload {
                min-height: auto;
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
    
        # Main interface layout - Chat on left, File upload on right
    with gr.Row():
        # Left Column - Chat Interface
        with gr.Column(scale=1):
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
        
        # Right Column - Equipment List Upload Section
        with gr.Column(scale=1, elem_classes=["equipment-upload"]):
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
            
            # Download section - will be populated after processing
            download_section = gr.Markdown(
                value="",
                visible=False,
                elem_classes=["download-section"]
            )
            download_btn = gr.DownloadButton(
                label="📄 Download Assessment Results (Excel)", 
                visible=False,
                scale=1,
                variant="primary",
                elem_classes=["download-button"]
            )
            
            # Data preview section
            preview_section = gr.HTML(
                value="",
                visible=True,
                elem_classes=["preview-section"]
            )
    
    # Cache status display below the main interface
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
    
    # Function to handle equipment list processing with progress tracking
    def process_equipment_file(file, progress=gr.Progress()):
        if file is None:
            return None, "", "❌ Please select a file to upload."
        
        try:
            file_path = file.name
            
            # Start progress tracking
            progress(0, desc="Starting file processing...")
            
            # Process the equipment list with progress updates
            result_file, status_message = process_equipment_list(file_path)
            
            # Update progress to 100% when complete
            progress(1.0, desc="Processing complete!")
            
            if result_file:
                # Add to cleanup list
                add_temp_file(result_file)
                
                # Log the result file path for debugging
                logger.info(f"Processing complete. Result file: {result_file}")
                logger.info(f"File exists: {os.path.exists(result_file)}")
                logger.info(f"File size: {os.path.getsize(result_file) if os.path.exists(result_file) else 'N/A'}")
                
                # Show download section with completion summary
                download_html = """
                ### 📥 Assessment Results Ready!
                
                Your equipment accreditation assessment is complete. Click the download button below to get your results.
                
                **Processing Summary:**
                - ✅ Equipment items processed successfully
                - 📄 Excel report generated
                - ⚡ Processed with optimized performance
                """
                
                # Configure download button with the result file
                # For Gradio 5.x, we need to return the file path and make button visible using gr.update()
                
                # Return the file path for the download button
                # The gr.DownloadButton will use this file for downloads
                # Ensure the file path is absolute for proper handling
                abs_result_file = os.path.abspath(result_file)
                logger.info(f"Absolute result file path: {abs_result_file}")
                
                # For Gradio to properly handle file downloads, we need to ensure the file is accessible
                # and the path is properly formatted
                if os.path.exists(abs_result_file):
                    file_size = os.path.getsize(abs_result_file)
                    logger.info(f"File ready for download: {abs_result_file} (size: {file_size} bytes)")
                    
                    # The gr.DownloadButton should automatically use this file for downloads
                    # In Gradio 5.x, use gr.update() to make the button visible and set the file
                    download_update = gr.update(
                        visible=True,
                        value=abs_result_file,
                        label=f"📄 Download Assessment Results ({os.path.basename(abs_result_file)})"
                    )
                    return download_update, download_html, status_message
                else:
                    logger.error(f"File not found for download: {abs_result_file}")
                    download_update = gr.update(visible=False)
                    return download_update, download_html, "❌ Error: Result file not found"
            else:
                # No result file, keep download button hidden
                download_update = gr.update(visible=False)
                return download_update, "", status_message
                
        except Exception as e:
            error_msg = f"❌ Error processing file: {str(e)}"
            logger.error(error_msg)
            # Keep download button hidden on error
            download_update = gr.update(visible=False)
            return download_update, "", error_msg
    

    
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
    # Launch with Gradio's built-in server
    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),  # Allow external connections in Docker
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        share=False,
        show_error=True,
        # Add cache-busting to prevent browser caching issues
        favicon_path=None,
        ssl_verify=False,
        # Enable auto-reload in development to help with caching issues
        debug=os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
    )
