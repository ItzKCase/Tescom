# 🤖 LangGraph Agent with Gradio Web Interface

A conversational AI agent built with LangGraph and LangChain, featuring a beautiful Gradio web interface powered by GPT-4o mini.

## ✨ Features

- **LangGraph Agent**: Built with modern LangGraph for complex conversation flows
- **GPT-4o mini Integration**: Powered by OpenAI's latest model
- **Beautiful Web UI**: Modern Gradio interface with responsive design
- **Tool Integration**: Built-in tools for enhanced functionality
- **Context Awareness**: Maintains conversation history and context
- **Easy Setup**: Simple installation and configuration

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key.

### 3. Optional: Add Tescom Logo

Place a logo image at `assets/tescom-logo.png` to display it in the header. PNG at ~40px height works well.

Alternatively, set an external URL:

```bash
TESCOM_LOGO_URL=https://example.com/path/to/tescom-logo.png
```

### 4. Run the Application

```bash
python app.py
```

The web interface will be available at `http://localhost:7860`

## 📁 Project Structure

```
Tescom/
├── agent.py          # LangGraph agent implementation
├── app.py            # Gradio web interface
├── requirements.txt  # Python dependencies
├── env_template.txt  # Environment variables template
├── README.md         # This file
└── assets/
    └── tescom-logo.png  # Optional local logo image used in the header
```

## 🛠️ How It Works

### Agent Architecture

The agent is built using LangGraph's state-based workflow:

1. **State Management**: Maintains conversation context and history
2. **Tool Integration**: Can use various tools (currently includes time tool)
3. **LLM Processing**: Uses GPT-4o mini for natural language understanding
4. **Response Generation**: Generates contextual and helpful responses

### Web Interface

The Gradio interface provides:

- **Chat Interface**: Clean, modern chat UI
- **Message History**: Persistent conversation memory
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: Graceful error handling and user feedback

## 🔧 Customization

### Adding New Tools

To add new tools to your agent, modify `agent.py`:

```python
@tool
def your_custom_tool() -> str:
    """Description of what your tool does."""
    # Your tool implementation
    return "Tool result"

# Add to tools list
tools = [get_current_time, your_custom_tool]
```

### Modifying the Agent Logic

The main agent logic is in the `agent_function` in `agent.py`. You can:

- Add conditional logic based on message content
- Implement multi-step workflows
- Add memory or external data sources
- Customize response generation

### Styling the Web Interface

Modify the CSS in `app.py` to customize the appearance:

```python
css="""
    .gradio-container {
        max-width: 800px;
        margin: 0 auto;
        background-color: #f5f5f5;
    }
"""
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `.env` file contains the correct OpenAI API key
2. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
3. **Port Already in Use**: Change the port in `app.py` if 7860 is occupied

### Getting Help

- Check that all dependencies are properly installed
- Verify your OpenAI API key is valid and has sufficient credits
- Ensure you're using Python 3.8+ for compatibility

## 📚 Dependencies

- **LangGraph**: Modern agent workflow framework
- **LangChain**: LLM application framework
- **LangChain OpenAI**: OpenAI integration
- **Gradio**: Web interface framework
- **Python-dotenv**: Environment variable management

## 🔮 Future Enhancements

Potential improvements you could add:

- **Memory Persistence**: Save conversations to database
- **Multi-modal Support**: Handle images, documents, etc.
- **Advanced Tools**: Web search, file operations, API calls
- **User Authentication**: Multi-user support
- **Analytics**: Conversation insights and metrics
- **Custom Models**: Support for other LLM providers

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project!

---

**Happy chatting with your AI agent! 🎉**
