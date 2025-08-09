import os
import json
import re
from typing import Dict, Any, TypedDict, List, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Define a simple tool for demonstration
@tool
def get_current_time() -> str:
    """Get the current time and date."""
    from datetime import datetime
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Define Google search tool using Serper API
@tool
def google_search(query: str) -> str:
    """
    Search Google for information about a query using Serper API.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as a formatted string
    """
    import requests
    import json
    
    # Get Serper API key from environment
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        return "Error: SERPER_API_KEY not found in environment variables. Please add your Serper API key to the .env file."
    
    url = "https://google.serper.dev/search"
    
    payload = json.dumps({
        "q": query,
        "num": 5  # Return top 5 results
    })
    
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Format the search results
        results = []
        
        # Add organic results
        if 'organic' in data:
            results.append("🔍 Search Results:")
            for i, result in enumerate(data['organic'][:5], 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No description')
                link = result.get('link', '')
                results.append(f"\n{i}. **{title}**")
                results.append(f"   {snippet}")
                if link:
                    results.append(f"   🔗 {link}")
        
        # Add answer box if available
        if 'answerBox' in data:
            answer = data['answerBox']
            if 'answer' in answer:
                results.insert(0, f"💡 Quick Answer: {answer['answer']}\n")
            elif 'snippet' in answer:
                results.insert(0, f"💡 Quick Answer: {answer['snippet']}\n")
        
        if not results:
            return f"No search results found for: {query}"
            
        return "\n".join(results)
        
    except requests.exceptions.RequestException as e:
        return f"Error performing search: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error parsing search response: {str(e)}"
    except Exception as e:
        return f"Unexpected error during search: {str(e)}"

############################
# Equipment capability tools
############################

# Internal caches for equipment data
_EQUIP_DATA: Optional[List[Dict[str, Any]]] = None
_ALIAS_TO_MANUFACTURER: Dict[str, str] = {}
_MANUFACTURER_TO_MODELS: Dict[str, List[str]] = {}
_MODEL_TO_RECORDS: Dict[str, List[Dict[str, Any]]] = {}
_ALIAS_MAX_WORDS: int = 1


def _normalize_text(value: str) -> str:
    return value.strip().lower()


def _normalize_for_alias(value: str) -> str:
    # Lowercase, replace non-alphanumeric with spaces, collapse multiple spaces
    lowered = value.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _load_equipment_data() -> None:
    global _EQUIP_DATA, _ALIAS_TO_MANUFACTURER, _MANUFACTURER_TO_MODELS, _MODEL_TO_RECORDS
    if _EQUIP_DATA is not None:
        return

    data_path = os.path.join(os.path.dirname(__file__), "tescom_equip_list.json")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            _EQUIP_DATA = json.load(f)
    except Exception as e:
        # If loading fails, keep caches empty but allow graceful degradation
        _EQUIP_DATA = []
        _ALIAS_TO_MANUFACTURER = {}
        _MANUFACTURER_TO_MODELS = {}
        _MODEL_TO_RECORDS = {}
        return

    alias_map: Dict[str, str] = {}
    manu_to_models: Dict[str, List[str]] = {}
    model_to_records: Dict[str, List[Dict[str, Any]]] = {}
    alias_max_words = 1

    for rec in _EQUIP_DATA:
        manufacturer = rec.get("manufacturer", "").strip()
        model_num = rec.get("model_num", "").strip()
        aliases = rec.get("aliases", []) or []

        if not manufacturer or not model_num:
            continue

        # Map aliases (including the manufacturer itself) to canonical manufacturer
        alias_map[_normalize_for_alias(manufacturer)] = manufacturer
        alias_max_words = max(alias_max_words, len(_normalize_for_alias(manufacturer).split()))
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                norm_alias = _normalize_for_alias(alias)
                alias_map[norm_alias] = manufacturer
                alias_max_words = max(alias_max_words, len(norm_alias.split()))

        # Manufacturer -> models
        manu_to_models.setdefault(manufacturer, [])
        manu_to_models[manufacturer].append(model_num)

        # Model -> records (case-insensitive match later)
        key = _normalize_text(model_num)
        model_to_records.setdefault(key, [])
        model_to_records[key].append(rec)

    _ALIAS_TO_MANUFACTURER = alias_map
    _MANUFACTURER_TO_MODELS = manu_to_models
    _MODEL_TO_RECORDS = model_to_records
    global _ALIAS_MAX_WORDS
    _ALIAS_MAX_WORDS = alias_max_words


def _invalidate_equipment_cache() -> None:
    global _EQUIP_DATA, _ALIAS_TO_MANUFACTURER, _MANUFACTURER_TO_MODELS, _MODEL_TO_RECORDS
    _EQUIP_DATA = None
    _ALIAS_TO_MANUFACTURER = {}
    _MANUFACTURER_TO_MODELS = {}
    _MODEL_TO_RECORDS = {}


def _find_manufacturer_in_text(text: str) -> Optional[str]:
    _load_equipment_data()
    if not _ALIAS_TO_MANUFACTURER:
        return None

    # Generate n-grams from the text and look up directly in alias map
    norm_text = _normalize_for_alias(text)
    if not norm_text:
        return None
    tokens = norm_text.split()
    max_n = min(_ALIAS_MAX_WORDS, len(tokens))

    # Try longer n-grams first for specificity
    for n in range(max_n, 0, -1):
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i+n])
            if ngram in _ALIAS_TO_MANUFACTURER:
                return _ALIAS_TO_MANUFACTURER[ngram]
    return None


def _find_model_for_manufacturer(text: str, manufacturer: str) -> Optional[str]:
    _load_equipment_data()
    models = _MANUFACTURER_TO_MODELS.get(manufacturer, [])
    if not models:
        return None
    # Prefer longest model strings first
    for model in sorted(models, key=len, reverse=True):
        if not model:
            continue
        if re.search(re.escape(model), text, flags=re.IGNORECASE):
            return model
    return None


def _find_model_any_manufacturer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (manufacturer, model) if a model appears in text without known manufacturer."""
    _load_equipment_data()
    if not _MODEL_TO_RECORDS:
        return None, None

    # Extract candidate tokens that look like models
    candidates = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\/_]{1,}\b", text):
        candidates.add(_normalize_text(token))

    # Try longest candidates first
    for cand in sorted(candidates, key=len, reverse=True):
        if cand in _MODEL_TO_RECORDS:
            recs = _MODEL_TO_RECORDS[cand]
            if recs:
                return recs[0].get("manufacturer"), recs[0].get("model_num")
    return None, None


@tool
def parse_equipment_details(user_text: str) -> str:
    """Parse manufacturer and model from a free-text equipment description. Returns a JSON string with keys: manufacturer, model, strategy."""
    try:
        _load_equipment_data()
        manufacturer = _find_manufacturer_in_text(user_text)
        model: Optional[str] = None
        strategy = ""

        if manufacturer:
            model = _find_model_for_manufacturer(user_text, manufacturer)
            strategy = "manufacturer_first"
        if not model:
            # Try model-first discovery
            manufacturer2, model2 = _find_model_any_manufacturer(user_text)
            if model2:
                manufacturer = manufacturer or manufacturer2
                model = model2
                strategy = "model_first"

        result = {
            "manufacturer": manufacturer,
            "model": model,
            "strategy": strategy or "none",
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def check_lab_capability(manufacturer: str, model: str) -> str:
    """Compare a manufacturer and model against Tescom's equipment list. Returns a formatted summary string of matches including gage description and cert level."""
    try:
        _load_equipment_data()
        if not manufacturer and not model:
            return "No manufacturer or model provided."

        # Find canonical manufacturer via alias map if input is an alias
        canonical_manu = _ALIAS_TO_MANUFACTURER.get(_normalize_text(manufacturer), manufacturer)

        matches: List[Dict[str, Any]] = []
        if model:
            recs = _MODEL_TO_RECORDS.get(_normalize_text(model), [])
            if canonical_manu:
                matches = [r for r in recs if r.get("manufacturer") == canonical_manu]
            else:
                matches = recs

        # If no exact model match, try partial match within the manufacturer's models
        if not matches and canonical_manu:
            models = _MANUFACTURER_TO_MODELS.get(canonical_manu, [])
            for m in models:
                if model and re.search(re.escape(model), m, flags=re.IGNORECASE):
                    matches.extend(_MODEL_TO_RECORDS.get(_normalize_text(m), []))

        if not matches:
            base = f"No exact match found for manufacturer='{manufacturer}' model='{model}'."
            if canonical_manu and canonical_manu != manufacturer:
                base += f" Interpreted manufacturer as '{canonical_manu}'."
            return base

        # Format results (deduplicate by model+desc)
        seen = set()
        lines = [
            "Matches found:",
        ]
        for rec in matches:
            key = (rec.get("manufacturer"), rec.get("model_num"), rec.get("gage_descr"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {rec.get('manufacturer')} {rec.get('model_num')} — {rec.get('gage_descr')} (Cert: {rec.get('cert_type')})"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error checking capability: {str(e)}"


############################
# Update agent tools (JSON editor)
############################

def _atomic_write_json(path: str, data: Any) -> None:
    import tempfile
    import shutil
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=directory, encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = tmp.name
    shutil.move(temp_path, path)


def _backup_json(path: str) -> str:
    from datetime import datetime
    base_dir = os.path.dirname(path)
    backups_dir = os.path.join(base_dir, "backups")
    os.makedirs(backups_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = os.path.join(backups_dir, f"tescom_equip_list.{ts}.json")
    with open(path, "r", encoding="utf-8") as fsrc, open(backup_path, "w", encoding="utf-8") as fdst:
        fdst.write(fsrc.read())
    return backup_path


def _append_change_log(entry: Dict[str, Any]) -> None:
    from datetime import datetime
    entry = dict(entry)
    entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
    log_path = os.path.join(os.path.dirname(__file__), "capability_changes.log")
    line = json.dumps(entry, ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@tool
def parse_update_request(request_text: str) -> str:
    """Parse a free-text capability update like 'we can calibrate Tektronix TDS3012 at Level 1'.
    Returns JSON: {manufacturer, model, new_level}. Level normalized to 'LEVEL 1|2|3'.
    """
    try:
        # Extract level
        level_match = re.search(r"level\s*([123])", request_text, flags=re.IGNORECASE)
        level = None
        if level_match:
            level = f"LEVEL {level_match.group(1)}"

        # Reuse existing parsing for manufacturer/model
        parsed = json.loads(parse_equipment_details.invoke({"user_text": request_text}))
        manufacturer = parsed.get("manufacturer")
        model = parsed.get("model")

        return json.dumps({
            "manufacturer": manufacturer,
            "model": model,
            "new_level": level,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def apply_update(manufacturer: str, model: str, new_level: str, user_name: str, password: str) -> str:
    """Apply a certification level update to tescom_equip_list.json after password check.
    Password is 'bigbrain' (case-insensitive). Returns a concise summary of changes or an error.
    """
    try:
        if not password or password.lower().strip() != "bigbrain":
            return "Error: Invalid password."
        if not new_level or new_level.upper().strip() not in {"LEVEL 1", "LEVEL 2", "LEVEL 3"}:
            return "Error: new_level must be one of LEVEL 1, LEVEL 2, LEVEL 3."
        if not manufacturer or not model:
            return "Error: manufacturer and model are required."

        _load_equipment_data()
        data_path = os.path.join(os.path.dirname(__file__), "tescom_equip_list.json")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        matches = []
        for rec in data:
            if rec.get("manufacturer") == manufacturer and _normalize_text(rec.get("model_num", "")) == _normalize_text(model):
                matches.append(rec)

        if not matches:
            return f"No records found for {manufacturer} {model}."

        backup_path = _backup_json(data_path)

        changed = 0
        before_after = []
        for rec in matches:
            old = rec.get("cert_type")
            if old != new_level:
                before_after.append({
                    "manufacturer": rec.get("manufacturer"),
                    "model_num": rec.get("model_num"),
                    "gage_descr": rec.get("gage_descr"),
                    "old_level": old,
                    "new_level": new_level,
                })
                rec["cert_type"] = new_level
                changed += 1

        if changed == 0:
            return f"No changes applied. Certification level already set to {new_level} for {manufacturer} {model}."

        _atomic_write_json(data_path, data)
        _append_change_log({
            "action": "update_cert_level",
            "user": user_name or "unknown",
            "manufacturer": manufacturer,
            "model": model,
            "changes": before_after,
            "backup": backup_path,
        })

        # Invalidate caches so next lookup reflects changes
        _invalidate_equipment_cache()

        return f"Updated {changed} record(s) for {manufacturer} {model} to {new_level}. Backup: {os.path.basename(backup_path)}"
    except Exception as e:
        return f"Error applying update: {str(e)}"


# Initialize the LLM with tools
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=1,
    timeout=30,
    max_tokens=300,
)

# Update agent LLM (higher-accuracy model)
update_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=1,
    timeout=45,
)

# Bind tools to the orchestrator LLM
llm_with_tools = llm.bind_tools([
    get_current_time,
    google_search,
    parse_equipment_details,
    check_lab_capability,
    # Orchestrator will also be able to trigger parsing updates, but not apply them directly
    parse_update_request,
    apply_update,
])

# Bind tools to the update LLM
update_llm_with_tools = update_llm.bind_tools([
    parse_update_request,
    apply_update,
])

# Define the state structure using TypedDict for better compatibility
class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]

# Define the agent function
def agent_function(state: AgentState) -> AgentState:
    """Main agent function that processes user input and generates responses.

    Runs a small loop to allow multiple sequential tool calls (e.g., parse then check capability).
    """
    messages = state["messages"]

    max_steps = 6
    for _ in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            # We have a final answer
            break

        # Execute tool calls and append their ToolMessages
        for tool_call in response.tool_calls:
            name = tool_call.get("name")
            args = tool_call.get("args", {}) or {}

            tool_result: str
            if name == "get_current_time":
                tool_result = get_current_time.invoke({})
            elif name == "google_search":
                tool_result = google_search.invoke({"query": args.get("query", "")})
            elif name == "parse_equipment_details":
                tool_result = parse_equipment_details.invoke({"user_text": args.get("user_text", "")})
            elif name == "check_lab_capability":
                tool_result = check_lab_capability.invoke({
                    "manufacturer": args.get("manufacturer", ""),
                    "model": args.get("model", ""),
                })
            elif name == "parse_update_request":
                tool_result = parse_update_request.invoke({
                    "request_text": args.get("request_text", "")
                })
            elif name == "apply_update":
                tool_result = apply_update.invoke({
                    "manufacturer": args.get("manufacturer", ""),
                    "model": args.get("model", ""),
                    "new_level": args.get("new_level", ""),
                    "user_name": args.get("user_name", ""),
                    "password": args.get("password", ""),
                })
            else:
                tool_result = f"Tool '{name}' not recognized."

            # Echo tool results as ToolMessage so UI can show progress
            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            ))

        # Continue loop to let the model consume tool outputs

    return {"messages": messages}

# Create the graph
def create_agent_graph():
    """Create the LangGraph agent workflow."""
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_function)
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_edge("agent", END)
    
    # Compile the graph
    return workflow.compile()

# Initialize the agent
agent = create_agent_graph()

def chat_with_agent(user_input: str, conversation_history: list = None) -> tuple[str, list]:
    """
    Chat with the agent using the provided input. Uses an LLM orchestrator in front
    of the runtime tools to collect missing information and present results.
    """
    if conversation_history is None:
        conversation_history = []

    # System instruction guiding tool use and behavior
    system_prompt = SystemMessage(content=(
        "You are a metrology lab capabilities assistant and orchestrator."
        " Primary intents: \n"
        " - Capability check: parse equipment and check against Tescom list via tools.\n"
        " - Database update: only proceed when the user explicitly asks to 'update database' or equivalent.\n"
        "Update flow: Confirm intent, ask what to update, call 'parse_update_request' to extract manufacturer, model, and new level. Then ask for the user's name and the password. Only after getting both, pass values to 'apply_update'.\n"
        "Password is required and case-insensitive ('bigbrain'). Keep questions concise."
    ))

    prior_messages = conversation_history
    if not any(isinstance(m, SystemMessage) for m in conversation_history):
        prior_messages = [system_prompt] + conversation_history

    state = {
        "messages": prior_messages + [HumanMessage(content=user_input)]
    }

    # Run the graph-backed agent (multi-step tool loop inside agent_function)
    result = agent.invoke(state)

    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    if ai_messages:
        response = ai_messages[-1].content
    else:
        response = "I apologize, but I couldn't generate a response."

    return response, result["messages"]

if __name__ == "__main__":
    # Test the agent
    print("Agent initialized successfully!")
    print("You can now run the Gradio app with: python app.py")
