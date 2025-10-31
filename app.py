import gradio as gr
import re
from openai import AzureOpenAI
import databricks.sql as dbsql
import mlflow
from pymongo import MongoClient
import datetime
from dotenv import load_dotenv
import os 

# Load .env file
load_dotenv()

# Azure OpenAI
AZURE_LLM_ENDPOINT = os.getenv("AZURE_LLM_ENDPOINT")
AZURE_LLM_KEY = os.getenv("AZURE_LLM_KEY")
AZURE_LLM_DEPLOYMENT_NAME = os.getenv("AZURE_LLM_DEPLOYMENT_NAME")

# Databricks
DB_HOST = os.getenv("DB_HOST")
DB_HTTP_PATH = os.getenv("DB_HTTP_PATH")
DB_TOKEN = os.getenv("DB_TOKEN")
TABLE_NAME = os.getenv("TABLE_NAME")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")

# Validate required environment variables
required_vars = {
    "AZURE_LLM_ENDPOINT": AZURE_LLM_ENDPOINT,
    "AZURE_LLM_KEY": AZURE_LLM_KEY,
    "AZURE_LLM_DEPLOYMENT_NAME": AZURE_LLM_DEPLOYMENT_NAME,
    "MONGO_URI": MONGO_URI
}

missing_vars = [k for k, v in required_vars.items() if not v]
if missing_vars:
    print(f"⚠️  WARNING: Missing environment variables: {', '.join(missing_vars)}")
    print("Please check your .env file")

try:
    client = MongoClient(MONGO_URI)
    db = client["fleet_data"]
    collection = db["fleet_collection"]
    client.server_info()
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"❌ ERROR connecting to MongoDB: {e}")
    collection = None

USERID = "manager123"
USER = "FleetManager"

# Global state for current chat
current_chat = []

# Delta Table Schema for SQL Generation
DELTA_TABLE_SCHEMA = """
Table Name: logistics_maintenance_predictions

Columns:
- Vehicle_ID (int): Unique identifier for each vehicle
- Make_and_Model (string): Vehicle make and model
- Vehicle_Type (string): Type of vehicle (e.g., Truck, Van)
- Usage_Hours (int): Total hours the vehicle has been in use
- Route_Info (string): Information about routes
- Load_Capacity (double): Maximum load capacity
- Actual_Load (double): Current or actual load
- Last_Maintenance_Date (date): Date of last maintenance
- Maintenance_Type (string): Type of maintenance performed
- Maintenance_Cost (double): Cost of maintenance
- Engine_Temperature (double): Engine temperature in °C
- Tire_Pressure (double): Tire pressure
- Fuel_Consumption (double): Fuel consumption metrics
- Battery_Status (double): Battery status percentage
- Vibration_Levels (double): Vibration levels
- Oil_Quality (double): Oil quality rating
- Brake_Condition (string): Condition of brakes
- Road_Conditions (string): Road conditions
- Impact_on_Efficiency (double): Impact on vehicle efficiency
- Predictive_Score (double): Predictive maintenance score
- Maintenance_Required (bigint): 1 if maintenance required, 0 otherwise
- Failure_Type (string): Type of potential failure
- Risk_Factors (string): Risk factors for maintenance
"""

# Initialize Azure OpenAI client
try:
    llm_client = AzureOpenAI(
        api_key=AZURE_LLM_KEY,
        api_version="2024-02-01",
        azure_endpoint=AZURE_LLM_ENDPOINT
    )
    print("✅ Azure OpenAI client initialized.")
except Exception as e:
    print(f"❌ ERROR initializing Azure OpenAI client: {e}")
    llm_client = None

# --- MONGODB FUNCTIONS ---
def save_chat_session(title, chat, mode):
    """Save chat session to MongoDB"""
    if chat and collection is not None:
        try:
            collection.insert_one({
                "userid": USERID,
                "user": USER,
                "mode": mode,
                "title": title,
                "timestamp": datetime.datetime.utcnow(),
                "conversation": chat
            })
            print(f"✅ Chat session saved: {title}")
        except Exception as e:
            print(f"❌ ERROR saving chat session: {e}")

def get_all_chats():
    """Retrieve all chat sessions from MongoDB"""
    if collection is None:
        return []
    try:
        docs = collection.find({"userid": USERID, "conversation": {"$exists": True}}).sort("timestamp", -1)
        return [(doc.get("title", "Untitled"), doc.get("mode", "Unknown"), doc["conversation"], doc["timestamp"].strftime("%Y-%m-%d %H:%M")) for doc in docs]
    except Exception as e:
        print(f"❌ ERROR retrieving chats: {e}")
        return []

def show_previous_chats():
    """Format previous chats for display"""
    chats = get_all_chats()
    markdown = ""
    for title, mode, convo, timestamp in chats:
        markdown += f"• **{title}**  \n*{mode}* - {timestamp}\n\n"
    return markdown if markdown else "No previous chats found."

# --- HELPER FUNCTIONS FOR FLEET MANAGER MODE ---
def generate_sql_query(user_question):
    """Use LLM to generate SQL query based on user's natural language question"""
    print(f"[DEBUG] Generating SQL for question: {user_question}")
   
    system_prompt = f"""You are an expert SQL query generator for a fleet maintenance database.

{DELTA_TABLE_SCHEMA}

Generate a VALID SQL query based on the user's question. Follow these rules:
1. ONLY generate the SQL query, nothing else
2. Use proper SQL syntax for Databricks
3. Always use the table name: {TABLE_NAME}
4. For date comparisons, use proper date formatting
5. Use LIMIT clause when appropriate to avoid overwhelming results (max 50 rows)
6. Return ONLY the SQL query without any explanation, markdown formatting, or additional text
"""

    try:
        response = llm_client.chat.completions.create(
            model=AZURE_LLM_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.1,
            max_tokens=300,
            timeout=20
        )
        sql_query = response.choices[0].message.content.strip()
        sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query)
        sql_query = sql_query.strip()
        print(f"[DEBUG] Generated SQL: {sql_query}")
        return sql_query
    except Exception as e:
        print(f"❌ ERROR generating SQL: {e}")
        return None

def execute_sql_query(sql_query):
    """Execute the generated SQL query on Databricks"""
    print(f"[DEBUG] Executing SQL: {sql_query}")
    connection = None
    cursor = None
    
    try:
        connection = dbsql.connect(
            server_hostname=DB_HOST,
            http_path=DB_HTTP_PATH,
            access_token=DB_TOKEN
        )
        cursor = connection.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
       
        if not results:
            return "No data found matching your query.", []
       
        formatted_results = f"Query Results ({len(results)} rows):\n\n"
       
        if len(results) == 1 and len(columns) == 1:
            formatted_results += f"{columns[0]}: {results[0][0]}\n"
        else:
            display_limit = min(50, len(results))
            for row in results[:display_limit]:
                row_dict = dict(zip(columns, row))
                for col, val in row_dict.items():
                    formatted_results += f"{col}: {val}\n"
                formatted_results += "\n"
           
            if len(results) > display_limit:
                formatted_results += f"\n(Showing first {display_limit} of {len(results)} results)"
       
        return formatted_results, results
    except Exception as e:
        print(f"❌ ERROR executing SQL: {e}")
        return f"Error executing query: {str(e)}", []
    finally:
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if connection:
            try:
                connection.close()
            except:
                pass

def interpret_sql_results(user_question, sql_query, query_results):
    """Use LLM to interpret SQL results and provide natural language response"""
    print("[DEBUG] Interpreting SQL results with LLM")
   
    system_prompt = """You are a fleet management assistant. You help interpret database query results
    and present them in a clear, professional manner.
    Provide insights and actionable information based on the data.
    Keep responses concise and under 400 words.
    """
   
    user_prompt = f"""
    User's Question: "{user_question}"
   
    SQL Query Executed:
    {sql_query}
   
    Query Results:
    {query_results[:2000]}
   
    Please interpret these results and answer the user's question clearly.
    """
   
    try:
        response = llm_client.chat.completions.create(
            model=AZURE_LLM_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            timeout=25
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ ERROR interpreting results: {e}")
        return f"Here are the query results:\n\n{query_results[:1000]}"

# --- MAIN CHAT ORCHESTRATOR ---
def chat_orchestrator(message, history, mode):
    """Main orchestrator that routes to appropriate mode"""
    global current_chat
    print(f"[DEBUG] Mode: {mode}, Message: {message}")
   
    if not llm_client:
        yield "The AI service is not available. Please check the configuration."
        return
   
    if mode == "👔 Fleet Manager Mode":
        print("[DEBUG] Entering Fleet Manager Mode")
        
        try:
            # Generate SQL query
            sql_query = generate_sql_query(message)
            if not sql_query:
                response = "I couldn't generate a valid query for your question. Please try rephrasing it."
                current_chat.append({"message": message, "response": response})
                yield response
                return
            
            # Execute query
            query_results, raw_data = execute_sql_query(sql_query)
            
            if not raw_data:
                response = f"📊 Query executed successfully.\n\n{query_results}"
                current_chat.append({"message": message, "response": response})
                yield response
                return
            
            # Interpret results
            final_answer = interpret_sql_results(message, sql_query, query_results)
            complete_response = f"**Query Executed:**\n```sql\n{sql_query}\n```\n\n{final_answer}"
            
            print("\n--- FLEET MANAGER MODE Response ---")
            print(complete_response)
            print("-----------------------------------\n")
            
            current_chat.append({"message": message, "response": complete_response})
            yield complete_response
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nPlease try again."
            print(f"❌ ERROR in Fleet Manager Mode: {e}")
            current_chat.append({"message": message, "response": error_msg})
            yield error_msg
   
    else:
        # USER MODE
        print("[DEBUG] Entering User Mode")
        system_prompt = """You are a knowledgeable vehicle maintenance assistant helping regular users.
        Provide general advice about vehicle maintenance best practices.
        You DO NOT have access to specific fleet data."""
       
        try:
            response = llm_client.chat.completions.create(
                model=AZURE_LLM_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500,
                timeout=30
            )
            final_answer = response.choices[0].message.content
            print("\n--- USER MODE Response ---")
            print(final_answer)
            print("-------------------------\n")
            
            current_chat.append({"message": message, "response": final_answer})
            yield final_answer
        except Exception as e:
            print(f"❌ ERROR in User Mode: {e}")
            error_msg = "I'm having trouble processing your request. Please try again."
            current_chat.append({"message": message, "response": error_msg})
            yield error_msg

# --- GRADIO UI FUNCTIONS ---
def respond(message, history, mode):
    """Handle user message and generate response with proper streaming"""
    if not message.strip():
        yield history, ""
        return
   
    # Add user message to history
    history = history + [(message, "")]
    
    # Stream the response
    for partial_response in chat_orchestrator(message, history[:-1], mode):
        history[-1] = (message, partial_response)
        yield history, ""
    
    # Final yield
    yield history, ""

def start_new_chat(history, mode):
    """Start a new chat session and save the current one"""
    global current_chat
   
    if current_chat:
        title = current_chat[0]["message"][:50] if current_chat else f"Chat {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        save_chat_session(title, current_chat, mode)
   
    current_chat = []
    return [], gr.update(value=show_previous_chats())

def toggle_sidebar(current_visible):
    """Toggle sidebar visibility"""
    new_state = not current_visible
    content_update = gr.update(value=show_previous_chats()) if new_state else gr.update(value="")
    return new_state, gr.update(visible=new_state), content_update

# --- BUILD GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css="""
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    #main-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        min-height: 100vh;
    }
    #header {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
    }
    .top-buttons-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    #new-chat-btn, #sidebar-btn {
        flex: 1;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 12px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    #sidebar {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(10px);
        height: calc(100vh - 250px);
        overflow-y: auto;
    }
    .mode-description {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1));
        border-left: 4px solid #22c55e;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
""") as demo:
    sidebar_state = gr.State(False)
   
    with gr.Column(elem_id="main-container"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                with gr.Column():
                    gr.HTML("<h3 style='color: #60a5fa; margin-top: 0;'>📚 Chat History</h3>")
                    with gr.Row(elem_classes="top-buttons-container"):
                        new_chat_btn = gr.Button("➕ New Chat", elem_id="new-chat-btn", scale=1)
                        sidebar_btn = gr.Button("☰ Previous Chats", elem_id="sidebar-btn", scale=1)
                    sidebar = gr.Column(visible=False, elem_id="sidebar")
                    with sidebar:
                        previous_chats_display = gr.Markdown(elem_classes="previous-chats-display")
       
            with gr.Column(scale=3):
                gr.HTML("<div id='header'><h1 style='margin: 0; text-align: center; color: white;'>🚛 Intelligent Fleet Maintenance Assistant</h1><p style='margin: 8px 0 0 0; text-align: center; color: rgba(255,255,255,0.8); font-size: 14px;'>Advanced Predictive Maintenance & Fleet Analytics</p></div>")
                mode_selector = gr.Radio(
                    choices=["👤 User Mode", "👔 Fleet Manager Mode"],
                    value="👤 User Mode",
                    label="Select Your Role",
                    info="Choose between general advice or fleet database access",
                    elem_id="mode-selector"
                )
                mode_description = gr.Markdown("""
                <div class="mode-description">
                <strong>👤 User Mode Active</strong><br>
                • Ask and learn about vehicle maintenance<br>
                • No access to specific fleet data
                </div>
                """, elem_classes="mode-description")
                chatbot = gr.Chatbot(height=500, elem_classes="chatbot-container")
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question here...",
                        show_label=False,
                        lines=3,
                        max_lines=5,
                        scale=9
                    )
                    send_btn = gr.Button("📤 Send", scale=1)
   
    def update_mode_description(mode):
        if mode == "👤 User Mode":
            return """<div class="mode-description"><strong>👤 User Mode Active</strong><br>• Ask and learn about vehicle maintenance<br>• No access to specific fleet data</div>"""
        else:
            return """<div class="mode-description"><strong>👔 Fleet Manager Mode Active</strong><br>• Monitor and predict fleet performance<br>• Access and analyze vehicle data</div>"""
   
    # Connect events
    mode_selector.change(
        fn=update_mode_description,
        inputs=[mode_selector],
        outputs=[mode_description],
        queue=False
    )
    send_btn.click(respond, [msg, chatbot, mode_selector], [chatbot, msg], queue=True)
    msg.submit(respond, [msg, chatbot, mode_selector], [chatbot, msg], queue=True)
    new_chat_btn.click(start_new_chat, [chatbot, mode_selector], [chatbot, previous_chats_display], queue=False)
    demo.load(show_previous_chats, outputs=previous_chats_display, queue=False)
    sidebar_btn.click(toggle_sidebar, sidebar_state, [sidebar_state, sidebar, previous_chats_display], queue=False)

# Configure queue for Gradio 3.x
demo.queue(concurrency_count=3, status_update_rate="auto")

# Launch
if __name__ == "__main__":
    print("🚀 Dual-Mode Fleet Maintenance Chatbot Starting...")
    print(f"Gradio Version: {gr.__version__}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        enable_queue=True,  # CRITICAL for Gradio 3.x
        debug=False
    )
