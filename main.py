 
import gradio as gr
import re
from openai import AzureOpenAI
import databricks.sql as dbsql
import mlflow
from pymongo import MongoClient
import datetime
from dotenv import load_dotenv
import os 

mlflow.openai.autolog()
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

client = MongoClient(MONGO_URI)
db = client["fleet_data"]
collection = db["fleet_collection"]
 
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
 
# --- Initialize Azure OpenAI client ---
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
 
# --- 2. MONGODB FUNCTIONS ---
 
def save_chat_session(title, chat, mode):
    """Save chat session to MongoDB"""
    if chat:
        collection.insert_one({
            "userid": USERID,
            "user": USER,
            "mode": mode,
            "title": title,
            "timestamp": datetime.datetime.utcnow(),
            "conversation": chat
        })
        print(f"✅ Chat session saved: {title}")
 
def get_all_chats():
    """Retrieve all chat sessions from MongoDB"""
    docs = collection.find({"userid": USERID, "conversation": {"$exists": True}}).sort("timestamp", -1)
    return [(doc.get("title", "Untitled"), doc.get("mode", "Unknown"), doc["conversation"], doc["timestamp"].strftime("%Y-%m-%d %H:%M")) for doc in docs]
 
def show_previous_chats():
    """Format previous chats for display"""
    chats = get_all_chats()
    markdown = ""
    for title, mode, convo, timestamp in chats:
        markdown += f"• **{title}**  \n*{mode}*\n\n"
    return markdown if markdown else "No previous chats found."
 
# --- 3. HELPER FUNCTIONS FOR FLEET MANAGER MODE ---
 
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
5. Use LIMIT clause when appropriate to avoid overwhelming results (max 100 rows)
6. Return ONLY the SQL query without any explanation, markdown formatting, or additional text
 
Examples:
Question: "Show me all vehicles that need maintenance"
SQL: SELECT * FROM {TABLE_NAME} WHERE Maintenance_Required = 1 LIMIT 50
 
Question: "What is the average engine temperature?"
SQL: SELECT AVG(Engine_Temperature) as avg_temp FROM {TABLE_NAME}
 
Question: "How many trucks need maintenance?"
SQL: SELECT COUNT(*) as count FROM {TABLE_NAME} WHERE Vehicle_Type = 'Truck' AND Maintenance_Required = 1
"""
 
    try:
        response = llm_client.chat.completions.create(
            model=AZURE_LLM_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.1,
            max_tokens=300
        )
        sql_query = response.choices[0].message.content.strip()
       
        # Clean up the SQL query
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
    try:
        with dbsql.connect(
            server_hostname=DB_HOST,
            http_path=DB_HTTP_PATH,
            access_token=DB_TOKEN
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
               
                if not results:
                    return "No data found matching your query.", []
               
                formatted_results = f"Query Results ({len(results)} rows):\n\n"
               
                if len(results) == 1 and len(columns) == 1:
                    formatted_results += f"{columns[0]}: {results[0][0]}\n"
                else:
                    for row in results[:100]:
                        row_dict = dict(zip(columns, row))
                        for col, val in row_dict.items():
                            formatted_results += f"{col}: {val}\n"
                        formatted_results += "\n"
                   
                    if len(results) > 100:
                        formatted_results += f"\n(Showing first 100 of {len(results)} results)"
               
                return formatted_results, results
               
    except Exception as e:
        print(f"❌ ERROR executing SQL: {e}")
        return f"Error executing query: {str(e)}", []
 
def interpret_sql_results(user_question, sql_query, query_results):
    """Use LLM to interpret SQL results and provide natural language response"""
    print("[DEBUG] Interpreting SQL results with LLM")
   
    system_prompt = """You are a fleet management assistant. You help interpret database query results
    and present them in a clear, professional manner.
   
    Based on the user's question and the query results, provide a concise, helpful response.
    Format your response clearly and highlight important information.
    """
   
    user_prompt = f"""
    User's Question: "{user_question}"
   
    SQL Query Executed:
    {sql_query}
   
    Query Results:
    {query_results}
   
    Please interpret these results and answer the user's question in a clear, professional manner.
    Provide insights and actionable information based on the data.
    """
   
    try:
        response = llm_client.chat.completions.create(
            model=AZURE_LLM_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ ERROR interpreting results: {e}")
        return query_results
 
# --- 4. MAIN CHAT ORCHESTRATOR ---
 
def chat_orchestrator(message, history, mode):
    """Main orchestrator that routes to appropriate mode"""
    global current_chat
    print(f"[DEBUG] Mode: {mode}, Message: {message}")
   
    if not llm_client:
        yield "The AI service is not available. Please check the configuration."
        return
   
    if mode == "👔 Fleet Manager Mode":
        # FLEET MANAGER MODE: Query database with LLM-generated SQL
        print("[DEBUG] Entering Fleet Manager Mode")
       
        yield "🔍 Analyzing your question and generating database query..."
       
        sql_query = generate_sql_query(message)
        if not sql_query:
            response = "I couldn't generate a valid query for your question. Please try rephrasing it."
            current_chat.append({"message": message, "response": response})
            yield response
            return
       
        yield f"⚙️ Executing query...\n\n`{sql_query}`\n\n"
       
        query_results, raw_data = execute_sql_query(sql_query)
       
        if not raw_data:
            response = f"📊 Query executed successfully.\n\n{query_results}"
            current_chat.append({"message": message, "response": response})
            yield response
            return
       
        yield "📊 Analyzing results..."
       
        final_answer = interpret_sql_results(message, sql_query, query_results)
       
        print("\n--- FLEET MANAGER MODE Response ---")
        print(final_answer)
        print("-----------------------------------\n")
       
        current_chat.append({"message": message, "response": final_answer})
        yield final_answer
   
    else:
        # USER MODE: General maintenance advice without database access
        print("[DEBUG] Entering User Mode")
       
        system_prompt = """You are a knowledgeable vehicle maintenance assistant helping regular users.
 
You provide general advice about:
- Vehicle maintenance best practices
- Understanding maintenance indicators
- When to schedule maintenance
- Common vehicle issues and solutions
- Interpreting warning signs
- Predictive maintenance concepts
 
You DO NOT have access to specific fleet data or databases.
If users ask about specific vehicles or fleet data, politely inform them that they need
to switch to Fleet Manager mode to access that information.
 
Be helpful, professional, and educational in your responses.
"""
       
        try:
            response = llm_client.chat.completions.create(
                model=AZURE_LLM_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            final_answer = response.choices[0].message.content
            print("\n--- USER MODE Response ---")
            print(final_answer)
            print("-------------------------\n")
           
            current_chat.append({"message": message, "response": final_answer})
            yield final_answer
        except Exception as e:
            print(f"❌ ERROR in User Mode: {e}")
            error_msg = "I'm sorry, I'm having trouble processing your request. Please try again later."
            current_chat.append({"message": message, "response": error_msg})
            yield error_msg
 
# --- 5. GRADIO INTERFACE ---
 
def respond(message, history, mode):
    """Handle user message and generate response"""
    if not message.strip():
        return history, ""
   
    history.append((message, None))
   
    for response in chat_orchestrator(message, history, mode):
        history[-1] = (message, response)
        yield history, ""
 
def start_new_chat(history, mode):
    """Start a new chat session and save the current one"""
    global current_chat
   
    if current_chat:
        title = current_chat[0]["message"][:50] if current_chat else f"Chat on {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        save_chat_session(title, current_chat, mode)
   
    current_chat = []
    welcome_msg = "<div style='text-align:center; font-size:20px; font-weight:bold; color:#2196F3;'>✨ New chat started! How can I help you today? ✨</div>"
    history = [(welcome_msg, "")]
   
    return history, gr.update(value=show_previous_chats())
 
def toggle_sidebar(current_visible):
    """Toggle sidebar visibility"""
    new_state = not current_visible
    content_update = gr.update(value=show_previous_chats()) if new_state else gr.update(value="")
    return new_state, gr.update(visible=new_state), content_update
 
# --- 6. BUILD GRADIO UI ---
 
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
   
    #new-chat-btn:hover, #sidebar-btn:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
   
    #new-chat-btn:active, #sidebar-btn:active {
        transform: translateY(0);
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
   
    #sidebar-btn {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
   
    #sidebar-btn:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        box-shadow: 0 6px 16px rgba(139, 92, 246, 0.4);
    }
   
    #mode-selector {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.2);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
   
    .mode-description {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1));
        border-left: 4px solid #22c55e;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
   
    #send-btn {
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
        width: 100%;
    }
   
    #send-btn:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
   
    #send-btn:active {
        transform: translateY(0);
    }
   
    .chatbot-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 15px;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
    }
   
    textarea {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px !important;
        color: white;
        padding: 12px !important;
        transition: all 0.3s ease;
    }
   
    textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
    }
   
    .previous-chats-display {
        color: #e2e8f0;
        font-size: 13px;
    }
   
    .previous-chats-display h3 {
        color: #60a5fa;
        margin-top: 10px;
        font-size: 14px;
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
                        max_lines=5
                    )
                    send_btn = gr.Button("📤 Send", elem_id="send-btn", scale=0, min_width=80)
   
    def update_mode_description(mode):
        """Update description based on selected mode"""
        if mode == "👤 User Mode":
            return """
            <div class="mode-description">
            <strong>👤 User Mode Active</strong><br>
            • Ask and learn about vehicle maintenance<br>
            • No access to specific fleet data
            </div>
            """
        else:
            return """
            <div class="mode-description">
            <strong>👔 Fleet Manager Mode Active</strong><br>
            • Monitor and predict fleet performance<br>
            • Access and analyze vehicle data
            </div>
            """
   
    # Connect events
    mode_selector.change(
        fn=update_mode_description,
        inputs=[mode_selector],
        outputs=[mode_description]
    )
   
    send_btn.click(respond, [msg, chatbot, mode_selector], [chatbot, msg])
    msg.submit(respond, [msg, chatbot, mode_selector], [chatbot, msg])
    new_chat_btn.click(start_new_chat, [chatbot, mode_selector], [chatbot, previous_chats_display])
    demo.load(show_previous_chats, outputs=previous_chats_display)
    sidebar_btn.click(toggle_sidebar, sidebar_state, [sidebar_state, sidebar, previous_chats_display])
 
# --- 7. LAUNCH ---
 
if __name__ == "__main__":
    print("🚀 Dual-Mode Fleet Maintenance Chatbot Starting...")
    demo.launch(share=True)