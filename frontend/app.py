import asyncio
import logging
from collections.abc import AsyncGenerator
from dotenv import load_dotenv
from markdown2 import Markdown
from weasyprint import HTML
from streamlit.runtime.scriptrunner import get_script_run_ctx
import boto3
import os
from io import BytesIO
from typing import Tuple
import streamlit as st
from typing import Optional
import time

from client import AgentClient
from codelab import process_markdown_string
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData
from services.session_store import session_store
from services.authentication import auth
from app_pages.home_page import home_page

# Load environment variables
load_dotenv(override=True)
# Constants
APP_TITLE = "AI Chat Platform"
APP_ICON = "💬"
API_BASE_URL = os.getenv("API_BASE_URL")
print(API_BASE_URL)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)




def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "processing_report" not in st.session_state:
        st.session_state.processing_report = False
    if "processing_codelab" not in st.session_state:
        st.session_state.processing_codelab = False
    if "report_status" not in st.session_state:
        st.session_state.report_status = None
    if "codelab_status" not in st.session_state:
        st.session_state.codelab_status = None

initialize_session_state()
def handle_report_generation():
    """Handle PDF report generation with proper feedback."""
    if len(st.session_state.messages) > 0:
        st.session_state.processing_report = True
        st.session_state.report_status = "processing"

        try:
            # Your existing PDF generation logic
            st.session_state["pdf_path"] = markdown_to_pdf(
                extract_and_format_markdown(st.session_state.messages[-1].content),
                "report.pdf"
            )
            st.session_state.report_status = "success"
        except Exception as e:
            st.session_state.report_status = "error"
            st.session_state.error_message = str(e)

        st.session_state.processing_report = False
    else:
        st.session_state.report_status = "no_content"


def handle_codelab_generation():
    """Handle codelab generation with proper feedback."""
    if len(st.session_state.messages) > 0:
        st.session_state.processing_codelab = True
        st.session_state.codelab_status = "processing"

        try:
            # Your existing codelab generation logic
            process_markdown_string(
                extract_and_format_markdown(st.session_state.messages[-1].content)
            )
            st.session_state.codelab_status = "success"
        except Exception as e:
            st.session_state.codelab_status = "error"
            st.session_state.error_message = str(e)

        st.session_state.processing_codelab = False
    else:
        st.session_state.codelab_status = "no_content"


def render_status_message(status: str, process_type: str):
    """Render appropriate status messages based on the process status."""
    if status == "processing":
        return st.info(f"🔄 Generating {process_type}...")
    elif status == "success":
        return st.success(f"✅ {process_type} generated successfully!")
    elif status == "error":
        return st.error(f"❌ Error generating {process_type}: {st.session_state.get('error_message', 'Unknown error')}")
    elif status == "no_content":
        return st.warning("⚠️ No chat history to generate from.")


def render_sidebar_actions():
    """Render the sidebar actions with proper organization and feedback."""
    initialize_session_state()

    # Only show the expander if using multi-modal-rag agent
    if st.session_state.agent_client.agent == "multi-modal-rag":
        with st.sidebar:
            with st.expander("📑 Export Options", expanded=True):
                col1, col2 = st.columns(2)

                # PDF Report Generation
                with col1:
                    report_button = st.button(
                        "📄 Generate PDF",
                        key="pdf_report",
                        disabled=st.session_state.processing_report,
                        help="Generate a PDF report from the chat history"
                    )

                    if report_button:
                        handle_report_generation()

                    if st.session_state.report_status:
                        render_status_message(st.session_state.report_status, "PDF report")

                # Codelab Generation
                with col2:
                    codelab_button = st.button(
                        "🔧 Generate Codelab",
                        key="codelab",
                        disabled=st.session_state.processing_codelab,
                        help="Generate a codelab from the chat history"
                    )

                    if codelab_button:
                        handle_codelab_generation()

                    if st.session_state.codelab_status:
                        render_status_message(st.session_state.codelab_status, "Codelab")

                # PDF Download Button (only show if PDF was generated successfully)
                if st.session_state.get("pdf_path") and st.session_state.report_status == "success":
                    try:
                        with open(st.session_state["pdf_path"], "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()

                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name="chat_export.pdf",
                            mime="application/pdf",
                            help="Download the generated PDF report"
                        )
                    except Exception as e:
                        st.error(f"❌ Error preparing download: {str(e)}")

def initialize_s3_client() -> boto3.client:
    """
    Initialize and return an S3 client using environment variables.
    """
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )


def list_pdf_files(s3_client: boto3.client, bucket: str, prefix: str) -> list:
    """
    List all PDF files in the specified S3 bucket and prefix.
    """
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )

        pdf_files = [obj['Key'].split('/')[-1] for obj in response['Contents']
                     if obj['Key'].endswith('.pdf')]

        return sorted(pdf_files)  # Return sorted list for better UX
    except Exception as e:
        st.error(f"Error listing PDF files: {str(e)}")
        return []


def download_pdf(s3_client: boto3.client, bucket: str, prefix: str, filename: str) -> Tuple[BytesIO, int]:
    """
    Download a PDF file from S3 and return it as a BytesIO object along with its size.
    """
    try:
        # Construct the full S3 key
        full_key = f"{prefix}{filename}"

        # Get the object from S3
        response = s3_client.get_object(Bucket=bucket, Key=full_key)

        # Read the file content
        file_content = response['Body'].read()

        # Create BytesIO object
        pdf_bytes = BytesIO(file_content)

        return pdf_bytes, len(file_content)
    except Exception as e:
        st.error(f"Error downloading PDF: {str(e)}")
        return None, 0


def pdf_downloader_sidebar():
    """
    Create a sidebar expander with PDF listing and download functionality.
    """
    # Create an expander in the sidebar

    st.markdown("### Select and Download PDFs")

    # Initialize S3 client
    s3_client = initialize_s3_client()

    # S3 configuration
    bucket_name = "researchagent-bigdata"
    pdfs_prefix = "raw/pdfs/"

    # List available PDFs
    pdf_files = list_pdf_files(s3_client, bucket_name, pdfs_prefix)

    if not pdf_files:
        st.warning("No PDF files found in the repository.")
        return

    # Create a dropdown for file selection
    selected_pdf = st.selectbox(
        "Select a PDF document:",
        options=pdf_files,
        key="pdf_selector"
    )

    # Handle file download
    if selected_pdf:
        # Get the PDF content
        pdf_content, file_size = download_pdf(
            s3_client,
            bucket_name,
            pdfs_prefix,
            selected_pdf
        )

        if pdf_content:
            # Display file size
            size_mb = file_size / (1024 * 1024)
            st.caption(f"Size: {size_mb:.1f} MB")

            # Create download button
            st.download_button(
                label="📥 Download PDF",
                data=pdf_content,
                file_name=selected_pdf,
                mime="application/pdf",
                help="Click to download the selected PDF",
                use_container_width=True  # Make button full width
            )


# Usage in your Streamlit app:
# if __name__ == "__main__":
#     pdf_downloader_sidebar()



# Available LLM models
MODELS = {
    "OpenAI GPT-4o-mini (streaming)": "gpt-4o-mini",
    # "Gemini 1.5 Flash (streaming)": "gemini-1.5-flash",
    # "Claude 3 Haiku (streaming)": "claude-3-haiku",
    # "llama-3.1-70b on Groq": "llama-3.1-70b",
    # "AWS Bedrock Haiku (streaming)": "bedrock-haiku",
}


def display_login_page():
    """Display an elegant login page with modern features overview."""
    # Configure page
    st.set_page_config(
        page_title="AI Document Analysis Platform",
        page_icon="🤖",
        layout="wide"
    )

    # Two-column layout for main content
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Platform introduction
        st.title("🤖 AI Agents Platform")
        st.markdown("### Unleash the Power of Multi-Agent Intelligence")

        # Feature highlights with modern styling
        st.markdown("""
        <style>
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 2rem;
        }
        .feature-card {
            padding: 1.5rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.info("🤖 **Intelligent Agents**\nMultiple specialized AI agents working in harmony")
            st.success("🔍 **Smart Research**\nAdvanced web and academic research capabilities")

        with col2:
            st.warning("📊 **Visual Analytics**\nComprehensive reports with visual insights")
            st.error("🛡️ **Enterprise Security**\nRobust safety checks with LlamaGuard")

        # Additional platform benefits
        st.markdown("### Why Choose Our Platform?")
        st.markdown("""
        - 🎯 **Purpose-Built Agents**: Specialized for different tasks
        - 🔄 **Real-time Processing**: Instant responses and streaming
        - 📚 **Comprehensive Analysis**: Multi-modal document processing
        - 🌐 **Global Access**: Multi-language support
        """)

    with right_col:
        # Authentication container
        with st.container():
            if session_store.get_value('display_login'):
                display_login_form()
            elif session_store.get_value('display_register'):
                display_register_form()


def display_login_form():
    """Display an elegant login form."""
    st.markdown("### Welcome Back! 👋")

    with st.form("login_form", clear_on_submit=True):
        # Add some padding
        st.write("")

        # Styled input fields
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        st.write("")
        st.write("")

        # Submit button
        submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")

        if submit:
            if not email or not password:
                st.error("🚫 Please enter both email and password.")
                return

            try:
                auth.login(email, password)
                st.success("✅ Welcome back!")
                st.session_state['current_page'] = 'Chat'
                st.rerun()
            except Exception as e:
                st.error(f"❌ Login failed: {str(e)}")

    # Divider with "or" text
    st.markdown("<div style='text-align: center; margin: 1rem 0;'>OR</div>", unsafe_allow_html=True)

    # Registration button
    if st.button("Create New Account ✨", use_container_width=True, type="secondary"):
        show_register_form()


def display_register_form():
    """Display an elegant registration form."""
    st.markdown("### Create Account ✨")

    with st.form("register_form", clear_on_submit=True):
        # Add some padding
        st.write("")

        # Styled input fields
        username = st.text_input("👤 Username", placeholder="Choose a username")
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔑 Password", type="password", placeholder="Create a strong password")




        # Submit button
        submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

        if submit:
            if not username or not email or not password:
                st.error("🚫 Please fill in all fields.")
                return

            try:
                auth.register(username, email, password)
                st.success("✅ Account created successfully!")
                show_login_form()
            except Exception as e:
                st.error(f"❌ Registration failed: {str(e)}")

    # Back to login option
    st.markdown("<div style='text-align: center; margin: 1rem 0;'>Already have an account?</div>",
                unsafe_allow_html=True)
    if st.button("Sign In →", use_container_width=True, type="secondary"):
        show_login_form()


# Helper functions remain the same
def show_register_form():
    session_store.set_value('display_login', False)
    session_store.set_value('display_register', True)
    st.rerun()


def show_login_form():
    session_store.set_value('display_login', True)
    session_store.set_value('display_register', False)
    st.rerun()

# Session state defaults
SESSION_DEFAULTS = {
    'display_login': True,
    'display_register': False,
    'current_page': 'Home',
    'thread_id': None,
    'messages': [],
    # 'last_feedback': (None, None)
}

def initialize_session_state():
    """Initialize session state with default values."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

def clear_session_storage():
    """Clear all session storage and reinitialize defaults."""
    logging.info("Clearing session storage")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()


async def setup_agent_client():
    """Initialize the agent client if not already present."""
    if "agent_client" not in st.session_state:
        agent_url = os.getenv("AGENT_URL", "http://backend:8000")
        print("-----Acess token---")
        print(st.session_state.get("access_token"))
        st.session_state.agent_client = AgentClient(agent_url)

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            history: ChatHistory = st.session_state.agent_client.get_history(thread_id=thread_id)
            messages = history.messages
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id


def setup_sidebar():
    """Configure and display the sidebar with conditional sections."""
    with st.sidebar:
        # App Header
        st.header(f"{APP_ICON} {APP_TITLE}")

        # Navigation (Always Visible) - Not in an expander
        st.subheader("🧭 Navigation")
        selected_page = st.selectbox(
            "Go to",
            ["Home", "Chat"],
            format_func=lambda x: f"{'🏠 ' if x == 'Home' else '💬 '} {x}",
            index=list(["Home", "Chat"]).index(st.session_state['current_page']),
            label_visibility="collapsed"
        )

        if selected_page != st.session_state['current_page']:
            st.session_state['current_page'] = selected_page
            st.rerun()


        # Show additional sections only on Chat page
        if st.session_state['current_page'] == "Chat":
            # 1. Agent Settings
            with st.expander("⚙️ Agent Settings", expanded=True):
                m = st.radio(
                    "🤖 Select LLM Model",
                    options=MODELS.keys(),
                    help="Choose the language model to power your agent"
                )
                st.session_state['model'] = MODELS[m]

                st.session_state.agent_client.agent = st.selectbox(
                    "🎯 Select Agent Type",
                    options=["research-assistant", "chatbot", "multi-modal-rag"],
                    help="Choose the type of agent for your task"
                )

                st.session_state['use_streaming'] = True

            # 2. Document Manager
            with st.expander("📑 Document Manager", expanded=True):
                pdf_downloader_sidebar()

            # 3. Export Options (only shown for multi-modal-rag)
            if st.session_state.agent_client.agent == "multi-modal-rag":
                render_sidebar_actions()  # This is your export options expander

        # Add a separator before logout
        st.divider()

        # Logout button (always at the bottom)
        if st.button("🚪 Logout", use_container_width=True):
            try:
                logging.info("Logging out user")
                auth.logout()
                clear_session_storage()
                st.success("Logged out successfully!")
                st.rerun()
            except Exception as e:
                logging.error(f"Error during logout: {e}")
                st.sidebar.error("❌ Logout failed. Please try again.")
async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draw chat messages, handling both existing and streaming messages.
    
    Args:
        messages_agen: Async generator of messages to draw
        is_new: Whether these are new messages being streamed
    """
    last_message_type = None
    st.session_state.last_message = None
    streaming_content = ""
    streaming_placeholder = None

    print("Drawing messages")
    print(messages_agen)

    async for msg in messages_agen:
        if isinstance(msg, str):
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue

        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        if msg.type == "human":
            last_message_type = "human"
            st.chat_message("human").write(msg.content)
        
        elif msg.type == "ai":
            if is_new:
                st.session_state.messages.append(msg)

            if last_message_type != "ai":
                last_message_type = "ai"
                st.session_state.last_message = st.chat_message("ai")

            with st.session_state.last_message:
                if msg.content:
                    if streaming_placeholder:
                        streaming_placeholder.write(msg.content)
                        streaming_content = ""
                        streaming_placeholder = None
                    else:
                        st.write(msg.content)

                if msg.tool_calls:
                    await handle_tool_calls(msg.tool_calls, messages_agen, is_new)

async def handle_tool_calls(tool_calls, messages_agen, is_new):
    """Handle tool calls and their results."""
    call_results = {}
    for tool_call in tool_calls:
        status = st.status(
            f"""Tool Call: {tool_call["name"]}""",
            state="running" if is_new else "complete",
        )
        call_results[tool_call["id"]] = status
        status.write("Input:")
        status.write(tool_call["args"])

    for _ in range(len(call_results)):
        print("Waiting for tool result")
        tool_result: ChatMessage = await anext(messages_agen)
        if tool_result.type != "tool":
            st.error(f"Unexpected ChatMessage type: {tool_result.type}")
            st.write(tool_result)
            st.stop()

        if is_new:
            st.session_state.messages.append(tool_result)
        status = call_results[tool_result.tool_call_id]
        status.write("Output:")
        status.write(tool_result.content)
        status.update(state="complete")


import re


def extract_and_format_markdown(content: str) -> str:
    """
    Extract markdown code blocks and format them properly for Streamlit rendering.
    Returns the processed content with proper markdown rendering.
    """
    # Pattern to match markdown code blocks: ```markdown ... ```
    pattern = r'```markdown\s*(.*?)\s*```'

    def replace_markdown(match):
        # Extract the markdown content and render it
        markdown_content = match.group(1).strip()
        return markdown_content

    # Replace all markdown code blocks
    processed_content = re.sub(pattern, replace_markdown, content, flags=re.DOTALL)
    return processed_content
def markdown_to_pdf(markdown_content, output_filename):
    # Convert Markdown to HTML


    markdowner = Markdown()
    html_content = markdowner.convert(markdown_content)
    logging.info(f"Converted Markdown to HTML:\n{html_content}")
    # Wrap the HTML content in a basic HTML structure
    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1, h2, h3 {{ color: #333; }}
            code {{ background-color: #f0f0f0; padding: 2px 4px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Create a PDF from the HTML
    html = HTML(string=full_html)
    pdf_file = html.write_pdf()
    output_filename = output_filename
    # Save the PDF file
    with open(output_filename, 'wb') as f:
        f.write(pdf_file)

    # Return the absolute path of the saved PDF
    p = os.path.abspath(output_filename)
    logging.info(f"PDF saved to: {p}")
    return p


async def chat_interface():
    """Display and handle the chat interface."""
    messages: list[ChatMessage] = st.session_state.messages
    if len(messages) == 0:
        with st.chat_message("ai"):
            st.write("Hello! I'm an AI assistant. How can I help you today?")

    async def amessage_iter():
        for m in messages:
            yield m

    await draw_messages(amessage_iter())
        # Add buttons in a horizontal layout




    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)

        use_streaming = st.session_state.get('use_streaming', True)
        model = st.session_state.get('model', MODELS["OpenAI GPT-4o-mini (streaming)"])

        if use_streaming and st.session_state.agent_client.agent != "multi-modal-rag":
            stream = st.session_state.agent_client.astream(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
            )
            print("Streaming response")
            print(stream)
            await draw_messages(stream, is_new=True)
        else:
            response = await st.session_state.agent_client.ainvoke(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
            )
            messages.append(response)
            print("-------------Response---------")
            print(response)
            # Process the content for markdown blocks before displaying
            processed_content = extract_and_format_markdown(response.content)
            print("------------------Processed alsalasl content------------------\n\n")
            print("Processed content")
            print(processed_content)
            with st.chat_message("ai"):
                st.markdown(processed_content)

        st.rerun()
async def main():
    """Main application entry point."""
    initialize_session_state()    
    
    if not session_store.is_authenticated():
        display_login_page()
        return

    await setup_agent_client()
    setup_sidebar()
    st.markdown(
        f'''
            <style>
                .sidebar .sidebar-content {{
                    width: 40%;
                }}
            </style>
        ''',
        unsafe_allow_html=True
    )

    current_page = st.session_state['current_page']
    
    if current_page == "Chat":
        await chat_interface()
    elif current_page == "Home":
        home_page()

if __name__ == "__main__":
    asyncio.run(main())

