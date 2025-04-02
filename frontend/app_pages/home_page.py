import streamlit as st
from typing import Dict

# Configure the page


def create_agent_container(emoji: str, title: str, features: list):
    """Create a styled container for each agent with consistent formatting."""
    st.subheader(f"{emoji} {title}")
    for feature in features:
        st.markdown(f"- {feature}")


def home_page():
    # Header with custom styling
    st.title("🤖 Advanced Multi-Agent Analysis Platform")

    st.markdown("""
    Transform your documents into insights with our AI-powered analysis platform!
    """)

    # Add a visual divider
    st.divider()

    # Agent Descriptions - Using columns for layout
    st.header("🎯 Our Intelligent Agents")
    agent_col1, agent_col2, agent_col3 = st.columns(3)

    with agent_col1:
        st.info("💬 Chatbot Agent")
        st.markdown("""
        - 🗣️ Natural conversational interface
        - ⚡ Quick response generation
        - 🔄 Context-aware interactions
        - 🎯 Task-focused dialogue
        """)

    with agent_col2:
        st.success("🔍 Research Assistant")
        st.markdown("""
        - 🌐 DuckDuckGo web search
        - 📚 arXiv paper analysis
        - 🧮 Built-in calculations
        - ⚔️ LlamaGuard safety checks
        """)

    with agent_col3:
        st.warning("📊 Multi-Modal RAG")
        st.markdown("""
        - 🎨 Visual content processing
        - 📑 Multi-format support
        - 📈 Data visualization
        - 📝 Auto-report generation
        """)

    # Key Features Section
    st.divider()
    st.header("✨ Platform Features")

    # Use tabs for organizing different feature categories
    feature_tab1, feature_tab2, feature_tab3 = st.tabs([
        "🛡️ Security & Processing",
        "📊 Analytics & Reports",
        "🔧 Technical Details"
    ])

    with feature_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔒 Security")
            st.markdown("""
            - JWT token authentication
            - Secure document storage
            - Access control management            
            """)
        with col2:
            st.markdown("### ⚙️ Processing")
            st.markdown("""
            - Multi-agent task routing
            - Parallel processing            
            - Real-time updates
            """)

    with feature_tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Analytics")
            st.markdown("""
            - Deep document analysis
            - Visual content processing            
            - Semantic understanding
            """)
        with col2:
            st.markdown("### 📊 Reporting")
            st.markdown("""
            - Report generation
            - Codelabs generation
            """)

    with feature_tab3:
        st.markdown("### 🛠️ Technical Stack")
        tech_col1, tech_col2 = st.columns(2)

        with tech_col1:
            st.markdown("""
            - 🧠 **AI Framework**: LangGraph & LangChain
            - 🎨 **Frontend**: Streamlit
            - 🛡️ **Security**: LlamaGuard
            - 🔍 **Search**: DuckDuckGo & arXiv
            """)

        with tech_col2:
            st.markdown("""
            - 📚 **Storage**: Vector Database
            - 🔄 **Processing**: RAG Architecture
            - 📊 **Analytics**: Custom Pipeline
            - 🌐 **API**: REST
            """)


    # Footer
    st.divider()
    st.caption("Powered by AI 🤖 | Built with by Akash, Uday & Surya")


if __name__ == "__main__":
    home_page()