import asyncio
import sniffio
import streamlit as st
import uuid

from app.agents.orchestrator import remedy_graph
from app.agents.state import AgentState

st.set_page_config(
    page_title="RemedyRadar",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 RemedyRadar")
st.markdown("Find the right specialist for your symptoms.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.turn_number = 0
    st.session_state.conversation_history = []
    st.session_state.last_state = None
    
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm **RemedyRadar**. Describe your symptoms and tell me your city — I'll find the right specialist for you.\n\n*Example: \"I've been having persistent chest tightness when I exercise, I'm in Bangalore\"*"}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your symptoms..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # We will use st.status to show the agent steps
        with st.status("Agents are thinking...", expanded=True) as status_box:
            st.session_state.turn_number += 1
            
            state: AgentState = {
                "raw_query": prompt,
                "session_id": st.session_state.session_id,
                "turn_number": st.session_state.turn_number,
                "conversation_history": st.session_state.conversation_history,
            }

            if st.session_state.last_state:
                if st.session_state.last_state.get("location"):
                    state["location"] = st.session_state.last_state["location"]
                if st.session_state.last_state.get("preferences"):
                    state["preferences"] = st.session_state.last_state["preferences"]
                if st.session_state.last_state.get("applied_filters"):
                    state["applied_filters"] = st.session_state.last_state["applied_filters"]

            # Run steps synchronously using asyncio.run but stream events
            async def process_query(current_state, status_box):
                # Inject sniffio locally so we don't break Streamlit's Starlette server
                token = sniffio.current_async_library_cvar.set("asyncio")
                try:
                    final_state = current_state
                    # We yield control to the graph and just listen to events
                    async for event in remedy_graph.astream(current_state):
                        if "intake_agent" in event:
                            node_state = event["intake_agent"]
                            symptoms = node_state.get("extracted_symptoms", [])
                            st.write(f"✓ Extracted: {', '.join(symptoms)} | Urgency: {node_state.get('urgency_level', 'routine')}")
                            final_state = node_state
                        elif "emergency_detector" in event:
                            node_state = event["emergency_detector"]
                            if node_state.get("is_emergency"):
                                st.write("🚨 EMERGENCY DETECTED")
                            else:
                                st.write("✅ No emergency — proceeding")
                            final_state = node_state
                        elif "knowledge_agent" in event:
                            node_state = event["knowledge_agent"]
                            st.write(f"📚 Specialty: {node_state.get('recommended_specialty', 'Unknown')} (confidence: {node_state.get('specialty_confidence', 0):.0%})")
                            final_state = node_state
                        elif "retrieval_agent" in event:
                            node_state = event["retrieval_agent"]
                            n_found = len(node_state.get("candidate_doctors", []))
                            if not node_state.get("candidate_doctors"):
                                st.write("❌ No candidates found.")
                            else:
                                st.write(f"🔍 Found {n_found} candidates")
                            final_state = node_state
                        elif "ranking_agent" in event:
                            node_state = event["ranking_agent"]
                            n_ranked = len(node_state.get("ranked_doctors", []))
                            st.write(f"📊 Top {n_ranked} ranked")
                            final_state = node_state
                        elif "explanation_agent" in event:
                            node_state = event["explanation_agent"]
                            st.write("✍️ Generated Explanations")
                            final_state = node_state
                        elif "format_emergency" in event:
                            final_state = event["format_emergency"]
                        elif "format_no_results" in event:
                            final_state = event["format_no_results"]
                        elif "format_results" in event:
                            final_state = event["format_results"]
                    return final_state
                finally:
                    sniffio.current_async_library_cvar.reset(token)
            
            st.write("🧠 Analyzing Symptoms...")
            state = asyncio.run(process_query(state, status_box))
            final_response = state.get("final_response", "Sorry, an error occurred.")
            
            status_box.update(label="Response ready!", state="complete", expanded=False)
        
        # Display final response outside the status box
        st.markdown(final_response)
        
        # Save to session history
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({
            "role": "assistant",
            "specialty": state.get("recommended_specialty"),
            "n_results": len(state.get("ranked_doctors", [])),
        })
        st.session_state.last_state = {
            "location": state.get("location"),
            "preferences": state.get("preferences"),
            "applied_filters": state.get("applied_filters"),
            "recommended_specialty": state.get("recommended_specialty"),
        }
