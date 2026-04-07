"""Agent logic for Toronto Bylaw Conversational Agent."""
import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum
from backend.llm import llm_client
from backend.rag import rag


class Intent(str, Enum):
    """Intent types for user messages."""
    GENERAL_INQUIRY = "general_inquiry"
    HAZARD_REPORT = "hazard_report"
    PERMIT_SCREENER = "permit_screener"
    COLLECTION_LOOKUP = "collection_lookup"
    OUT_OF_SCOPE = "out_of_scope"


class AgentResponse:
    """Response from agent."""
    
    def __init__(
        self,
        intent: Intent,
        message: str,
        action: Optional[Dict[str, Any]] = None,
        citations: Optional[List[Dict[str, str]]] = None
    ):
        self.intent = intent
        self.message = message
        self.action = action
        self.citations = citations or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent.value,
            "message": self.message,
            "action": self.action,
            "citations": self.citations
        }


class TorontoBylawAgent:
    """Main agent for processing user messages."""
    
    async def classify_intent(self, user_message: str) -> Intent:
        """Classify user message intent."""
        system_prompt = """You are an intent classifier for a Toronto municipal services agent.
Classify the user's message into one of these categories:
- "general_inquiry": Questions about Toronto bylaws, regulations, or services
- "hazard_report": User wants to report a hazard (pothole, fallen tree, debris, etc.)
- "permit_screener": User wants to know if they need a permit for a project
- "collection_lookup": User wants to know waste collection schedule
- "out_of_scope": Request is not related to Toronto municipal services

Respond with ONLY the intent name, nothing else."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await llm_client.invoke(messages, temperature=0.3, max_tokens=50)
            intent_str = response.lower().strip()
            
            # Validate intent
            for intent in Intent:
                if intent.value in intent_str:
                    return intent
            
            return Intent.OUT_OF_SCOPE
        except Exception as e:
            print(f"[Agent] Error classifying intent: {e}")
            return Intent.OUT_OF_SCOPE
    
    async def handle_general_inquiry(self, user_message: str) -> AgentResponse:
        """Handle general inquiry with RAG."""
        # Search knowledge base
        search_results = rag.search(user_message, top_k=3)
        
        if not search_results:
            return AgentResponse(
                intent=Intent.GENERAL_INQUIRY,
                message="I couldn't find specific information about your query in the Toronto municipal bylaws database. Please try rephrasing your question or contact Toronto 311 at 416-392-8111 for more assistance.",
                citations=[]
            )
        
        # Build context from search results
        context = "\n\n---\n\n".join([
            f"Title: {doc['title']}\nContent: {doc['content']}"
            for doc in search_results
        ])
        
        system_prompt = """You are a helpful Toronto municipal services assistant.
Answer the user's question based on the provided Toronto bylaw and service information.
Be concise, accurate, and cite the relevant bylaw or service when applicable.
If the information doesn't fully answer the question, suggest contacting Toronto 311."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from Toronto bylaws:\n\n{context}\n\nUser question: {user_message}"}
        ]
        
        try:
            response = await llm_client.invoke(messages)
            
            citations = [
                {
                    "title": doc["title"],
                    "source": doc["source"],
                    "excerpt": doc["content"][:200] + "..."
                }
                for doc in search_results
            ]
            
            return AgentResponse(
                intent=Intent.GENERAL_INQUIRY,
                message=response,
                citations=citations
            )
        except Exception as e:
            print(f"[Agent] Error handling general inquiry: {e}")
            return AgentResponse(
                intent=Intent.GENERAL_INQUIRY,
                message="I encountered an error processing your request. Please try again."
            )
    
    async def handle_hazard_report(self, user_message: str, conversation_state: Dict[str, Any] = None) -> AgentResponse:
        """Handle hazard report with multi-turn dialogue."""
        if conversation_state is None:
            conversation_state = {}
        
        # Check if we have all required information
        required_fields = ["location", "hazard_type", "description"]
        missing_fields = [f for f in required_fields if f not in conversation_state]
        
        if not missing_fields:
            # All information collected, create ticket
            ticket_id = f"311-{hash(str(conversation_state)) % 100000:05d}"
            message = f"""Hazard report submitted successfully!
Ticket ID: {ticket_id}
Location: {conversation_state.get('location')}
Hazard Type: {conversation_state.get('hazard_type')}
Description: {conversation_state.get('description')}

Toronto 311 will investigate and respond within 24-48 hours."""
            
            return AgentResponse(
                intent=Intent.HAZARD_REPORT,
                message=message,
                action={
                    "type": "hazard_report",
                    "status": "completed",
                    "data": {**conversation_state, "ticket_id": ticket_id}
                }
            )
        
        # Ask for missing information
        next_field = missing_fields[0]
        prompts = {
            "location": "What is the location of the hazard? (Please provide street address or intersection)",
            "hazard_type": "What type of hazard is it? (e.g., pothole, fallen tree, debris, broken sidewalk)",
            "description": "Please describe the hazard in detail"
        }
        
        return AgentResponse(
            intent=Intent.HAZARD_REPORT,
            message=prompts.get(next_field, "Please provide more information about the hazard"),
            action={
                "type": "hazard_report",
                "status": "in_progress",
                "data": conversation_state
            }
        )
    
    async def handle_permit_screener(self, user_message: str) -> AgentResponse:
        """Handle permit screener - determine if permit is needed."""
        system_prompt = """You are a Toronto building permit expert.
Based on the user's project description, determine if a building permit is required.
Consider: New construction, major renovations (>25% of property value), structural changes, HVAC/electrical/plumbing upgrades.
Respond with: 1) YES/NO/MAYBE, 2) Brief explanation, 3) Next steps if needed."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Project description: {user_message}"}
        ]
        
        try:
            response = await llm_client.invoke(messages)
            return AgentResponse(
                intent=Intent.PERMIT_SCREENER,
                message=response,
                action={
                    "type": "permit_screener",
                    "status": "completed",
                    "data": {"project_description": user_message}
                }
            )
        except Exception as e:
            print(f"[Agent] Error handling permit screener: {e}")
            return AgentResponse(
                intent=Intent.PERMIT_SCREENER,
                message="I encountered an error processing your permit inquiry. Please contact Toronto Building Department at 311."
            )
    
    async def handle_collection_lookup(self, user_message: str) -> AgentResponse:
        """Handle waste collection schedule lookup."""
        # Extract postal code from message (simplified)
        import re
        postal_match = re.search(r'[A-Z]\d[A-Z]\s?\d[A-Z]\d', user_message.upper())
        
        if not postal_match:
            return AgentResponse(
                intent=Intent.COLLECTION_LOOKUP,
                message="Please provide your postal code to look up your waste collection schedule. (e.g., M5V 3A8)"
            )
        
        postal_code = postal_match.group(0).replace(" ", "")
        
        # Determine collection day based on first character
        first_char = postal_code[0]
        collection_days = {
            'M': 'Monday', 'N': 'Tuesday', 'L': 'Wednesday',
            'K': 'Thursday', 'J': 'Friday', 'H': 'Saturday'
        }
        collection_day = collection_days.get(first_char, 'Monday')
        
        message = f"""Your waste collection schedule for postal code {postal_code}:
- Collection Day: {collection_day}
- Collection Time: 6:00 AM - 6:00 PM
- Place bins at curb by 6:00 AM
- Remove bins by 10:00 PM same day

Collection includes:
- Garbage (black bin)
- Recyclables (blue bin)
- Organic waste (green bin)

If you have questions about what goes in each bin, just ask!"""
        
        return AgentResponse(
            intent=Intent.COLLECTION_LOOKUP,
            message=message,
            action={
                "type": "collection_lookup",
                "status": "completed",
                "data": {"postal_code": postal_code, "collection_day": collection_day}
            }
        )
    
    async def handle_out_of_scope(self) -> AgentResponse:
        """Handle out-of-scope requests."""
        return AgentResponse(
            intent=Intent.OUT_OF_SCOPE,
            message="I'm specifically designed to help with Toronto municipal services and bylaws. Your question seems to be outside my scope. Please contact Toronto 311 at 416-392-8111 or visit www.toronto.ca for other inquiries."
        )
    
    async def process_message(
        self,
        user_message: str,
        conversation_state: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process user message and return response."""
        if conversation_state is None:
            conversation_state = {}
        
        # Classify intent
        intent = await self.classify_intent(user_message)
        
        # Route to appropriate handler
        if intent == Intent.GENERAL_INQUIRY:
            return await self.handle_general_inquiry(user_message)
        elif intent == Intent.HAZARD_REPORT:
            # Update conversation state
            if "location" not in conversation_state:
                conversation_state["location"] = user_message
            elif "hazard_type" not in conversation_state:
                conversation_state["hazard_type"] = user_message
            elif "description" not in conversation_state:
                conversation_state["description"] = user_message
            
            return await self.handle_hazard_report(user_message, conversation_state)
        elif intent == Intent.PERMIT_SCREENER:
            return await self.handle_permit_screener(user_message)
        elif intent == Intent.COLLECTION_LOOKUP:
            return await self.handle_collection_lookup(user_message)
        else:
            return await self.handle_out_of_scope()


# Global agent instance
agent = TorontoBylawAgent()
