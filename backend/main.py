"""FastAPI backend for Toronto Bylaw Agent."""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
import asyncio

from backend.database import init_db, get_db
from backend.agent import agent
from backend.rag import rag
from backend.models import Conversation, Message, Action, User
from backend.config import BACKEND_HOST, BACKEND_PORT

# Initialize database
init_db()

# Initialize knowledge base
rag.initialize_knowledge_base()

# Create FastAPI app
app = FastAPI(title="Toronto Bylaw Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class CreateConversationRequest(BaseModel):
    """Request to create a conversation."""
    title: Optional[str] = None


class SendMessageRequest(BaseModel):
    """Request to send a message."""
    conversation_id: int
    message: str


class ConversationResponse(BaseModel):
    """Response with conversation details."""
    id: int
    title: Optional[str]
    created_at: str


class MessageResponse(BaseModel):
    """Response with message details."""
    id: int
    role: str
    content: str
    metadata: Dict[str, Any]
    created_at: str


class AgentResponseModel(BaseModel):
    """Response from agent."""
    intent: str
    message: str
    action: Optional[Dict[str, Any]] = None
    citations: Optional[list] = None


# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    try:
        # Create or get default user
        user = db.query(User).first()
        if not user:
            user = User(name="Default User", email="default@toronto.ca")
            db.add(user)
            db.commit()
        
        # Create conversation
        conversation = Conversation(
            user_id=user.id,
            title=request.title or "New Conversation"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at.isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: int,
    request: SendMessageRequest,
    db: Session = Depends(get_db)
):
    """Send a message and get agent response."""
    try:
        # Get conversation
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get conversation state
        conversation_state = conversation.state or {}
        
        # Process message with agent
        agent_response = await agent.process_message(
            request.message,
            conversation_state
        )
        
        # Save user message
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=request.message,
            metadata={"intent": agent_response.intent.value}
        )
        db.add(user_msg)
        
        # Save assistant response
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=agent_response.message,
            metadata={
                "intent": agent_response.intent.value,
                "citations": agent_response.citations
            }
        )
        db.add(assistant_msg)
        
        # Save action if present
        if agent_response.action:
            action = Action(
                conversation_id=conversation_id,
                action_type=agent_response.action["type"],
                status=agent_response.action["status"],
                data=agent_response.action["data"]
            )
            db.add(action)
        
        # Update conversation state
        if agent_response.action and agent_response.action["type"] == "hazard_report":
            conversation.state = agent_response.action["data"]
        
        db.commit()
        
        return agent_response.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error sending message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get all messages in a conversation."""
    try:
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at).all()
        
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "metadata": msg.metadata,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
async def list_conversations(db: Session = Depends(get_db)):
    """List all conversations."""
    try:
        user = db.query(User).first()
        if not user:
            return []
        
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user.id
        ).order_by(Conversation.created_at.desc()).all()
        
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat()
            }
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
