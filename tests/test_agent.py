"""Tests for Toronto Bylaw Agent."""
import pytest
import asyncio
from backend.agent import agent, Intent


@pytest.mark.asyncio
async def test_intent_classification_general_inquiry():
    """Test intent classification for general inquiry."""
    message = "What are the zoning requirements for residential properties?"
    intent = await agent.classify_intent(message)
    assert intent == Intent.GENERAL_INQUIRY


@pytest.mark.asyncio
async def test_intent_classification_hazard_report():
    """Test intent classification for hazard report."""
    message = "There's a large pothole on Queen Street that needs to be fixed"
    intent = await agent.classify_intent(message)
    assert intent == Intent.HAZARD_REPORT


@pytest.mark.asyncio
async def test_intent_classification_permit_screener():
    """Test intent classification for permit screener."""
    message = "Do I need a permit to renovate my kitchen?"
    intent = await agent.classify_intent(message)
    assert intent == Intent.PERMIT_SCREENER


@pytest.mark.asyncio
async def test_intent_classification_collection_lookup():
    """Test intent classification for collection lookup."""
    message = "When is my garbage collection day for postal code M5V 3A8?"
    intent = await agent.classify_intent(message)
    assert intent == Intent.COLLECTION_LOOKUP


@pytest.mark.asyncio
async def test_general_inquiry_response():
    """Test general inquiry response."""
    message = "What is the Toronto Municipal Code?"
    response = await agent.handle_general_inquiry(message)
    
    assert response.intent == Intent.GENERAL_INQUIRY
    assert len(response.message) > 0
    assert isinstance(response.citations, list)


@pytest.mark.asyncio
async def test_hazard_report_multi_turn():
    """Test hazard report multi-turn dialogue."""
    # First turn - ask for location
    response1 = await agent.handle_hazard_report("There's a hazard", {})
    assert response1.action["status"] == "in_progress"
    
    # Second turn - provide location
    state = response1.action["data"]
    state["location"] = "Queen Street and Spadina Avenue"
    response2 = await agent.handle_hazard_report("Queen Street and Spadina Avenue", state)
    assert response2.action["status"] == "in_progress"
    
    # Third turn - provide hazard type
    state = response2.action["data"]
    state["hazard_type"] = "Pothole"
    response3 = await agent.handle_hazard_report("Pothole", state)
    assert response3.action["status"] == "in_progress"
    
    # Fourth turn - provide description
    state = response3.action["data"]
    state["description"] = "Large pothole about 2 feet wide"
    response4 = await agent.handle_hazard_report("Large pothole about 2 feet wide", state)
    assert response4.action["status"] == "completed"
    assert "ticket_id" in response4.action["data"]


@pytest.mark.asyncio
async def test_permit_screener_response():
    """Test permit screener response."""
    message = "I want to add a second floor to my house"
    response = await agent.handle_permit_screener(message)
    
    assert response.intent == Intent.PERMIT_SCREENER
    assert len(response.message) > 0
    assert response.action["type"] == "permit_screener"


@pytest.mark.asyncio
async def test_collection_lookup_response():
    """Test collection lookup response."""
    message = "What's my waste collection day for M5V 3A8?"
    response = await agent.handle_collection_lookup(message)
    
    assert response.intent == Intent.COLLECTION_LOOKUP
    assert "collection_day" in response.action["data"]
    assert response.action["data"]["postal_code"] == "M5V3A8"


@pytest.mark.asyncio
async def test_out_of_scope_response():
    """Test out-of-scope response."""
    response = await agent.handle_out_of_scope()
    
    assert response.intent == Intent.OUT_OF_SCOPE
    assert "Toronto 311" in response.message


@pytest.mark.asyncio
async def test_process_message_general():
    """Test process_message for general inquiry."""
    message = "Tell me about Toronto zoning bylaws"
    response = await agent.process_message(message)
    
    assert response.intent in [Intent.GENERAL_INQUIRY, Intent.OUT_OF_SCOPE]
    assert len(response.message) > 0


@pytest.mark.asyncio
async def test_process_message_hazard():
    """Test process_message for hazard report."""
    message = "I need to report a hazard"
    response = await agent.process_message(message)
    
    assert response.intent == Intent.HAZARD_REPORT
    assert response.action["status"] in ["in_progress", "completed"]


@pytest.mark.asyncio
async def test_process_message_permit():
    """Test process_message for permit screening."""
    message = "Do I need a permit for my renovation?"
    response = await agent.process_message(message)
    
    assert response.intent == Intent.PERMIT_SCREENER
    assert len(response.message) > 0


@pytest.mark.asyncio
async def test_process_message_collection():
    """Test process_message for collection lookup."""
    message = "When is garbage day for M5V 3A8?"
    response = await agent.process_message(message)
    
    assert response.intent == Intent.COLLECTION_LOOKUP
    assert "collection_day" in response.action["data"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
