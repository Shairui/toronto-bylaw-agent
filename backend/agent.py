"""
agent.py — Core conversational logic for the Toronto Bylaw Agent.

All responses are LLM-generated with RAG context injected into the system prompt.
No rule-based fallbacks — if the LLM is unavailable, handlers return a 311 redirect.

Key components
--------------
SYSTEM_PROMPT_TEMPLATE   System prompt injected into every LLM call. Includes
                         retrieved RAG context, intent objective, guardrail rules,
                         multi-turn protocol, and formatting instructions.

_NON_TORONTO_CITIES      Cities rejected by the geographic guardrail (pre-LLM).
_OOS_PATTERNS            Regex for clearly off-topic queries (weather, politics, etc.).
_RAG_THRESHOLD           Cosine distance cutoff (0.80) — RAG docs above this are
                         dropped before being sent as context to the LLM.

Intent (enum)            GENERAL_INQUIRY | HAZARD_REPORT | PERMIT_SCREENER
                         | COLLECTION_LOOKUP | OUT_OF_SCOPE

AgentResponse            Return type: intent, message, action dict, citations list.
                         action carries multi-turn state consumed by the Streamlit UI.

TorontoBylawAgent
  Guardrails             _is_prompt_injection, _is_non_toronto_location,
                         _is_out_of_scope_topic  (all run before LLM classification)
  LLM helpers            _llm_respond (RAG retrieval + system prompt assembly + LLM call)
                         _extract_ticket_data (parse ticket/location/hazard from response)
  Classifier             classify_intent (LLM call; falls back to GENERAL_INQUIRY on error)
  Handlers               handle_general_inquiry, handle_hazard_report,
                         handle_permit_screener, handle_collection_lookup,
                         handle_out_of_scope
  Router                 process_message (guardrails → multi-turn state → classify → dispatch)
"""
import re
from typing import Dict, Any, List, Optional
from enum import Enum
from backend.llm import llm_client
from backend.rag import rag


# Plan A system prompt
# Injected as the first message in every LLM call.
# {objective} → intent label shown to the model.
# {retrieved_context} → top-3 RAG documents for grounding.

SYSTEM_PROMPT_TEMPLATE = """\
Classify the user's message into exactly one of these three categories:
- hazard_reporter: reporting a safety hazard, dangerous condition, or city infrastructure problem
- permit_screener: asking about building permits, construction permits, or renovation approvals
- collection_lookup: asking about waste disposal, recycling, garbage bins, or how to throw something away

You are the Toronto City Bylaw Agent.
Current Category: {objective}
You are the official City of Toronto Municipal Assistant.
1. STRICT GROUNDING: Use ONLY the provided context to answer.
2. GEOGRAPHIC LIMIT: You only provide information for the city of Toronto.
3. If the user asks about Vancouver, Montreal, Hamilton, or any non-Toronto location, politely refuse and state you only serve Toronto.
4. If the provided context does not mention the specific street or item the user asked for, state clearly that no record was found.
5. Use ONLY the provided context:
{retrieved_context}

STRICT OPERATING INSTRUCTIONS:
1. You have access to a local database of city records in the CONTEXT above.
2. If the CONTEXT contains information about a permit, address, or waste item, you MUST use that data.
3. DO NOT say you don't have access to real-time data if the information is present in the context.

MULTI-TURN ACTION PROTOCOL:
- If the user is reporting a Hazard:
    1. Check USER QUERY and HISTORY for a location (street name / postal code) AND a hazard type.
    2. If BOTH are present, immediately issue a MOCK service request: SR-2026-XXXXX.
    3. If location is missing, ask for it. If hazard type is missing, ask for it.
    4. Never ask for information already supplied in the conversation.

STRICT GUARDRAILS:
1. Only discuss Toronto municipal services (Hazards, Permits, Waste).
2. For out-of-scope queries (weather, legal advice, politics, non-Toronto cities), politely decline.
3. Only provide information grounded in the database context above.

FORMATTING RULES:
- Use short paragraphs separated by blank lines. Never write a wall of text.
- Use bullet points (- item) for lists of 3 or more items.
- Use **bold** for key terms, ticket numbers, and important values.
- Keep each paragraph to 2-3 sentences maximum.

Citations: Always indicate if information came from the Hazard, Permit, or Waste database.
"""

# Geographic + topic guardrail data

# Non-Toronto city list (geographic guardrail)

_NON_TORONTO_CITIES = [
    "hamilton", "vancouver", "montreal", "ottawa", "calgary", "edmonton",
    "winnipeg", "kitchener", "waterloo", "mississauga", "brampton", "markham",
    "richmond hill", "vaughan", "oakville", "burlington", "oshawa", "pickering",
    "ajax", "whitby", "barrie", "kingston", "windsor", "london ontario",
]

# Out-of-scope topic patterns

_OOS_PATTERNS = re.compile(
    r"\bweather\s+(like|today|forecast|in|for)\b"
    r"|what.s the weather"
    r"|\bweather\b.*\btoday\b"
    r"|\bstock\s+(price|market)\b"
    r"|\brecipe\b|\bcooking\b"
    r"|\bsports?\s+score\b"
    r"|\belection\b|\bpolitics\b|\bvote\b"
    r"|\bcapital of\b"           # general geography trivia
    r"|\bwho\s+is\s+the\s+(president|prime minister|king|queen|ceo)\b"
    r"|\btranslate\b|\btranslation\b",
    re.IGNORECASE,
)


class Intent(str, Enum):
    GENERAL_INQUIRY = "general_inquiry"
    HAZARD_REPORT = "hazard_report"
    PERMIT_SCREENER = "permit_screener"
    COLLECTION_LOOKUP = "collection_lookup"
    OUT_OF_SCOPE = "out_of_scope"


_INTENT_OBJECTIVE = {
    Intent.HAZARD_REPORT: "hazard_reporter",
    Intent.PERMIT_SCREENER: "permit_screener",
    Intent.COLLECTION_LOOKUP: "collection_lookup",
    Intent.GENERAL_INQUIRY: "general_inquiry",
    Intent.OUT_OF_SCOPE: "out_of_scope",
}


class AgentResponse:
    def __init__(
        self,
        intent: Intent,
        message: str,
        action: Optional[Dict[str, Any]] = None,
        citations: Optional[List[Dict[str, str]]] = None,
    ):
        self.intent = intent
        self.message = message
        self.action = action
        self.citations = citations or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "message": self.message,
            "action": self.action,
            "citations": self.citations,
        }


_INJECTION_RE = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions"
    r"|forget\s+(all\s+)?previous\s+instructions"
    r"|disregard\s+(all\s+)?previous"
    r"|new\s+instructions?\s*:"
    r"|you\s+are\s+now\s+(a\s+)?(?!toronto)"
    r"|act\s+as\s+(?!toronto)"
    r"|pretend\s+(you\s+are|to\s+be)\s+(?!toronto)"
    r"|jailbreak|dan\s+mode|developer\s+mode"
    r"|system\s+prompt|reveal\s+(your\s+)?instructions|repeat\s+(the\s+)?above",
    re.IGNORECASE,
)

_TICKET_RE = re.compile(r"\bSR-\d{4}-\w+\b", re.IGNORECASE)


class TorontoBylawAgent:

    # Guardrails
    # All four checks run before intent classification in process_message.
    # They cannot be bypassed by clever phrasing because they run first.

    def _is_prompt_injection(self, message: str) -> bool:
        return bool(_INJECTION_RE.search(message))

    @staticmethod
    def _is_non_toronto_location(message: str) -> bool:
        """Return True if the message references a non-Toronto city."""
        msg = message.lower()
        return any(city in msg for city in _NON_TORONTO_CITIES)

    @staticmethod
    def _is_out_of_scope_topic(message: str) -> bool:
        """Return True if the message is about an off-topic subject (weather, etc.)."""
        return bool(_OOS_PATTERNS.search(message))

    # Intent classifier
    # Plan A: LLM call with few-shot examples (classify_intent).
    # Plan B: ordered keyword rules (_keyword_classify).
    #   Priority: geographic guardrail → topic guardrail → hazard → permit → waste → general

    # Unified LLM call (Plan A)

    # RAG cosine-distance threshold — documents above this are too dissimilar to use.
    _RAG_THRESHOLD = 0.80

    async def _llm_respond(
        self,
        intent: Intent,
        user_message: str,
        chat_history: List[Dict[str, str]],
        extra_query: str = "",
    ) -> tuple:
        objective = _INTENT_OBJECTIVE.get(intent, "general_inquiry")
        search_query = f"{user_message} {extra_query}".strip()
        raw_results = rag.search(search_query, top_k=3)
        search_results = [r for r in raw_results if r.get("distance", 0.0) < self._RAG_THRESHOLD]

        context_parts, citations = [], []
        for doc in search_results:
            cat = doc.get("category", "General")
            context_parts.append(f"[{cat} Database] {doc['title']}:\n{doc['content']}")
            citations.append({
                "title": doc["title"],
                "source": doc["source"],
                "excerpt": doc["content"][:250] + "...",
            })

        retrieved_context = (
            "\n\n".join(context_parts)
            if context_parts
            else "No specific records found in the database for this query."
        )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            objective=objective,
            retrieved_context=retrieved_context,
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history or [])
        messages.append({"role": "user", "content": user_message})

        response_text = await llm_client.invoke(messages)
        return response_text, citations

    # Intent classification

    async def classify_intent(
        self, user_message: str, chat_history: List[Dict[str, str]] = None
    ) -> Intent:
        classify_system = (
            "You are an intent classifier for a Toronto municipal services agent.\n"
            "Classify the user's message (considering conversation history) into exactly one of:\n"
            '- "hazard_report": User is reporting a hazard or safety problem\n'
            '- "permit_screener": User is asking about building/construction permits\n'
            '- "collection_lookup": User is asking about waste disposal or garbage collection\n'
            '- "general_inquiry": Questions about Toronto bylaws or general services\n'
            '- "out_of_scope": Unrelated to Toronto municipal services, or references non-Toronto cities\n\n'
            "If history shows an ongoing hazard report, classify follow-up messages as hazard_report.\n"
            "Respond with ONLY the intent name, nothing else."
        )
        messages = [{"role": "system", "content": classify_system}]
        messages.extend(chat_history or [])
        messages.append({"role": "user", "content": user_message})
        try:
            response = await llm_client.invoke(messages, temperature=0.1, max_tokens=20)
            intent_str = response.lower().strip()
            for intent in Intent:
                if intent.value in intent_str:
                    return intent
            return Intent.GENERAL_INQUIRY
        except Exception as e:
            print(f"[Agent] LLM classify error: {e}")
            return Intent.GENERAL_INQUIRY

    # Handlers
    # All handlers return an AgentResponse with intent, message, action, citations.

    async def handle_general_inquiry(
        self, user_message: str, chat_history: List[Dict[str, str]] = None
    ) -> AgentResponse:
        try:
            response_text, citations = await self._llm_respond(
                Intent.GENERAL_INQUIRY, user_message, chat_history or []
            )
            return AgentResponse(Intent.GENERAL_INQUIRY, response_text, citations=citations)
        except Exception as e:
            print(f"[Agent] LLM general inquiry error: {e}")
            return AgentResponse(
                intent=Intent.GENERAL_INQUIRY,
                message=(
                    "I'm unable to process your request right now. "
                    "Please call **311** or visit **toronto.ca/311** for assistance."
                ),
            )

    @staticmethod
    def _extract_ticket_data(response_text: str, chat_history: List[Dict], user_message: str) -> Dict[str, str]:
        """Parse ticket ID, location, and hazard type from the LLM response + conversation."""
        data: Dict[str, str] = {}

        # Ticket ID
        m = _TICKET_RE.search(response_text)
        if m:
            data["ticket_id"] = m.group(0)

        # Search all text for location and hazard
        all_text = user_message + " " + " ".join(
            msg["content"] for msg in (chat_history or []) if msg["role"] == "user"
        ) + " " + response_text

        # Location: street address or intersection
        for pat in [
            r"\b(\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|Avenue|Ave|Road|Rd|Blvd|Drive|Dr|St)\b[^,.\n]*)",
            r"\b([A-Z][a-z]+\s+(?:Street|Avenue|Ave|St)\s+(?:and|&)\s+[A-Z][a-z]+\s+(?:Street|Avenue|Ave|St)\b)",
        ]:
            lm = re.search(pat, all_text)
            if lm:
                data["location"] = lm.group(1).strip()
                break

        # Hazard type: pick the first known keyword found
        hazard_keywords = [
            ("fallen tree", "fallen tree"), ("falling tree", "fallen tree"),
            ("pothole", "pothole"), ("flooding", "flooding"), ("flood", "flooding"),
            ("graffiti", "graffiti"), ("broken sidewalk", "broken sidewalk"),
            ("broken traffic sign", "broken traffic sign"), ("traffic sign", "broken traffic sign"),
            ("streetlight", "broken streetlight"), ("street light", "broken streetlight"),
            ("noise", "noise disturbance"), ("debris", "debris on road"),
            ("water main", "water main break"), ("fire hydrant", "damaged fire hydrant"),
        ]
        lower = all_text.lower()
        for kw, label in hazard_keywords:
            if kw in lower:
                data["hazard_type"] = label
                break

        return data

    async def handle_hazard_report(
        self,
        user_message: str,
        conversation_state: Dict[str, Any],
        chat_history: List[Dict[str, str]] = None,
    ) -> AgentResponse:
        try:
            response_text, citations = await self._llm_respond(
                Intent.HAZARD_REPORT, user_message, chat_history or [],
                extra_query="hazard safety toronto 311"
            )
            has_ticket = bool(_TICKET_RE.search(response_text))
            if has_ticket:
                extracted = self._extract_ticket_data(response_text, chat_history, user_message)
                action = {"type": "hazard_report", "status": "completed", "data": extracted}
            else:
                action = {"type": "hazard_report", "status": "in_progress",
                          "data": {**conversation_state, "awaiting": "location"}}
            return AgentResponse(Intent.HAZARD_REPORT, response_text, action=action, citations=citations)
        except Exception as e:
            print(f"[Agent] LLM hazard error: {e}")
            return AgentResponse(
                intent=Intent.HAZARD_REPORT,
                message=(
                    "I'm unable to process your hazard report right now. "
                    "Please call **311** directly to report the hazard."
                ),
            )

    async def handle_permit_screener(
        self, user_message: str, chat_history: List[Dict[str, str]] = None
    ) -> AgentResponse:
        try:
            response_text, citations = await self._llm_respond(
                Intent.PERMIT_SCREENER, user_message, chat_history or [],
                extra_query="building permit toronto construction"
            )
            return AgentResponse(
                intent=Intent.PERMIT_SCREENER,
                message=response_text,
                action={"type": "permit_screener", "status": "completed",
                        "data": {"project_description": user_message}},
                citations=citations,
            )
        except Exception as e:
            print(f"[Agent] LLM permit error: {e}")
            return AgentResponse(
                intent=Intent.PERMIT_SCREENER,
                message=(
                    "I'm unable to check permit requirements right now. "
                    "Please visit **toronto.ca/building** or call **311**."
                ),
            )

    async def handle_collection_lookup(
        self, user_message: str, chat_history: List[Dict[str, str]] = None
    ) -> AgentResponse:
        try:
            response_text, citations = await self._llm_respond(
                Intent.COLLECTION_LOOKUP, user_message, chat_history or [],
                extra_query="waste disposal recycling toronto"
            )
            return AgentResponse(
                intent=Intent.COLLECTION_LOOKUP,
                message=response_text,
                citations=citations,
                action={"type": "collection_lookup", "status": "completed", "data": {}},
            )
        except Exception as e:
            print(f"[Agent] LLM collection error: {e}")
            return AgentResponse(
                intent=Intent.COLLECTION_LOOKUP,
                message=(
                    "I'm unable to look up waste collection information right now. "
                    "Please visit **toronto.ca/garbage** or call **311**."
                ),
            )

    async def handle_out_of_scope(self, reason: str = "") -> AgentResponse:
        if reason == "geography":
            msg = (
                "I'm only able to assist with municipal services within the **City of Toronto**. "
                "The location you mentioned is outside Toronto's jurisdiction.\n\n"
                "For services in that area, please contact the local municipality directly, "
                "or call your regional 311 service."
            )
        else:
            msg = (
                "I'm designed specifically to help with **Toronto municipal services and bylaws**. "
                "Your question appears to be outside my scope.\n\n"
                "**What I can help with:**\n"
                "- Reporting a hazard (pothole, fallen tree, etc.)\n"
                "- Checking if a building permit is required\n"
                "- Waste disposal and collection schedules\n\n"
                "For other inquiries, call **311** or visit **toronto.ca/311**."
            )
        return AgentResponse(intent=Intent.OUT_OF_SCOPE, message=msg)

    # Main router (process_message)
    # Execution order:
    #   1. Prompt-injection guardrail
    #   2. Geographic guardrail (non-Toronto cities)
    #   3. Topic guardrail (weather, politics, etc.)
    #   4. Multi-turn state continuation
    #      - "awaiting": "waste_postal_code"  → handle_collection_lookup
    #      - "awaiting": <hazard field>       → handle_hazard_report
    #   5. Classify intent → dispatch to handler

    async def process_message(
        self,
        user_message: str,
        conversation_state: Dict[str, Any] = None,
        chat_history: List[Dict[str, str]] = None,
    ) -> AgentResponse:
        if conversation_state is None:
            conversation_state = {}
        if chat_history is None:
            chat_history = []

        # 1. Prompt-injection guard
        if self._is_prompt_injection(user_message):
            return AgentResponse(
                intent=Intent.OUT_OF_SCOPE,
                message=(
                    "I'm sorry — I can only assist with Toronto municipal services. "
                    "I'm not able to change my instructions or operate outside my designated role."
                ),
            )

        # 2. Geographic guardrail (before classification so it can't be bypassed)
        if self._is_non_toronto_location(user_message):
            return await self.handle_out_of_scope(reason="geography")

        # 3. Topic guardrail (weather, politics, etc.)
        if self._is_out_of_scope_topic(user_message):
            return await self.handle_out_of_scope()

        # 4. Continue active hazard multi-turn flows
        if "awaiting" in conversation_state:
            field = conversation_state["awaiting"]
            conversation_state.pop("awaiting")
            conversation_state[field] = user_message
            return await self.handle_hazard_report(user_message, conversation_state, chat_history)

        # 5. Classify and route
        intent = await self.classify_intent(user_message, chat_history)

        if intent == Intent.GENERAL_INQUIRY:
            return await self.handle_general_inquiry(user_message, chat_history)
        elif intent == Intent.HAZARD_REPORT:
            return await self.handle_hazard_report(user_message, conversation_state, chat_history)
        elif intent == Intent.PERMIT_SCREENER:
            return await self.handle_permit_screener(user_message, chat_history)
        elif intent == Intent.COLLECTION_LOOKUP:
            return await self.handle_collection_lookup(user_message, chat_history)
        else:
            return await self.handle_out_of_scope()


agent = TorontoBylawAgent()
