"""Agent logic for Toronto Bylaw Conversational Agent.

Plan A (LLM-powered): active when LLM_API_KEY is set in the environment.
Plan B (rule-based):   active automatically when no API key is available.
Switch from B → A: set the LLM_API_KEY env var and restart the server.
"""
import re
from typing import Dict, Any, List, Optional
from enum import Enum
from backend.llm import llm_client
from backend.rag import rag
from backend.config import USE_LLM


class Intent(str, Enum):
    GENERAL_INQUIRY = "general_inquiry"
    HAZARD_REPORT = "hazard_report"
    PERMIT_SCREENER = "permit_screener"
    COLLECTION_LOOKUP = "collection_lookup"
    OUT_OF_SCOPE = "out_of_scope"


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


# ── Prompt injection patterns ────────────────────────────────────────────────
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


class TorontoBylawAgent:

    # ── Guardrail ────────────────────────────────────────────────────────────

    def _is_prompt_injection(self, message: str) -> bool:
        return bool(_INJECTION_RE.search(message))

    # ── Keyword helpers ──────────────────────────────────────────────────────

    def _keyword_classify(self, message: str) -> Intent:
        """Keyword-based intent classification (Plan B fallback)."""
        msg = message.lower()
        if any(k in msg for k in [
            "report", "hazard", "pothole", "fallen tree", "debris",
            "broken sidewalk", "flooding", "incident", "danger", "unsafe",
            "damage", "crack", "leak", "spill", "graffiti",
        ]):
            return Intent.HAZARD_REPORT
        if any(k in msg for k in [
            "permit", "renovation", "renovate", "build", "construction",
            "deck", "addition", "demolish", "pool", "garage", "fence",
            "basement", "extension", "structural",
        ]):
            return Intent.PERMIT_SCREENER
        if any(k in msg for k in [
            "garbage", "collection", "waste", "recycling", "recycle",
            "green bin", "blue bin", "pickup", "pick up", "compost",
            "trash", "rubbish", "bin", "bulk",
        ]):
            return Intent.COLLECTION_LOOKUP
        # Default: try to answer using the knowledge base
        return Intent.GENERAL_INQUIRY

    def _rule_based_permit(self, message: str) -> tuple:
        """Return (decision, explanation, steps) without using LLM."""
        msg = message.lower()
        yes_kw = [
            "new construction", "new house", "new building", "add floor",
            "second floor", "third floor", "addition", "structural",
            "demolish", "pool", "major renovation", "new garage",
            "new deck", "extend", "knock down", "tear down",
        ]
        no_kw = [
            "paint", "painting", "wallpaper", "flooring", "carpet",
            "landscaping", "garden", "minor repair", "replace fixture",
            "clean", "window cleaning", "caulk", "weather strip",
        ]
        if any(k in msg for k in yes_kw):
            decision = "YES — A building permit is required."
            explanation = (
                "Your project involves construction or significant alterations "
                "that require a building permit under Toronto Municipal Code "
                "Chapter 363 and the Ontario Building Code."
            )
            steps = (
                "1. Prepare site plan and construction drawings.\n"
                "2. Apply online at **toronto.ca/building** or call **311**.\n"
                "3. Pay the permit fee (approx. $13.73 per $1,000 of construction value).\n"
                "4. Wait for permit approval before starting work."
            )
        elif any(k in msg for k in no_kw):
            decision = "NO — A building permit is likely not required."
            explanation = (
                "The work you described (cosmetic/minor repairs) is generally "
                "exempt from building permit requirements under the Ontario Building Code."
            )
            steps = (
                "No permit needed, but ensure the work meets property standards.\n"
                "If unsure, call **311** or visit **toronto.ca/building** to confirm."
            )
        else:
            decision = "POSSIBLY — A permit may be required."
            explanation = (
                "Based on your description, a building permit may or may not be required "
                "depending on the scope and structural impact of the work."
            )
            steps = (
                "We recommend confirming with Toronto Building:\n"
                "- Online: **toronto.ca/building**\n"
                "- Phone: **311** (ask for Toronto Building)\n"
                "- In person: visit a Civic Centre"
            )
        return decision, explanation, steps

    # ── Intent classification ─────────────────────────────────────────────────

    async def classify_intent(self, user_message: str) -> Intent:
        if not USE_LLM:
            return self._keyword_classify(user_message)

        # ── Plan A: LLM classification ────────────────────────────────────
        system_prompt = (
            "You are an intent classifier for a Toronto municipal services agent.\n"
            "Classify the user's message into one of these categories:\n"
            '- "general_inquiry": Questions about Toronto bylaws, regulations, or services\n'
            '- "hazard_report": User wants to report a hazard (pothole, fallen tree, debris, etc.)\n'
            '- "permit_screener": User wants to know if they need a permit for a project\n'
            '- "collection_lookup": User wants to know waste collection schedule\n'
            '- "out_of_scope": Request is not related to Toronto municipal services\n\n'
            "Respond with ONLY the intent name, nothing else."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            response = await llm_client.invoke(messages, temperature=0.3, max_tokens=50)
            intent_str = response.lower().strip()
            for intent in Intent:
                if intent.value in intent_str:
                    return intent
            return self._keyword_classify(user_message)
        except Exception as e:
            print(f"[Agent] LLM classify error, using keyword fallback: {e}")
            return self._keyword_classify(user_message)

    # ── Handlers ─────────────────────────────────────────────────────────────

    async def handle_general_inquiry(self, user_message: str) -> AgentResponse:
        search_results = rag.search(user_message, top_k=3)

        if not search_results:
            return AgentResponse(
                intent=Intent.GENERAL_INQUIRY,
                message=(
                    "I couldn't find specific information about your query in the "
                    "Toronto municipal bylaws database. Please try rephrasing your "
                    "question or contact Toronto 311 at **416-392-8111**."
                ),
            )

        citations = [
            {
                "title": doc["title"],
                "source": doc["source"],
                "excerpt": doc["content"][:250] + "...",
            }
            for doc in search_results
        ]

        if USE_LLM:
            # ── Plan A: synthesise with LLM ───────────────────────────────
            context = "\n\n---\n\n".join(
                f"Title: {d['title']}\nContent: {d['content']}" for d in search_results
            )
            system_prompt = (
                "You are a helpful Toronto municipal services assistant. "
                "Answer the user's question based only on the provided bylaw context. "
                "Be concise and cite the relevant bylaw or section. "
                "If the context does not contain a clear answer, say so and suggest calling 311."
            )
            llm_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"},
            ]
            try:
                response = await llm_client.invoke(llm_messages)
                return AgentResponse(
                    intent=Intent.GENERAL_INQUIRY,
                    message=response,
                    citations=citations,
                )
            except Exception as e:
                print(f"[Agent] LLM general inquiry error: {e}")
                # Fall through to Plan B response

        # ── Plan B: format top RAG result directly ────────────────────────
        top = search_results[0]
        lines = [f"**{top['title']}**\n\n{top['content']}"]
        if len(search_results) > 1:
            lines.append(
                f"\n\n---\n**Also relevant — {search_results[1]['title']}**\n\n"
                f"{search_results[1]['content'][:300]}..."
            )
        lines.append(
            "\n\n*For more details, contact Toronto 311 (call **311** or visit **toronto.ca/311**).*"
        )
        return AgentResponse(
            intent=Intent.GENERAL_INQUIRY,
            message="".join(lines),
            citations=citations,
        )

    async def handle_hazard_report(
        self, user_message: str, conversation_state: Dict[str, Any] = None
    ) -> AgentResponse:
        if conversation_state is None:
            conversation_state = {}

        required_fields = ["location", "hazard_type", "description"]
        missing = [f for f in required_fields if f not in conversation_state]

        if not missing:
            ticket_id = f"311-{abs(hash(str(conversation_state))) % 100000:05d}"
            state_data = {k: v for k, v in conversation_state.items() if k != "awaiting"}
            state_data["ticket_id"] = ticket_id
            return AgentResponse(
                intent=Intent.HAZARD_REPORT,
                message=(
                    f"Your hazard report has been submitted successfully.\n\n"
                    f"**Ticket ID:** `{ticket_id}`\n"
                    f"**Location:** {state_data.get('location')}\n"
                    f"**Hazard Type:** {state_data.get('hazard_type')}\n"
                    f"**Description:** {state_data.get('description')}\n\n"
                    f"Toronto 311 will investigate within **24–48 hours**. "
                    f"You can track your request at **toronto.ca/311**."
                ),
                action={"type": "hazard_report", "status": "completed", "data": state_data},
            )

        next_field = missing[0]
        prompts = {
            "location": (
                "I can help you report that hazard to Toronto 311. "
                "What is the **location**? (street address or intersection)"
            ),
            "hazard_type": (
                "What **type of hazard** is it?\n"
                "*(e.g., pothole, fallen tree, debris, broken sidewalk, flooding)*"
            ),
            "description": "Please **describe** the hazard — size, severity, any immediate danger.",
        }
        state_data = {**conversation_state, "awaiting": next_field}
        return AgentResponse(
            intent=Intent.HAZARD_REPORT,
            message=prompts[next_field],
            action={"type": "hazard_report", "status": "in_progress", "data": state_data},
        )

    async def handle_permit_screener(self, user_message: str) -> AgentResponse:
        if USE_LLM:
            # ── Plan A: LLM analysis + RAG context ───────────────────────
            search_results = rag.search(user_message + " building permit", top_k=2)
            context = ""
            citations = []
            if search_results:
                context = "\n\n".join(
                    f"Title: {d['title']}\nContent: {d['content']}" for d in search_results
                )
                citations = [
                    {"title": d["title"], "source": d["source"], "excerpt": d["content"][:200] + "..."}
                    for d in search_results
                ]
            system_prompt = (
                "You are a Toronto building permit expert. "
                "Determine if a building permit is required for the described project. "
                "Respond with: 1) YES/NO/POSSIBLY  2) Brief explanation with bylaw reference  "
                "3) Recommended next steps."
            )
            user_content = f"Bylaw context:\n{context}\n\nProject: {user_message}" if context else f"Project: {user_message}"
            try:
                response = await llm_client.invoke(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
                )
                return AgentResponse(
                    intent=Intent.PERMIT_SCREENER,
                    message=response,
                    action={"type": "permit_screener", "status": "completed", "data": {"project_description": user_message}},
                    citations=citations,
                )
            except Exception as e:
                print(f"[Agent] LLM permit error: {e}")
                # Fall through to Plan B

        # ── Plan B: rule-based permit decision ───────────────────────────
        decision, explanation, steps = self._rule_based_permit(user_message)
        message = f"**{decision}**\n\n{explanation}\n\n**Next Steps:**\n{steps}"
        return AgentResponse(
            intent=Intent.PERMIT_SCREENER,
            message=message,
            action={
                "type": "permit_screener",
                "status": "completed",
                "data": {"project_description": user_message, "decision": decision},
            },
        )

    async def handle_collection_lookup(self, user_message: str) -> AgentResponse:
        postal_match = re.search(r"[A-Z]\d[A-Z]\s?\d[A-Z]\d", user_message.upper())
        if not postal_match:
            return AgentResponse(
                intent=Intent.COLLECTION_LOOKUP,
                message=(
                    "Please provide your **postal code** to look up your waste collection schedule.\n\n"
                    "*(e.g., M5V 3A8)*"
                ),
            )
        postal_code = postal_match.group(0).replace(" ", "")
        second_char = postal_code[1] if len(postal_code) > 1 else "5"
        day_map = {
            "1": "Monday", "2": "Tuesday", "3": "Wednesday",
            "4": "Thursday", "5": "Friday", "6": "Monday",
            "7": "Tuesday", "8": "Wednesday", "9": "Thursday",
        }
        collection_day = day_map.get(second_char, "Monday")
        display_code = f"{postal_code[:3]} {postal_code[3:]}"
        message = (
            f"Here is your waste collection schedule for **{display_code}**:\n\n"
            f"- 📅 **Collection Day:** {collection_day}\n"
            f"- 🕖 Place bins at the curb by **7:00 AM**\n"
            f"- 🌙 Remove bins by **midnight** the same day\n\n"
            f"**Weekly collection includes:**\n"
            f"- 🗑️ Garbage (black bin)\n"
            f"- ♻️ Recyclables (blue bin)\n"
            f"- 🌿 Organic waste (green bin)\n"
            f"- 🍂 Yard waste in paper bags *(April – December)*\n\n"
            f"For schedule exceptions or large-item pickup, call **311** or visit **toronto.ca/garbage**."
        )
        return AgentResponse(
            intent=Intent.COLLECTION_LOOKUP,
            message=message,
            action={
                "type": "collection_lookup",
                "status": "completed",
                "data": {"postal_code": postal_code, "collection_day": collection_day},
            },
        )

    async def handle_out_of_scope(self) -> AgentResponse:
        return AgentResponse(
            intent=Intent.OUT_OF_SCOPE,
            message=(
                "I'm designed specifically to help with **Toronto municipal services and bylaws**. "
                "Your question appears to be outside my scope.\n\n"
                "**For other inquiries:**\n"
                "- Toronto 311: call **311** (24/7) or visit **toronto.ca/311**\n"
                "- Ontario government: **ontario.ca**\n"
                "- Federal services: **canada.ca**"
            ),
        )

    # ── Main router ───────────────────────────────────────────────────────────

    async def process_message(
        self, user_message: str, conversation_state: Dict[str, Any] = None
    ) -> AgentResponse:
        if conversation_state is None:
            conversation_state = {}

        # 1. Prompt-injection guard
        if self._is_prompt_injection(user_message):
            return AgentResponse(
                intent=Intent.OUT_OF_SCOPE,
                message=(
                    "I'm sorry — I can only assist with Toronto municipal services. "
                    "I'm not able to change my instructions or operate outside my designated role."
                ),
            )

        # 2. Continue active multi-turn flow (skip re-classification)
        if "awaiting" in conversation_state:
            field = conversation_state.pop("awaiting")
            conversation_state[field] = user_message
            return await self.handle_hazard_report(user_message, conversation_state)

        # 3. Classify and route
        intent = await self.classify_intent(user_message)

        if intent == Intent.GENERAL_INQUIRY:
            return await self.handle_general_inquiry(user_message)
        elif intent == Intent.HAZARD_REPORT:
            return await self.handle_hazard_report(user_message, conversation_state)
        elif intent == Intent.PERMIT_SCREENER:
            return await self.handle_permit_screener(user_message)
        elif intent == Intent.COLLECTION_LOOKUP:
            return await self.handle_collection_lookup(user_message)
        else:
            return await self.handle_out_of_scope()


agent = TorontoBylawAgent()
