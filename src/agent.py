import asyncio
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from utils.knowledge_store import KnowledgeStore

logger = logging.getLogger("agent")

load_dotenv(".env.local", override=True)


class Assistant(Agent):
    def __init__(self) -> None:
        self.vector_store = None
        self._initializing = False
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant for Fidelity. "
                "The user is interacting with you via voice. "
                "You can provide information about Fidelity's investment products, account services, "
                "retirement planning, and financial guidance. "
                "Keep responses concise and clear, without complex formatting or punctuation. "
                "If asked about something outside Fidelity's services, politely say you can only "
                "help with Fidelity-related questions."
            )
        )

    async def preload_knowledge(self, max_pages: int, force_refresh: bool) -> bool:
        if self.vector_store is None:
            self.vector_store = KnowledgeStore()

        try:
            await asyncio.wait_for(
                self.vector_store.initialize(
                    preload_website=True,
                    max_pages=max_pages,
                    force_refresh=force_refresh,
                ),
                timeout=600,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to preload knowledge store: {e}", exc_info=True)
            return False

    async def _ensure_vector_store(self) -> bool:
        if self.vector_store is not None:
            return True
        if self._initializing:
            return False
        self._initializing = True
        try:
            self.vector_store = KnowledgeStore()
            await asyncio.wait_for(self.vector_store.initialize(), timeout=60)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            self.vector_store = None
            return False
        finally:
            self._initializing = False

    async def on_session_started(self, session: AgentSession) -> None:
        await super().on_session_started(session)
        await self._ensure_vector_store()
        await session.say(
            "Hello! Thank you for calling Fidelity. How can I help you today?"
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        if not await self._ensure_vector_store():
            turn_ctx.add_message(
                role="assistant",
                content="I'm sorry, my knowledge base is not ready yet. Please try again in a moment.",
            )
            return

        query = new_message.text_content
        if not query:
            return

        try:
            results = await self.vector_store.search(query, top_k=3)
            if not results:
                turn_ctx.add_message(
                    role="assistant",
                    content=(
                        "I don't have that specific information available. "
                        "You can ask me about Fidelity's investment products, account services, "
                        "or retirement planning."
                    ),
                )
                return

            response = "Based on Fidelity's information:\n"
            for doc in results:
                if doc.get("text", "").strip():
                    response += f"- {doc['text']}\n"
            turn_ctx.add_message(role="assistant", content=response.strip())

        except Exception as e:
            logger.error(f"Error handling user message: {e}", exc_info=True)
            turn_ctx.add_message(
                role="assistant",
                content="I encountered an error. Please try asking your question again.",
            )

    @function_tool()
    async def search_knowledge_base(
        self, context: RunContext, query: str, top_k: int = 3
    ):
        """Search Fidelity's knowledge base for information.

        Args:
            query: The search query
            top_k: Number of top results to return (default: 3)
        """
        if not await self._ensure_vector_store():
            return "Knowledge base is not ready. Please try again shortly."

        results = await self.vector_store.search(query, top_k)
        if not results:
            return (
                "I couldn't find specific information about that. "
                "Try asking about Fidelity's investment products, account services, or retirement planning."
            )

        response = "Here's what I found:\n\n"
        for result in results:
            response += f"- {result['text']}\n"
        return response

    @function_tool()
    async def refresh_knowledge_base(self, context: RunContext):
        """Refresh the knowledge base with the latest content from Fidelity's website."""
        if not await self._ensure_vector_store():
            return "Knowledge base is not ready. Please try again shortly."
        try:
            await self.vector_store.scrape_website(max_pages=20)
            return "Knowledge base refreshed successfully."
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {e}")
            return "An error occurred while refreshing. Please try again later."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    preload_max_pages = int(os.getenv("KNOWLEDGE_PRELOAD_MAX_PAGES", "100"))
    force_refresh = os.getenv("KNOWLEDGE_FORCE_REFRESH", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    telephony_mode = os.getenv("TELEPHONY_MODE", "auto").strip().lower()
    telephony_noise_cancellation = getattr(noise_cancellation, "BVCTelephony", None)
    selected_noise_cancellation = (
        telephony_noise_cancellation()
        if telephony_mode != "off" and telephony_noise_cancellation is not None
        else noise_cancellation.BVC()
    )

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    assistant = Assistant()
    await assistant.preload_knowledge(
        max_pages=preload_max_pages,
        force_refresh=force_refresh,
    )

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=selected_noise_cancellation,
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="telephone-agent",
            initialize_process_timeout=120.0,
        )
    )
