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
        self.vector_store: KnowledgeStore | None = None
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant for Fidelity. "
                "The user is interacting via voice. "
                "Provide information about Fidelity investment products, account services, "
                "retirement planning, and financial guidance. "
                "Keep responses concise and clear, without complex formatting or punctuation. "
                "For questions outside Fidelity services, politely say you can only help with Fidelity-related questions."
            )
        )

    async def preload_knowledge(self, max_pages: int, force_refresh: bool) -> None:
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
        except Exception as e:
            logger.error(f"Failed to preload knowledge: {e}", exc_info=True)

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """Inject relevant knowledge as system context before the LLM replies."""
        if self.vector_store is None or not new_message.text_content:
            return
        results = await self.vector_store.search(new_message.text_content, top_k=3)
        if results:
            context = "\n".join(
                f"- {r['text']}" for r in results if r.get("text", "").strip()
            )
            turn_ctx.add_message(
                role="system",
                content=f"Relevant Fidelity information:\n{context}",
            )

    @function_tool()
    async def search_knowledge_base(self, context: RunContext, query: str, top_k: int = 3):
        """Search Fidelity knowledge base for information.

        Args:
            query: The search query
            top_k: Number of top results to return (default: 3)
        """
        if self.vector_store is None:
            return "Knowledge base is not ready."
        results = await self.vector_store.search(query, top_k)
        if not results:
            return "No specific information found. Try asking about Fidelity investment products, account services, or retirement planning."
        return "Here is what I found:\n\n" + "\n".join(f"- {r['text']}" for r in results)

    @function_tool()
    async def refresh_knowledge_base(self, context: RunContext):
        """Refresh the knowledge base with the latest content from Fidelity website."""
        if self.vector_store is None:
            return "Knowledge base is not ready."
        try:
            await self.vector_store.scrape_website(max_pages=20)
            return "Knowledge base refreshed successfully."
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {e}")
            return "Failed to refresh. Please try again later."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    preload_max_pages = int(os.getenv("KNOWLEDGE_PRELOAD_MAX_PAGES", "100"))
    force_refresh = os.getenv("KNOWLEDGE_FORCE_REFRESH", "false").strip().lower() in {"1", "true", "yes"}

    _bnc = getattr(noise_cancellation, "BVCTelephony", None)
    nc = _bnc() if _bnc and os.getenv("TELEPHONY_MODE", "auto") != "off" else noise_cancellation.BVC()

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    assistant = Assistant()

    # Connect, then preload knowledge and wait for the participant concurrently
    # so knowledge is ready the moment the user picks up.
    await ctx.connect()
    await asyncio.gather(
        assistant.preload_knowledge(max_pages=preload_max_pages, force_refresh=force_refresh),
        ctx.wait_for_participant(),
    )

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=nc),
    )

    # Greet after session.start() so the audio pipeline is fully ready.
    await session.say("Hello! Thank you for calling Fidelity. How can I help you today?")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="telephone-agent",
            initialize_process_timeout=120.0,
        )
    )
