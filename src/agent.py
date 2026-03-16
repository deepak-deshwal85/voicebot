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

from utils.knowledge_store import KnowledgeStore

logger = logging.getLogger("agent")

load_dotenv(".env.local", override=True)


class Assistant(Agent):
    def __init__(self, preemptive_generation: bool) -> None:
        self.vector_store: KnowledgeStore | None = None
        self.preemptive_generation = preemptive_generation
        self.rag_on_turn = os.getenv("RAG_ON_TURN", "true").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.rag_top_k = int(os.getenv("RAG_TOP_K", "2"))
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
        query = (new_message.text_content or "").strip()
        if (
            not self.rag_on_turn
            or self.vector_store is None
            or not query
            or len(query.split()) < 2
        ):
            return

        # RAG injection is faster than a tool call
        # (~50 ms local search + 500 ms LLM ≈ 550 ms vs 1,900 ms LLM→tool→LLM)
        results = await self.vector_store.search(query, top_k=self.rag_top_k)
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
    use_silero_vad = os.getenv("USE_SILERO_VAD", "auto").strip().lower()
    turn_detection_mode = os.getenv("TURN_DETECTION_MODE", "stt").strip().lower()
    enable_vad = use_silero_vad in {"1", "true", "yes", "on"} or (
        use_silero_vad == "auto" and turn_detection_mode != "stt"
    )
    proc.userdata["vad"] = silero.VAD.load() if enable_vad else None


async def entrypoint(ctx: JobContext):
    preload_max_pages = int(os.getenv("KNOWLEDGE_PRELOAD_MAX_PAGES", "100"))
    force_refresh = os.getenv("KNOWLEDGE_FORCE_REFRESH", "false").strip().lower() in {"1", "true", "yes"}

    _bnc = getattr(noise_cancellation, "BVCTelephony", None)
    nc = _bnc() if _bnc and os.getenv("TELEPHONY_MODE", "auto") != "off" else noise_cancellation.BVC()

    # Preemptive generation is disabled by default because RAG injection
    # (~50 ms local search + ~500 ms LLM first-token) is significantly faster
    # than the LLM→function-call→LLM round-trip (~1,900 ms) that becomes
    # necessary when preemptive skips RAG context injection.
    preemptive_generation = os.getenv("PREEMPTIVE_GENERATION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    # AssemblyAI universal-streaming has its own phrase-endpointing model.
    # Set min_endpointing_delay=0 so LiveKit doesn't add extra latency on top.
    # Use max_turn_silence (ms) to control how long AssemblyAI waits for silence.
    min_endpointing_delay = float(os.getenv("MIN_ENDPOINTING_DELAY", "0"))
    max_endpointing_delay = float(os.getenv("MAX_ENDPOINTING_DELAY", "0.5"))
    # Slightly higher default helps merge brief pauses ("account" ... "i" ...
    # "want to open isa account") into one turn with minimal added delay.
    min_consecutive_speech_delay = float(
        os.getenv("MIN_CONSECUTIVE_SPEECH_DELAY", "0.08")
    )
    max_tool_steps = int(os.getenv("MAX_TOOL_STEPS", "1"))
    # AssemblyAI default max_turn_silence is 1280 ms. 1200 ms prevents mid-sentence
    # cuts (e.g. "i want to know about the junior essay [pause] account") while
    # still ending turns faster than the 1280 ms API default.
    max_turn_silence_ms = int(os.getenv("MAX_TURN_SILENCE_MS", "1200"))

    turn_detection_mode = os.getenv("TURN_DETECTION_MODE", "stt").strip().lower()
    if turn_detection_mode == "multilingual":
        # Lazy import: importing at top-level unconditionally registers the
        # lk_end_of_utterance_multilingual inference runner, which spins up a
        # separate subprocess and adds ~5 s of startup latency per job even
        # when the model is not used.
        from livekit.plugins.turn_detector.multilingual import MultilingualModel
        turn_detection = MultilingualModel()
    elif turn_detection_mode in {"vad", "stt", "realtime_llm", "manual"}:
        turn_detection = turn_detection_mode
    else:
        turn_detection = "stt"

    session = AgentSession(
        stt=inference.STT(
            model="assemblyai/universal-streaming",
            language="en",
            extra_kwargs={"max_turn_silence": max_turn_silence_ms},
        ),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=turn_detection,
        vad=ctx.proc.userdata.get("vad"),
        min_endpointing_delay=min_endpointing_delay,
        max_endpointing_delay=max_endpointing_delay,
        min_consecutive_speech_delay=min_consecutive_speech_delay,
        max_tool_steps=max_tool_steps,
        preemptive_generation=preemptive_generation,
    )

    assistant = Assistant(preemptive_generation=preemptive_generation)

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
