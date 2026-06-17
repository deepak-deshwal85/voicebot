import asyncio
import logging
import os
import re
from collections.abc import Awaitable

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    StopResponse,
    WorkerOptions,
    cli,
    function_tool,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from utils.config import (
    AgentConfig,
    load_agent_config,
    load_knowledge_preload_settings,
    load_worker_settings,
)
from utils.knowledge_store import KnowledgeStore
from utils.search_query import is_valid_search_query
from utils.tenant import (
    coerce_phone_value,
    is_telephony_room,
    resolve_client_id_for_phone,
    wait_for_sip_trunk_phone,
)

logger = logging.getLogger("agent")

load_dotenv(".env")
load_dotenv(".env.local", override=True)

WORKER_SETTINGS = load_worker_settings()
PRELOAD_SETTINGS = load_knowledge_preload_settings()

_store_cache: dict[str, KnowledgeStore] = {}

_GREETING_WORDS = {
    "good",
    "hello",
    "hey",
    "hi",
    "morning",
    "afternoon",
    "evening",
    "there",
}
_QUESTION_STARTERS = {
    "can",
    "could",
    "do",
    "does",
    "how",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
_IGNORE_TRANSCRIPT_PHRASES = (
    "put your call on hold",
    "stay on the line",
    "please hold",
    "please stay on the line",
    "your call is important",
    "all representatives are busy",
    "estimated wait time",
    "press 1",
    "press one",
    "this call may be recorded",
    "for quality assurance",
)


def _is_greeting_only(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return False
    if words[0] in _QUESTION_STARTERS:
        return False
    return all(word in _GREETING_WORDS for word in words)


def _should_ignore_transcript(text: str) -> bool:
    normalized = text.lower().strip()
    if len(normalized) < 3:
        return True
    return any(phrase in normalized for phrase in _IGNORE_TRANSCRIPT_PHRASES)


class Assistant(Agent):
    def __init__(self, config: AgentConfig, *, is_telephony: bool = False) -> None:
        super().__init__(instructions=config.instructions)
        self.config = config
        self.is_telephony = is_telephony
        self.vector_store: KnowledgeStore | None = None
        self.is_first_interaction = True
        self._greeting_spoken = False

    async def _get_store(self) -> KnowledgeStore | None:
        if self.vector_store is not None:
            return self.vector_store

        cached = _store_cache.get(self.config.client_id)
        if cached is not None:
            self.vector_store = cached
            return cached

        store = KnowledgeStore(self.config)
        try:
            await asyncio.wait_for(store.initialize(), timeout=30)
        except asyncio.TimeoutError:
            logger.error(
                "Knowledge index setup timed out for %s", self.config.client_id
            )
            return None

        _store_cache[self.config.client_id] = store
        self.vector_store = store
        return store

    async def preload_knowledge(
        self, *, preload_pdf: bool, preload_website: bool
    ) -> None:
        """Warm configured knowledge indexes at call start without blocking the greeting."""
        if not preload_pdf and not preload_website:
            return

        try:
            store = await self._get_store()
            if store is None:
                return

            names: list[str] = []
            coros: list[Awaitable[bool]] = []
            if preload_pdf:
                names.append("pdf")
                coros.append(store.preload_pdf())
            if preload_website:
                names.append("website")
                coros.append(store.preload_website())

            results = await asyncio.gather(*coros, return_exceptions=True)
            for name, result in zip(names, results, strict=True):
                if isinstance(result, Exception):
                    logger.exception(
                        "%s preload failed for %s", name, self.config.client_id
                    )
                    continue
                if not result:
                    continue

                chunk_count = (
                    len(store.pdf.documents)
                    if name == "pdf"
                    else len(store.website.documents)
                )
                logger.info(
                    "Preloaded %s knowledge for %s (%s chunks)",
                    name,
                    self.config.client_id,
                    chunk_count,
                )
        except Exception:
            logger.exception("Knowledge preload failed for %s", self.config.client_id)

    def _speak_greeting(self) -> None:
        if self._greeting_spoken:
            return
        self._greeting_spoken = True
        self.is_first_interaction = False
        self.session.say(
            self.config.initial_greeting,
            allow_interruptions=True,
        )

    async def on_enter(self) -> None:
        if self.is_telephony:
            self._speak_greeting()

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        query = new_message.text_content

        if not query or _should_ignore_transcript(query):
            logger.info("Ignoring non-user transcript: %r", query)
            raise StopResponse()

        if self.is_first_interaction and _is_greeting_only(query):
            self._speak_greeting()
            raise StopResponse()

        if self.is_first_interaction:
            self.is_first_interaction = False

    async def _search(
        self,
        query: str,
        *,
        source: str,
        search_fn,
    ) -> str:
        if not is_valid_search_query(query):
            logger.info("Skipping %s search for invalid query: %r", source, query)
            return self.config.invalid_search_query_message

        store = await self._get_store()
        if store is None:
            return self.config.knowledge_not_ready_message

        logger.info("%s search for: %s", source, query)
        results = await search_fn(store, query)
        if not results:
            return self.config.no_results_message
        return results

    @function_tool()
    async def search_website_docs(self, context: RunContext, query: str) -> str:
        """Search the website for investments, funds, ETFs, pensions, fees, and company information.

        Args:
            query: Short search query based on the user's question.
        """
        return await self._search(
            query,
            source="Website",
            search_fn=lambda store, q: store.search_website(q),
        )

    @function_tool()
    async def search_document_library(self, context: RunContext, query: str) -> str:
        """Search the resume PDF for education, skills, experience, and projects.

        Args:
            query: Short search query based on the user's question.
        """
        return await self._search(
            query,
            source="PDF",
            search_fn=lambda store, q: store.search_pdf(q),
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def resolve_session_config(ctx: JobContext) -> tuple[AgentConfig, str | None]:
    """Load tenant client config from SIP trunk phone number or local fallback."""
    await ctx.connect()

    trunk_phone = coerce_phone_value(os.getenv("TENANT_PHONE_OVERRIDE"))
    if not trunk_phone and is_telephony_room(ctx.room.name):
        trunk_phone = coerce_phone_value(await wait_for_sip_trunk_phone(ctx.room))
    elif not trunk_phone:
        logger.info(
            "Non-telephony room %s; using default client %s",
            ctx.room.name,
            WORKER_SETTINGS.default_client_id,
        )

    try:
        client_id = resolve_client_id_for_phone(trunk_phone)
        config = load_agent_config(client_id=client_id)
    except ValueError as exc:
        if trunk_phone:
            logger.error("Tenant resolution failed: %s", exc)
            raise
        logger.warning(
            "No SIP trunk phone; falling back to default client %s",
            WORKER_SETTINGS.default_client_id,
        )
        config = load_agent_config(client_id=WORKER_SETTINGS.default_client_id)

    if trunk_phone:
        logger.info(
            "SIP session client=%s trunk_phone=%s room=%s",
            config.client_id,
            trunk_phone,
            ctx.room.name,
        )

    return config, trunk_phone


async def entrypoint(ctx: JobContext):
    agent_config, trunk_phone = await resolve_session_config(ctx)

    ctx.log_context_fields = {
        "room": ctx.room.name,
        "client": agent_config.client_id,
        "trunk_phone": trunk_phone or "none",
    }

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        min_endpointing_delay=0.5,
        max_endpointing_delay=1.8,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    assistant = Assistant(config=agent_config, is_telephony=bool(trunk_phone))
    if PRELOAD_SETTINGS.pdf or PRELOAD_SETTINGS.website:
        assistant._knowledge_preload_task = asyncio.create_task(
            assistant.preload_knowledge(
                preload_pdf=PRELOAD_SETTINGS.pdf,
                preload_website=PRELOAD_SETTINGS.website,
            )
        )

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=WORKER_SETTINGS.agent_name,
            initialize_process_timeout=120.0,
        )
    )
