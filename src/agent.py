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
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from utils.config import AgentConfig, load_agent_config, load_worker_settings
from utils.knowledge_store import KnowledgeStore
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


class Assistant(Agent):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__(instructions=config.instructions)
        self.config = config
        self.vector_store: KnowledgeStore | None = None
        self.is_first_interaction = True
        self._initializing = False

    async def _ensure_vector_store(self) -> bool:
        if self.vector_store is not None:
            return True

        if self._initializing:
            return False

        self._initializing = True
        try:
            logger.info(
                "Loading knowledge store for client %s...",
                self.config.client_id,
            )
            self.vector_store = KnowledgeStore(self.config)
            await asyncio.wait_for(self.vector_store.initialize(), timeout=60)
            logger.info("Knowledge store loaded successfully.")
            return True
        except asyncio.TimeoutError:
            logger.error("Knowledge store load timed out.")
            self.vector_store = None
            return False
        except Exception as exc:
            logger.error("Failed to load knowledge store: %s", exc, exc_info=True)
            self.vector_store = None
            return False
        finally:
            self._initializing = False

    async def on_session_started(self, session: AgentSession) -> None:
        await super().on_session_started(session)
        await self._ensure_vector_store()

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        if not await self._ensure_vector_store():
            logger.error("Knowledge store not initialized")
            turn_ctx.add_message(
                role="assistant",
                content=self.config.knowledge_not_ready_message,
            )
            return

        try:
            if self.is_first_interaction:
                self.is_first_interaction = False
                turn_ctx.add_message(
                    role="assistant",
                    content=self.config.initial_greeting,
                )
                return

            query = new_message.text_content
            if not query:
                return

            results = await self.vector_store.search(query, top_k=3)
            if not results:
                logger.info("No relevant information found for query: %s", query)
                turn_ctx.add_message(
                    role="assistant",
                    content=self.config.no_results_message,
                )
                return

            context_lines = []
            for doc in results:
                text = doc.get("text", "").strip()
                if not text:
                    continue
                source = doc.get("source", "knowledge base")
                context_lines.append(f"[{source}] {text}")

            turn_ctx.add_message(
                role="assistant",
                content=(
                    f"Use the following {self.config.website_name} knowledge base "
                    f"context to answer the user concisely:\n"
                    + "\n".join(f"• {line}" for line in context_lines)
                ),
            )
        except Exception as exc:
            logger.error("Error handling user message: %s", exc, exc_info=True)
            turn_ctx.add_message(
                role="assistant",
                content=(
                    f"I apologize, but I encountered an error while accessing "
                    f"{self.config.website_name} information. Please try again."
                ),
            )

    @function_tool()
    async def search_knowledge_base(
        self,
        context: RunContext,
        query: str,
        top_k: int = 3,
    ):
        """Search the pre-built knowledge base containing website and document content.

        Args:
            query: The search query
            top_k: Number of top results to return (default: 3)
        """
        if not await self._ensure_vector_store():
            return self.config.knowledge_not_ready_message

        logger.info("Searching knowledge base for: %s", query)
        results = await self.vector_store.search(query, top_k)
        if not results:
            return self.config.no_results_message

        response = f"Here's what I found for {self.config.website_name}:\n\n"
        for result in results:
            source = result.get("source", "knowledge base")
            response += f"[{source}] {result['text']}\n\n"
        return response.strip()


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

    await session.start(
        agent=Assistant(config=agent_config),
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
