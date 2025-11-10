import logging
import asyncio

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
    ChatContext,
    ChatMessage,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from utils.knowledge_store import KnowledgeStore

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant that provides information about Fidelity UK.
            The user is interacting with you via voice, even if you perceive the conversation as text.
            You can provide information about Fidelity UK's services, investment options, financial products, and company information.
            Your responses should be professional, concise, and to the point, without any complex formatting or punctuation.
            If asked about something not available on the Fidelity UK website, politely state that you can only provide information from their official website.""",
        )
        self.vector_store = None
        self.is_first_interaction = True
        self._initializing = False
        
    async def _ensure_vector_store(self) -> bool:
        """Ensure vector store is initialized. Returns True if successful."""
        if self.vector_store is not None:
            return True
            
        if self._initializing:
            # Already initializing, wait a bit
            return False
            
        self._initializing = True
        try:
            logger.info("Initializing vector store...")
            self.vector_store = KnowledgeStore()
            await asyncio.wait_for(self.vector_store.initialize(), timeout=60)
            logger.info("Vector store initialized successfully.")
            return True
        except asyncio.TimeoutError:
            logger.error("Initialization timed out.")
            self.vector_store = None
            return False
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            self.vector_store = None
            return False
        finally:
            self._initializing = False
    
    async def initialize(self) -> None:
        """Initialize the agent."""
        await super().initialize()
        # Don't initialize vector store here to avoid timeout during worker startup
        # It will be initialized lazily when first needed
        
    async def on_session_started(self, session: AgentSession) -> None:
        """Called when a new session starts."""
        await super().on_session_started(session)
        
        # Try to initialize vector store if not already initialized
        await self._ensure_vector_store()
        
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        """Handle user messages and provide responses about the resume."""
        # Ensure vector store is initialized
        if not await self._ensure_vector_store():
            logger.error("Vector store not initialized")
            turn_ctx.add_message(
                role="assistant",
                content="I'm sorry, but my knowledge base is not ready yet. Please try again in a moment.",
            )
            return

        try:
            # First interaction message
            if self.is_first_interaction:
                self.is_first_interaction = False
                turn_ctx.add_message(
                    role="assistant",
                    content="Hello! I can help you learn about Fidelity UK. What would you like to know about their investment services, financial products, or company information?"
                )
                return
                
            # Regular query handling
            query = new_message.text_content
            if not query:
                return
                
            results = await self.vector_store.search_with_fallback(query, top_k=3)
            if not results:
                logger.info(f"No relevant information found for query: {query}")
                turn_ctx.add_message(
                    role="assistant",
                    content="I don't have that specific information on the Fidelity UK website. You can ask about their investment services, financial products, or company information."
                )
                return

            response = "Based on Fidelity UK's website:\n"
            for doc in results:
                if doc.get('text', '').strip():
                    response += f"• {doc['text']}\n"

            turn_ctx.add_message(
                role="assistant",
                content=response.strip()
            )
        except Exception as e:
            logger.error(f"Error handling user message: {str(e)}", exc_info=True)
            turn_ctx.add_message(
                role="assistant",
                content="I apologize, but I encountered an error while accessing the Fidelity UK website information. Please try asking your question again."
            )

    from livekit.agents import function_tool, RunContext

    @function_tool()
    async def update_website_content(self, context: RunContext, max_pages: int = 10):
        """Update the knowledge base with fresh content from the Fidelity UK website.

        Args:
            max_pages: Maximum number of pages to scrape (default: 10)
        """
        logger.info("Updating content from Fidelity UK website...")
        await self.vector_store.scrape_website(max_pages=max_pages)
        return "Website content updated successfully."

    @function_tool()
    async def search_website(self, context: RunContext, query: str, top_k: int = 3):
        """Search for information from Fidelity UK website if not found in resume.

        Args:
            query: The search query
            top_k: Number of top results to return (default: 3)
        """
        logger.info(f"Searching for information: {query}")
        results = await self.vector_store.search_with_fallback(query, top_k)
        if not results:
            return "I couldn't find specific information about that on the Fidelity UK website. Please try rephrasing your question or ask about their investment services, financial products, or company information."

        response = "Here's what I found on Fidelity UK's website:\n\n"
        for result in results:
            response += f"• {result['text']}\n"

        response += "\nPlease note: This information is for general reference. For specific financial advice, please consult with a financial professional."
        return response
        
    @function_tool()
    async def refresh_website_content(self, context: RunContext):
        """Refresh the website content in the vector store."""
        # Ensure vector store is initialized
        if not await self._ensure_vector_store():
            logger.error("Vector store not initialized")
            return "I'm sorry, but my knowledge base is not ready yet. Please try again later."

        try:
            logger.info("Refreshing website content...")
            await self.vector_store.scrape_website(max_pages=20)
            logger.info("Website content refreshed successfully.")
            return "Website content has been refreshed successfully."
        except Exception as e:
            logger.error(f"Error refreshing website content: {e}")
            return "An error occurred while refreshing the website content. Please try again later."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Disable preemptive generation to reduce initialization complexity
        preemptive_generation=False,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # Increase initialization timeout
            initialize_process_timeout=120.0,
        )
    )
