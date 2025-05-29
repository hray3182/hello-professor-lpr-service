from fastapi import Body
from ..core_setup import exit_router # Use the router instance from core_setup
from ..webrtc_handlers import handle_offer_logic, handle_capture_logic

@exit_router.post("/offer")
async def offer_exit(params: dict = Body(...)):
    return await handle_offer_logic(params, "exit")

@exit_router.get("/capture")
async def capture_exit():
    return await handle_capture_logic("exit") 