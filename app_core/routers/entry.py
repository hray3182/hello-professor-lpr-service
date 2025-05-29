from fastapi import Body
from ..core_setup import entry_router # Use the router instance from core_setup
from ..webrtc_handlers import handle_offer_logic, handle_capture_logic

@entry_router.post("/offer")
async def offer_entry(params: dict = Body(...)):
    return await handle_offer_logic(params, "entry")

@entry_router.get("/capture")
async def capture_entry():
    return await handle_capture_logic("entry") 