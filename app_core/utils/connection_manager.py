import logging
from aiortc import RTCPeerConnection

async def close_connection(stream_type: str, streams_data: dict, logger: logging.Logger, specific_pc_to_match: RTCPeerConnection = None):
    """Helper to close a specific WebRTC connection and clean up its resources."""
    stream_info = streams_data.get(stream_type)
    if stream_info and stream_info["pc"]:
        current_pc = stream_info["pc"]
        if specific_pc_to_match and current_pc is not specific_pc_to_match:
            logger.info(f"Not closing pc for {stream_type} as it's not the one that triggered the event (likely a new pc replaced it).")
            return

        if current_pc.signalingState != "closed":
            logger.info(f"Closing peer connection for stream type '{stream_type}'.")
            await current_pc.close()
        
        streams_data[stream_type] = {
            "pc": None,
            "latest_frame": None,
            "video_track": None
        }
        logger.info(f"Cleaned up resources for stream type '{stream_type}'.")
    elif stream_type in streams_data:
        streams_data[stream_type] = { "pc": None, "latest_frame": None, "video_track": None}

async def close_all_connections(streams_data: dict, logger: logging.Logger):
    """Helper to close all active WebRTC connections."""
    logger.info("Closing all active peer connections.")
    for stream_type_key in list(streams_data.keys()): # Iterate over a copy of keys
        await close_connection(stream_type_key, streams_data, logger, specific_pc_to_match=None) 