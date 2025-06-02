import os
import cv2
import numpy as np
import re
from datetime import datetime
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaStreamError
import aiohttp
import io

# Import the core_setup module itself
from . import core_setup

from .config import (
    DEBUG_SAVE_OCR_IMAGES, DEBUG_IMG_DIR, PROCESSED_WARPED_COLOR_PLATES_DIR,
    YOLO_CONF_FOR_OCR, OCR_FIXED_WIDTH, OCR_FIXED_HEIGHT, LP_FORMAT_REGEX,
    CONFIDENCE_THRESHOLD, EXTERNAL_API_BASE_URL
)
# Import close_connection specifically, assuming connection_manager also uses core_setup globals carefully or has them passed.
# For now, close_connection is modified to accept streams_data and logger, which handle_offer_logic will pass from core_setup.
from .utils.connection_manager import close_connection 

async def process_yolo_detection(img_np, stream_type: str):
    core_setup.logger.info(f"Performing YOLO prediction for '{stream_type}'...")
    results_yolo = core_setup.yolo_model.predict(img_np, conf=YOLO_CONF_FOR_OCR, device=core_setup.device, verbose=False)
    core_setup.logger.info(f"YOLO prediction done for '{stream_type}'.")

    if results_yolo and results_yolo[0].masks is not None and len(results_yolo[0].masks.xy) > 0:
        masks_polygons = results_yolo[0].masks.xy
        confs = results_yolo[0].boxes.conf.cpu().numpy()
        best_idx = np.argmax(confs)
        yolo_confidence = float(confs[best_idx])
        polygon = masks_polygons[best_idx].astype(np.int32)
        core_setup.logger.info(f"Highest YOLO confidence for '{stream_type}': {yolo_confidence:.4f} for a mask.")
        return yolo_confidence, polygon, results_yolo # Return results_yolo for other potential uses if any
    else:
        core_setup.logger.info(f"No objects (masks) detected by YOLO model for '{stream_type}'.")
        return 0.0, None, None

def preprocess_plate_image(img_np, polygon, frame_height, frame_width, stream_type: str, ts_debug: str = None):
    core_setup.logger.info(f"Using simple bounding box crop from YOLO mask for '{stream_type}'.")
    
    mask_for_crop = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(mask_for_crop, [polygon], 255)
    if DEBUG_SAVE_OCR_IMAGES and ts_debug: 
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_01a_simple_crop_mask.png"), mask_for_crop)

    segmented_plate_for_crop = cv2.bitwise_and(img_np, img_np, mask=mask_for_crop)
    if DEBUG_SAVE_OCR_IMAGES and ts_debug: 
            cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_01b_simple_segmented_for_crop.png"), segmented_plate_for_crop)
    
    x_poly, y_poly, w_poly, h_poly = cv2.boundingRect(polygon)
    cropped_color_plate = None
    if w_poly > 0 and h_poly > 0:
        cropped_color_plate = segmented_plate_for_crop[y_poly:y_poly+h_poly, x_poly:x_poly+w_poly]
        core_setup.logger.info(f"Simple bounding box crop for '{stream_type}'. Dimensions: w={w_poly}, h={h_poly}")
        if DEBUG_SAVE_OCR_IMAGES and ts_debug and cropped_color_plate.size > 0:
                cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_01c_simple_color_cropped.png"), cropped_color_plate)
    else:
        core_setup.logger.warning(f"Simple bounding box crop for '{stream_type}' has zero width or height.")
        return None, None # Return None if cropping failed

    if cropped_color_plate is None or cropped_color_plate.size == 0:
        core_setup.logger.warning(f"Cropped plate area for OCR is empty for '{stream_type}'.")
        return None, None # Return None if cropped plate is empty

    core_setup.logger.info(f"Cropped color plate for OCR for '{stream_type}' shape: {cropped_color_plate.shape}")
    unconditional_ts_color = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{stream_type}"
    color_plate_save_path = os.path.join(PROCESSED_WARPED_COLOR_PLATES_DIR, f"color_lp_{unconditional_ts_color}.png")
    cv2.imwrite(color_plate_save_path, cropped_color_plate)
    core_setup.logger.info(f"Saved original color cropped plate for '{stream_type}' to {color_plate_save_path}")

    core_setup.logger.info(f"Preprocessing image for OCR for '{stream_type}': Grayscale -> CLAHE -> Resize -> Binarize.")
    gray_plate = cv2.cvtColor(cropped_color_plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_gray_plate = clahe.apply(gray_plate)
    resized_enhanced_gray_plate = cv2.resize(enhanced_gray_plate, (OCR_FIXED_WIDTH, OCR_FIXED_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    _, binarized_plate = cv2.threshold(resized_enhanced_gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if DEBUG_SAVE_OCR_IMAGES and ts_debug: 
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_01_cropped_color.png"), cropped_color_plate)
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_02_gray.png"), gray_plate)
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_03_clahe_enhanced_gray.png"), enhanced_gray_plate)
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_04_resized_enhanced_gray.png"), resized_enhanced_gray_plate)
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_05_binarized_fixed_size.png"), binarized_plate)
        core_setup.logger.info(f"Saved intermediate OCR preprocessing images to debug dir for '{stream_type}'.")
        debug_final_ocr_input_path = os.path.join(DEBUG_IMG_DIR, f"_99_final_input_for_ocr_{stream_type}.png") 
        cv2.imwrite(debug_final_ocr_input_path, binarized_plate) 
        core_setup.logger.info(f"Updated final input for OCR for '{stream_type}': {debug_final_ocr_input_path}")
    
    return cropped_color_plate, binarized_plate # Return both color and binarized for potential use

def perform_ocr_and_validate(binarized_plate_for_ocr, stream_type: str):
    core_setup.logger.info(f"Performing EasyOCR for '{stream_type}'...")
    ocr_text = "N/A"
    raw_ocr_text_for_response = "N/A"
    format_valid = False
    ocr_message = ""

    try:
        ocr_results_easyocr = core_setup.easyocr_reader_instance.readtext(binarized_plate_for_ocr, detail=1, paragraph=False)
        core_setup.logger.info(f"EasyOCR raw results for '{stream_type}': {ocr_results_easyocr}")
        detected_texts = []
        if ocr_results_easyocr:
            for (bbox, text, prob) in ocr_results_easyocr:
                detected_texts.append((text, prob))
        
        if detected_texts:
            detected_texts.sort(key=lambda x: x[1], reverse=True)
            raw_ocr_text_for_response = detected_texts[0][0]
            ocr_confidence_easyocr = detected_texts[0][1]
            ocr_text = "".join(c for c in raw_ocr_text_for_response.strip().upper().replace(" ", "") if c.isalnum() or c == '-')
            core_setup.logger.info(f"EasyOCR Raw ('{stream_type}'): '{raw_ocr_text_for_response}', Cleaned: '{ocr_text}', Conf: {ocr_confidence_easyocr:.4f}")
            if re.match(LP_FORMAT_REGEX, ocr_text):
                format_valid = True
                ocr_message = f"EasyOCR text ('{stream_type}') '{ocr_text}' matches format."
                core_setup.logger.info(ocr_message)
            else:
                ocr_message = f"EasyOCR text ('{stream_type}') '{ocr_text}' (raw: '{raw_ocr_text_for_response}') does NOT match format."
                core_setup.logger.info(ocr_message)
        else:
            raw_ocr_text_for_response = "NO_TEXT_DETECTED_EASYOCR"
            ocr_text = "NO_TEXT_DETECTED_EASYOCR"
            ocr_message = f"EasyOCR detected no text on the plate for '{stream_type}'."
            core_setup.logger.info(ocr_message)
    except Exception as ocr_e_easyocr:
        error_type = type(ocr_e_easyocr).__name__
        error_msg = str(ocr_e_easyocr)
        ocr_message = f"EasyOCR processing error for '{stream_type}': {error_type} - {error_msg}"
        core_setup.logger.error(ocr_message, exc_info=True)
        ocr_text = "EASYOCR_ERROR"
        raw_ocr_text_for_response = "EASYOCR_ERROR"
        format_valid = False

    return ocr_text, raw_ocr_text_for_response, format_valid, ocr_message

async def call_external_parking_api(stream_type: str, license_plate: str, image_to_send_np):
    api_call_status_message = ""
    api_call_successful = False

    is_success, im_buf_arr = cv2.imencode(".jpg", image_to_send_np)
    if not is_success:
        log_msg = f"Failed to encode image to JPG for external API call for '{stream_type}'."
        core_setup.logger.error(log_msg)
        return False, log_msg

    byte_im = io.BytesIO(im_buf_arr)
    byte_im.name = f"{stream_type}_{license_plate}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"

    form_data = aiohttp.FormData()
    form_data.add_field('licensePlate', license_plate)
    form_data.add_field('image', byte_im, filename=byte_im.name, content_type='image/jpeg')
    
    api_url = f"{EXTERNAL_API_BASE_URL}/api/v1/parking-records/{stream_type}"
    core_setup.logger.info(f"Calling external API: POST {api_url} with plate: {license_plate}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, data=form_data) as response:
                response_text = await response.text()
                if response.status >= 200 and response.status < 300:
                    api_call_successful = True
                    api_call_status_message = f"External API call successful for '{stream_type}' (status: {response.status}): {response_text}"
                    core_setup.logger.info(api_call_status_message)
                else:
                    api_call_status_message = f"External API call failed for '{stream_type}' (status: {response.status}): {response_text}"
                    core_setup.logger.error(api_call_status_message)
    except aiohttp.ClientError as e_api:
        api_call_status_message = f"External API client error for '{stream_type}': {e_api}"
        core_setup.logger.error(api_call_status_message)
    except Exception as e_generic_api:
        api_call_status_message = f"Generic error during external API call for '{stream_type}': {e_generic_api}"
        core_setup.logger.error(api_call_status_message, exc_info=True)
    
    return api_call_successful, api_call_status_message

async def call_exit_check_api(license_plate: str):
    api_call_status_message = ""
    api_call_successful = False
    
    api_url = f"{EXTERNAL_API_BASE_URL}/api/v1/parking-records/exit"
    payload = {"licensePlate": license_plate}
    core_setup.logger.info(f"Calling Exit Check API: POST {api_url} with JSON payload: {payload}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers={"Content-Type": "application/json"}) as response:
                response_text = await response.text()
                if response.status >= 200 and response.status < 300:
                    api_call_successful = True
                    api_call_status_message = f"Exit Check API call successful (status: {response.status}): {response_text}"
                    core_setup.logger.info(api_call_status_message)
                else:
                    api_call_status_message = f"Exit Check API call failed (status: {response.status}): {response_text}"
                    core_setup.logger.error(api_call_status_message)
    except aiohttp.ClientError as e_api:
        api_call_status_message = f"Exit Check API client error: {e_api}"
        core_setup.logger.error(api_call_status_message)
    except Exception as e_generic_api:
        api_call_status_message = f"Generic error during Exit Check API call: {e_generic_api}"
        core_setup.logger.error(api_call_status_message, exc_info=True)
    
    return api_call_successful, api_call_status_message

async def handle_offer_logic(params: dict, stream_type: str):
    offer_sdp = params.get("sdp")
    offer_type = params.get("type")

    if not all([offer_sdp, offer_type]):
        missing = [name for name, val in [("sdp",offer_sdp), ("type",offer_type)] if not val]
        raise HTTPException(status_code=400, detail=f"Missing required parameters in offer: {', '.join(missing)}")
    
    # Access streams_data and logger via core_setup module
    await close_connection(stream_type, core_setup.streams_data, core_setup.logger) 
    core_setup.logger.info(f"Ensured no prior connection for stream_type '{stream_type}'.")

    offer_desc = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    core_setup.logger.info(f"Received Offer SDP for {stream_type}:\n{offer_sdp}")
    
    ice_servers = [
        RTCIceServer(urls='stun:stun.l.google.com:19302'),
        RTCIceServer(urls='stun:stun1.l.google.com:19302'),
        RTCIceServer(urls='stun:stun2.l.google.com:19302'),
        RTCIceServer(urls='stun:stun3.l.google.com:19302'),
        RTCIceServer(urls='stun:stun4.l.google.com:19302'),
        RTCIceServer(urls='stun:stun01.sipphone.com'),
        RTCIceServer(urls='stun:stun.ekiga.net'),
        RTCIceServer(urls='stun:stun.fwdnet.net'),
        RTCIceServer(urls='stun:stun.ideasip.com'),
        RTCIceServer(urls='stun:stun.iptel.org'),
        RTCIceServer(urls='stun:stun.rixtelecom.se'),
        RTCIceServer(urls='stun:stun.schlund.de'),
        RTCIceServer(urls='stun:stunserver.org'),
        RTCIceServer(urls='stun:stun.softjoys.com'),
        RTCIceServer(urls='stun:stun.voiparound.com'),
        RTCIceServer(urls='stun:stun.voipbuster.com'),
        RTCIceServer(urls='stun:stun.voipstunt.com'),
        RTCIceServer(urls='stun:stun.voxgratia.org'),
        RTCIceServer(urls='stun:stun.xten.com')
        # Consider adding a TURN server here if STUN is not enough
        # RTCIceServer(
        #     urls='turn:your-turn-server.com:3478',
        #     username='your_username',
        #     credential='your_password'
        # )
    ]
    configuration = RTCConfiguration(iceServers=ice_servers)
    pc = RTCPeerConnection(configuration=configuration)
    # Access streams_data via core_setup module
    core_setup.streams_data[stream_type] = {"pc": pc, "latest_frame": None, "video_track": None}
    core_setup.logger.info(f"New PeerConnection created for stream_type '{stream_type}'.")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        core_setup.logger.info(f"ICE connection state for '{stream_type}' is {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected", "closed"]:
            core_setup.logger.info(f"Peer connection for '{stream_type}' closed/failed. Cleaning up.")
            await close_connection(stream_type, core_setup.streams_data, core_setup.logger, specific_pc_to_match=pc)
    
    @pc.on("track")
    async def on_track(track):
        core_setup.logger.info(f"Track {track.kind} received for stream '{stream_type}'")
        if track.kind == "video":
            # Access streams_data via core_setup module
            if core_setup.streams_data.get(stream_type) and core_setup.streams_data[stream_type]["pc"] is pc:
                core_setup.streams_data[stream_type]["video_track"] = track
            else:
                core_setup.logger.warning(f"Track received for {stream_type}, but PC does not match current. Ignoring track.")
                return
            while True:
                current_stream_info = core_setup.streams_data.get(stream_type)
                if not (current_stream_info and current_stream_info["pc"] is pc and pc.signalingState != "closed"):
                    core_setup.logger.info(f"Stopping frame reception for '{stream_type}' as PC state changed or doesn't match.")
                    break
                try:
                    frame = await track.recv()
                    if core_setup.streams_data.get(stream_type) and core_setup.streams_data[stream_type]["pc"] is pc:
                         core_setup.streams_data[stream_type]["latest_frame"] = frame
                except MediaStreamError:
                    core_setup.logger.info(f"Video track for '{stream_type}' ended or media stream error.")
                    if core_setup.streams_data.get(stream_type) and core_setup.streams_data[stream_type]["pc"] is pc:
                        core_setup.streams_data[stream_type]["latest_frame"] = None
                    break
                except Exception as e_recv:
                    core_setup.logger.error(f"Error receiving frame for '{stream_type}': {e_recv}", exc_info=True)
                    if core_setup.streams_data.get(stream_type) and core_setup.streams_data[stream_type]["pc"] is pc:
                        core_setup.streams_data[stream_type]["latest_frame"] = None
                    break
            core_setup.logger.info(f"Stopped receiving frames from video track for '{stream_type}'.")

        @track.on("ended")
        async def on_ended():
            core_setup.logger.info(f"Track {track.kind} ended for stream '{stream_type}'")
            if track.kind == "video":
                if core_setup.streams_data.get(stream_type) and core_setup.streams_data[stream_type]["pc"] is pc:
                     core_setup.streams_data[stream_type]["latest_frame"] = None
    
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    core_setup.logger.info(f"Created Answer SDP for {stream_type}:\n{answer.sdp}")
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def handle_capture_logic(stream_type: str):
    if core_setup.yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO Model not loaded (checked in handler)")
    if core_setup.easyocr_reader_instance is None:
        core_setup.logger.error(f"Capture endpoint for {stream_type} called but EasyOCR reader is not available (checked in handler).")
        raise HTTPException(status_code=500, detail="EasyOCR reader not initialized or failed to initialize (checked in handler).")

    if stream_type not in core_setup.streams_data or not core_setup.streams_data[stream_type].get("pc"):
        raise HTTPException(status_code=404, detail=f"No active WebRTC connection found for stream_type '{stream_type}'.")

    current_pc_info = core_setup.streams_data[stream_type]
    pc_ref = current_pc_info["pc"]
    latest_frame = current_pc_info["latest_frame"]

    if pc_ref is None or pc_ref.signalingState == "closed" or latest_frame is None:
        raise HTTPException(status_code=404, detail=f"No active video stream or frame found for stream_type '{stream_type}'.")
    
    core_setup.logger.info(f"--- /{stream_type}/capture endpoint called --- (Core Handler)")
    ts_debug = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{stream_type}" if DEBUG_SAVE_OCR_IMAGES else None

    ocr_text = "N/A"
    yolo_confidence = 0.0
    format_valid = False
    best_status = "error"
    message = f"Processing started for '{stream_type}'."
    raw_ocr_text_for_response = "N/A"
    # Store original full car image for entry API call
    original_full_car_image = None 

    try:
        core_setup.logger.info(f"Starting try block in capture for stream '{stream_type}'")
        img_np = latest_frame.to_ndarray(format="bgr24")
        frame_height, frame_width = img_np.shape[:2]
        core_setup.logger.info(f"Frame received for '{stream_type}': {{frame_width}}x{{frame_height}}")
        
        # Store original image for entry API call
        if stream_type == "entry":
            original_full_car_image = img_np.copy()
            if DEBUG_SAVE_OCR_IMAGES and ts_debug:
                cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_00_original_full_car.png"), original_full_car_image)
        
        if DEBUG_SAVE_OCR_IMAGES and ts_debug:
            cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{ts_debug}_00_original_frame.png"), img_np)

        yolo_confidence, polygon, _ = await process_yolo_detection(img_np, stream_type)

        if polygon is None or yolo_confidence < YOLO_CONF_FOR_OCR:
            message = f"YOLO: No plate detected or confidence too low ({yolo_confidence:.2f}) for '{stream_type}'."
            core_setup.logger.info(message)
        else:
            message = f"YOLO: Plate detected (conf: {yolo_confidence:.2f}) for '{stream_type}'. Processing image."
            core_setup.logger.info(message)
            
            cropped_color_plate_for_api, binarized_plate = preprocess_plate_image(img_np, polygon, frame_height, frame_width, stream_type, ts_debug)

            if cropped_color_plate_for_api is None or binarized_plate is None:
                message = f"Image preprocessing failed for '{stream_type}' (e.g., empty crop)."
                core_setup.logger.warning(message)
            else:
                message = f"Preprocessing successful for '{stream_type}'. Performing OCR."
                core_setup.logger.info(message)
                ocr_text, raw_ocr_text_for_response, format_valid, ocr_message = perform_ocr_and_validate(binarized_plate, stream_type)
                message = ocr_message 

                if format_valid and yolo_confidence >= CONFIDENCE_THRESHOLD:
                    if stream_type == "exit":
                        core_setup.logger.info(f"Plate '{ocr_text}' recognized for 'exit'. Calling Exit Check API.")
                        api_success, api_msg = await call_exit_check_api(ocr_text)
                        if api_success:
                            best_status = "ok"
                            message = f"Plate '{ocr_text}' (conf: {yolo_confidence:.2f}) validated for 'exit' by API."
                            core_setup.logger.info(message)
                        else:
                            best_status = "error_exit_api_check"
                            message = f"Plate '{ocr_text}' (conf: {yolo_confidence:.2f}) detected for 'exit', but API check failed: {api_msg}"
                            core_setup.logger.error(message)
                    
                    elif stream_type == "entry":
                        best_status = "ok"
                        message = f"Plate '{ocr_text}' (conf: {yolo_confidence:.2f}) detected for '{stream_type}'."
                        core_setup.logger.info(message)

                        # Use original full car image for entry API call
                        if original_full_car_image is not None:
                            core_setup.logger.info(f"Calling entry API with original full car image for plate '{ocr_text}'.")
                            api_success, api_msg = await call_external_parking_api(stream_type, ocr_text, original_full_car_image)
                            if not api_success:
                                best_status = "error_api_call"
                                core_setup.logger.error(f"Entry API call failed for plate '{ocr_text}': {api_msg}")
                        else:
                            core_setup.logger.error(f"Original full car image not available for entry API call for plate '{ocr_text}'.")
                            best_status = "error_api_call"
                    
                    else: # For other stream types, if any
                        best_status = "ok"
                        message = f"Plate '{ocr_text}' (conf: {yolo_confidence:.2f}) detected for '{stream_type}'."
                        core_setup.logger.info(message)

                elif format_valid: 
                    best_status = "ocr_ok_low_confidence"
                    message = f"Plate '{ocr_text}' format valid, but overall confidence ({yolo_confidence:.2f}) may be low for '{stream_type}'."
                    core_setup.logger.warning(message)
                else: 
                    best_status = "error_ocr"
                    core_setup.logger.warning(f"OCR failed or format invalid for '{stream_type}'. Message: {message}")
        
        core_setup.logger.info(f"Capture logic for '{stream_type}' completed. Status: {best_status}, Message: {message}, Plate: {ocr_text}, YOLO Conf: {yolo_confidence:.4f}, Format Valid: {format_valid}, Raw OCR: {raw_ocr_text_for_response}")

        return JSONResponse(content={
            "status": best_status, "message": message,
            "yolo_confidence": round(yolo_confidence, 4),
            "ocr_text_raw": raw_ocr_text_for_response.strip() if isinstance(raw_ocr_text_for_response, str) else raw_ocr_text_for_response,
            "ocr_text_cleaned": ocr_text, "ocr_format_valid": format_valid,
            "stream_type": stream_type,
            "timestamp": datetime.now().isoformat()
        }, status_code=200)

    except Exception as e_capture:
        import traceback 
        core_setup.logger.error(f"!!! RAW EXCEPTION IN CAPTURE for '{stream_type}': {type(e_capture).__name__} - {str(e_capture)} !!!")
        core_setup.logger.error(f"!!! TRACEBACK START for '{stream_type}' !!!")
        traceback.print_exc()
        core_setup.logger.error(f"!!! TRACEBACK END for '{stream_type}' !!!")
        core_setup.logger.error(f"Capture/prediction error for '{stream_type}': {e_capture}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Capture error for '{stream_type}': {type(e_capture).__name__} - {str(e_capture)}") 