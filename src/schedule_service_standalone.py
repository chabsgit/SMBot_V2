import pytz
import threading
import time
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify

# Add src directory to Python path for Render
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logger_config import get_logger
from MainFile import TradingBot, LiveEngine

app = Flask(__name__)
app.config['ENV'] = 'development'
logger = get_logger()

# Global variable to control the running task
running_task = None
task_thread = None

def ist_time():
    """Get current IST time"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def format_time(dt):
    """Format datetime for display"""
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")

def run_until_3pm():
    from generate_TOTP_Token import GenerateTOTPToken
    flag = GenerateTOTPToken()

    if not flag:
        logger.error("TOTP generation failed")
        exit()

    logger.info("TOTP generated")

    bot = TradingBot()
    app_id = bot.app_id
    raw_token = bot.access_token
    if not app_id or not raw_token:
        raise RuntimeError("app_id or token missing; check fyers_appid.txt / fyers_token.txt")

    # v3 websocket token format: "client_id:access_token". [web:50]
    ws_access_token = f"{app_id}:{raw_token}"
    engine = LiveEngine(bot, access_token=ws_access_token)
    engine.start()
    logger.info("Scheduled task completed")
    # # Start engine in a separate thread to avoid blocking
    # engine_thread = threading.Thread(target=engine.start)
    # engine_thread.daemon = True
    # engine_thread.start()
    # logger.info("Trading engine started in background thread")

    # """Run logging task until 3 PM IST"""
    # global running_task
    
    # logger.info("Starting scheduled task - will run until 3 PM IST")
    
    # while running_task:
    #     try:
    #         current_time = ist_time()
    #         current_hour = current_time.hour
    #         current_minute = current_time.minute

    #         # Check if it's 3 PM or later
    #         if current_hour >= 6 and current_minute >= 0:
    #             logger.info("Reached 3 PM IST - stopping scheduled task")
    #             engine_thread.stop()
    #             break
            
    #         # Log current timestamp
    #         timestamp = format_time(current_time)
    #         logger.info(f"Trading engine running - Current time: {timestamp}")
            
    #         # Wait for 1 minute
    #         time.sleep(60)
            
    #     except Exception as e:
    #         logger.error(f"Error in scheduled task: {str(e)}")
    #         time.sleep(60)
    
    # logger.info("Scheduled task completed")
    # running_task = False

@app.route('/schedule', methods=['POST'])
def start_schedule():
    """Start the scheduled task"""
    global running_task, task_thread
    
    if running_task:
        logger.warning("Schedule task is already running")
        return jsonify({
            "status": "error", 
            "message": "Task is already running"
        }), 400
    
    # Get current IST time
    current_time = ist_time()
    timestamp = format_time(current_time)
    
    logger.info(f"Schedule endpoint called at: {timestamp}")
    
    # Start the background task
    run_until_3pm()
    # running_task = True
    # task_thread = threading.Thread(target=run_until_3pm)
    # task_thread.daemon = True
    # task_thread.start()
    
    return jsonify({
        "status": "success",
        "message": f"Hello! Schedule started at {timestamp}. Will run until 3 PM IST.",
        "start_time": timestamp,
        "end_time": "15:00:00 IST"
    })

@app.route('/schedule/status', methods=['GET'])
def schedule_status():
    """Check if schedule task is running"""
    global running_task
    
    current_time = format_time(ist_time())
    
    return jsonify({
        "status": "running" if running_task else "stopped",
        "current_time": current_time,
        "message": "Task is active" if running_task else "No active task"
    })

@app.route('/schedule/stop', methods=['POST'])
def stop_schedule():
    """Stop the scheduled task"""
    global running_task, task_thread
    
    if not task_thread.is_alive():
        task_thread.stop()
        running_task = False
    
    if not running_task:
        return jsonify({
            "status": "error",
            "message": "No active task to stop"
        }), 400
    
    running_task = False
    
    logger.info("Schedule task stopped manually")
    
    return jsonify({
        "status": "success",
        "message": f"Schedule stopped at {format_time(ist_time())}"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    
    return jsonify({
        "status": "healthy",
        "service": "Schedule Service",
        "current_time": format_time(ist_time()),
        "task_running": running_task
    })

if __name__ == '__main__':
    logger.info("Starting Schedule Service")
    port = int(os.environ.get('PORT', 5000))
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal - shutting down")
        logger.info("Service stopped")
