import pytz
import threading
import time
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify

# Add src directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from logger_config import get_logger
from VAMLive import VAMLiveBot, VAMLiveEngine   # <-- VAM live module

app = Flask(__name__)
app.config['ENV'] = 'development'
logger = get_logger()

# Global variable to control the running task
running_task = False
task_thread = None


def ist_time():
    """Get current IST time"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)


def format_time(dt):
    """Format datetime for display"""
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")


def run_until_3pm():
    """
    Generate TOTP, then start VAMLiveEngine (blocking loop).
    This runs in a background thread triggered by /schedule.
    """
    from generate_TOTP_Token import GenerateTOTPToken
    flag = GenerateTOTPToken()

    if not flag:
        logger.error("TOTP generation failed")
        return

    logger.info("TOTP generated")

    bot = VAMLiveBot()
    app_id = bot.app_id
    raw_token = bot.access_token

    if not app_id or not raw_token:
        logger.error("app_id or token missing; check fyers_appid.txt / fyers_token.txt")
        return

    # v3 websocket token format: "client_id:access_token" [web:61][web:65]
    ws_access_token = f"{app_id}:{raw_token}"
    engine = VAMLiveEngine(bot, access_token=ws_access_token)

    logger.info("Starting VAMLiveEngine in background thread (blocking loop)")
    engine.start()
    logger.info("VAMLiveEngine.start() returned (engine stopped)")


@app.route('/schedule', methods=['POST'])
def start_schedule():
    """Start the scheduled task (VAM live engine)"""
    global running_task, task_thread

    if running_task:
        logger.warning("Schedule task is already running")
        return jsonify({
            "status": "error",
            "message": "Task is already running"
        }), 400

    current_time = ist_time()
    timestamp = format_time(current_time)

    logger.info(f"Schedule endpoint called at: {timestamp}")

    running_task = True

    def task_wrapper():
        global running_task
        try:
            run_until_3pm()
        except Exception as e:
            logger.error(f"Error in run_until_3pm: {e}")
        finally:
            running_task = False
            logger.info("run_until_3pm finished, running_task set to False")

    task_thread = threading.Thread(target=task_wrapper, daemon=True)
    task_thread.start()

    return jsonify({
        "status": "success",
        "message": f"Schedule started at {timestamp}. VAMLive engine running.",
        "start_time": timestamp
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
    """
    Stop the scheduled task (soft stop).
    This just clears the flag; you need to wire a stop hook into VAMLiveEngine
    if you want a clean shutdown from inside engine.start().
    """
    global running_task, task_thread

    if not running_task:
        return jsonify({
            "status": "error",
            "message": "No active task to stop"
        }), 400

    running_task = False
    logger.info("Schedule task stop requested manually")

    return jsonify({
        "status": "success",
        "message": f"Schedule stop requested at {format_time(ist_time())}"
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
