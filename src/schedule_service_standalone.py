import pytz
import threading
import time
import logging
import logging.handlers
import os
from datetime import datetime
from flask import Flask, request, jsonify

# Standalone logger configuration (no imports)
def setup_logger():
    """Setup logger without importing from other modules"""
    logger = logging.getLogger('SMBot_V2')
    
    # Set log level based on environment
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # File handler with rotation
    log_file = f'logs/smbot_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    
    # Console handler (this will show in Render logs)
    console_handler = logging.StreamHandler()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set root logger level to ensure messages appear
    logging.getLogger().setLevel(logging.INFO)
    
    return logger

app = Flask(__name__)
logger = setup_logger()

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

    flag = GenerateTOTPToken(logger)

    if not flag:
        logger.error("❌ TOTP generation failed")
        exit()

    logger.info("TOTP generated")

    """Run logging task until 3 PM IST"""
    global running_task
    
    logger.info("Starting scheduled task - will run until 3 PM IST")
    
    while running_task:
        try:
            current_time = ist_time()
            current_hour = current_time.hour
            
            # Check if it's 3 PM or later
            if current_hour >= 16:
                logger.info("Reached 3 PM IST - stopping scheduled task")
                break
            
            # Log current timestamp
            timestamp = format_time(current_time)
            logger.info(f"Scheduled task running - Current time: {timestamp}")
            
            # Wait for 1 minute
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in scheduled task: {str(e)}")
            time.sleep(60)
    
    logger.info("Scheduled task completed")
    running_task = False

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
    running_task = True
    task_thread = threading.Thread(target=run_until_3pm)
    task_thread.daemon = True
    task_thread.start()
    
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
    global running_task
    
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
    app.run(host='0.0.0.0', port=port, debug=False)
