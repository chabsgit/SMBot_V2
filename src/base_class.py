from logger_config import get_logger


class BaseClass:
    """Base class that provides logger to all inherited classes"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def log_info(self, message):
        """Log info message with class context"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")
    
    def log_error(self, message):
        """Log error message with class context"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")
    
    def log_warning(self, message):
        """Log warning message with class context"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")
    
    def log_debug(self, message):
        """Log debug message with class context"""
        self.logger.debug(f"[{self.__class__.__name__}] {message}")
