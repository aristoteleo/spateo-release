from .logger import *


class LoggerManager:
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    CRITICAL = logging.CRITICAL
    EXCEPTION = logging.ERROR

    @staticmethod
    def gen_logger(namespace: str):
        return Logger(namespace)

    def __init__(self, namespace: str = "lack", temp_timer_logger: str = "lack-temp-timer-logger"):
        self.set_main_logger_namespace(namespace)
        self.temp_timer_logger = Logger(temp_timer_logger)

    def set_main_logger_namespace(self, namespace: str):
        self.main_logger = self.gen_logger(namespace)
        self.namespace = namespace

    def get_main_logger(self):
        return self.main_logger

    def get_temp_timer_logger(self):
        return self.temp_timer_logger

    def progress_logger(self, generator, logger=None, progress_name="", indent_level=1):
        if logger is None:
            logger = self.get_temp_timer_logger()
        iterator = iter(generator)
        logger.log_time()
        i = 0
        prev_progress_percent = 0
        while i < len(generator):
            i += 1
            new_progress_percent = i / len(generator) * 100
            # report every `interval` percent
            if new_progress_percent - prev_progress_percent > 1 or new_progress_percent >= 100:
                logger.report_progress(
                    count=i, total=len(generator), progress_name=progress_name, indent_level=indent_level
                )
                prev_progress_percent = new_progress_percent
            yield next(iterator)
        logger.finish_progress(progress_name=progress_name, indent_level=indent_level)

    def main_set_level(self, level):
        set_logger_level(self.namespace, level)

    def main_info(self, message, indent_level=1):
        self.main_logger.info(message, indent_level)

    def main_debug(self, message, indent_level=1):
        self.main_logger.debug(message, indent_level)

    def main_warning(self, message, indent_level=1):
        self.main_logger.warning(message, indent_level)

    def main_exception(self, message, indent_level=1):
        self.main_logger.exception(message, indent_level)

    def main_critical(self, message, indent_level=1):
        self.main_logger.critical(message, indent_level)

    def main_tqdm(self, generator, desc="", indent_level=1, logger=None):
        """a TQDM style wrapper for logging something like a loop.
        e.g.
        for item in main_tqdm(alist, desc=""):
            do something

        Parameters
        ----------
        generator : [type]
            same as what you put in tqdm
        desc : str, optional
            description of your progress
        """
        if logger is None:
            logger = self.main_logger
        return self.progress_logger(generator, logger=logger, progress_name=desc, indent_level=indent_level)

    def main_log_time(
        self,
    ):
        self.main_logger.log_time()

    def main_silence(
        self,
    ):
        self.main_logger.setLevel(logging.CRITICAL + 100)

    def main_finish_progress(self, progress_name=""):
        self.main_logger.finish_progress(progress_name=progress_name)

    def main_info_insert_adata(self, key, adata_attr="obsm", indent_level=1, *args, **kwargs):
        self.main_logger.info_insert_adata(key, adata_attr=adata_attr, indent_level=indent_level, *args, **kwargs)

    def main_info_insert_adata_var(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(self, key, "var", indent_level, *args, **kwargs)

    def main_info_insert_adata_uns(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(key, "uns", indent_level, *args, **kwargs)

    def main_info_insert_adata_obsm(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(key, "obsm", indent_level, *args, **kwargs)

    def main_info_insert_adata_obs(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(key, "obs", indent_level, *args, **kwargs)

    def main_info_insert_adata_layer(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(key, "layers", indent_level, *args, **kwargs)

    def main_info_verbose_timeit(self, msg):
        self.main_logger.info(msg)
