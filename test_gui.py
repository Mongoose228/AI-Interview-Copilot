import interview_copilot.gui.app as app_module
from PySide6.QtCore import QTimer
from interview_copilot.gui.app import start_gui
from PySide6.QtWidgets import QApplication


def run_test():
    # We will patch sys.exit so it doesn't kill our runner immediately?
    # Or just let it run.
    def kill_later():
        app = QApplication.instance()
        if app:
            app.quit()

    # Start a timer in a separate thread? No, QApplication must run timers in main thread.
    # We can inject a timer into the QApplication after it starts.
    pass


original_exec = app_module.QApplication.exec


def mock_exec(self):
    QTimer.singleShot(3000, self.quit)
    return original_exec(self)


app_module.QApplication.exec = mock_exec

start_gui()
