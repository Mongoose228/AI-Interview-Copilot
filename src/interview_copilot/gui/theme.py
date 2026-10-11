"""Shared visual tokens and stylesheets for the Interview Copilot GUI.

Visual direction: "Instrument Glass" — a translucent slate overlay with a
teal signal accent, clear hierarchy, and no emoji chrome.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

FONT_UI = "'IBM Plex Sans', 'Segoe UI Variable Text', 'Segoe UI'"
FONT_MONO = "'Cascadia Code', 'IBM Plex Mono', 'Consolas'"
FONT_SIZE = 13

# Slate glass surfaces (intentional translucent overlay, not flat black)
BG_APP = "rgba(18, 28, 36, 0.78)"
BG_PANEL = "rgba(28, 42, 54, 0.55)"
BG_PANEL_HOVER = "rgba(36, 54, 68, 0.72)"
BG_INPUT = "rgba(12, 20, 28, 0.88)"
BG_ELEVATED = "#1a2834"

BORDER_SUBTLE = "rgba(120, 150, 170, 0.22)"
BORDER_FOCUS = "rgba(45, 212, 191, 0.55)"
BORDER_WARN = "rgba(232, 168, 72, 0.45)"

# Accent: teal signal (not purple / indigo)
ACCENT = "#2DD4BF"
ACCENT_DIM = "rgba(45, 212, 191, 0.18)"
ACCENT_TEXT = "#5EEAD4"

TEXT_PRIMARY = "#E8EEF2"
TEXT_SECONDARY = "#9AAEBC"
TEXT_MUTED = "#6B8294"
TEXT_AI = "#5EEAD4"
TEXT_AI_BOLD = "#2DD4BF"
TEXT_RU = "#7EB8D4"
TEXT_AI_RU = "#8FCB9B"
TEXT_WARN = "#E8A848"
TEXT_ERROR = "#E86A5D"
TEXT_PROVISIONAL = "#7A8F9E"

RADIUS_SM = "6px"
RADIUS_MD = "10px"
RADIUS_LG = "14px"


def app_stylesheet() -> str:
    return f"""
        QWidget {{
            color: {TEXT_PRIMARY};
            font-family: {FONT_UI};
            font-size: {FONT_SIZE}px;
            background: transparent;
        }}
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        QScrollBar:vertical {{
            border: none;
            background: rgba(12, 20, 28, 0.45);
            width: 8px;
            margin: 2px 0;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(120, 150, 170, 0.35);
            min-height: 28px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: rgba(45, 212, 191, 0.45);
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 0;
        }}
        QToolTip {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_SUBTLE};
            padding: 6px 8px;
            font-size: 12px;
        }}
    """


def main_container_style() -> str:
    return f"""
        QWidget#MainContainer {{
            background-color: {BG_APP};
            border-radius: {RADIUS_LG};
            border: 1px solid {BORDER_SUBTLE};
        }}
    """


def title_style() -> str:
    return (
        f"color: {TEXT_PRIMARY}; font-family: {FONT_MONO}; font-size: 15px; "
        f"font-weight: 600; letter-spacing: 0.3px;"
    )


def caption_style() -> str:
    return (
        f"color: {TEXT_MUTED}; font-family: {FONT_MONO}; font-size: 11px; "
        f"font-weight: 500;"
    )


def tag_html(text: str, color: str = TEXT_MUTED) -> str:
    return (
        f"<span style=\"color:{color}; font-family:{FONT_MONO}; "
        f"font-weight:600;\">{text}</span>"
    )


def status_style_idle() -> str:
    return f"color: {TEXT_SECONDARY}; font-family: {FONT_MONO}; font-size: 11px;"


def status_style_warn() -> str:
    return f"color: {TEXT_WARN}; font-family: {FONT_MONO}; font-size: 11px;"


def status_style_error() -> str:
    return (
        f"color: {TEXT_ERROR}; font-family: {FONT_MONO}; font-size: 11px; "
        f"font-weight: 600;"
    )


def button_style(*, primary: bool = False) -> str:
    if primary:
        return f"""
            QPushButton {{
                background-color: {ACCENT_DIM};
                color: {ACCENT_TEXT};
                border: 1px solid {BORDER_FOCUS};
                border-radius: {RADIUS_SM};
                padding: 5px 12px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: rgba(45, 212, 191, 0.32);
                color: {TEXT_PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: rgba(45, 212, 191, 0.22);
            }}
            QPushButton:focus {{
                border: 1px solid {ACCENT};
            }}
        """
    return f"""
        QPushButton {{
            background-color: rgba(255, 255, 255, 0.06);
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: {RADIUS_SM};
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: rgba(255, 255, 255, 0.12);
            color: {TEXT_PRIMARY};
            border: 1px solid rgba(120, 150, 170, 0.4);
        }}
        QPushButton:pressed {{
            background-color: rgba(255, 255, 255, 0.04);
        }}
        QPushButton:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
    """


def close_button_style() -> str:
    return f"""
        QPushButton {{
            background-color: rgba(255, 255, 255, 0.06);
            color: {TEXT_MUTED};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: {RADIUS_SM};
            font-size: 13px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: rgba(232, 106, 93, 0.35);
            color: {TEXT_PRIMARY};
            border: 1px solid rgba(232, 106, 93, 0.55);
        }}
        QPushButton:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
    """


def combo_style() -> str:
    return f"""
        QComboBox {{
            background-color: {BG_INPUT};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: {RADIUS_SM};
            padding: 4px 8px;
            font-size: 12px;
            min-height: 26px;
        }}
        QComboBox:hover {{
            border: 1px solid rgba(120, 150, 170, 0.4);
        }}
        QComboBox:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 22px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            selection-background-color: {ACCENT_DIM};
            selection-color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_SUBTLE};
            outline: none;
            padding: 4px;
        }}
    """


def result_card_style() -> str:
    return f"""
        ResultWidget {{
            background-color: {BG_PANEL};
            border-radius: {RADIUS_MD};
            border: 1px solid {BORDER_SUBTLE};
            margin-bottom: 8px;
        }}
        ResultWidget:hover {{
            background-color: {BG_PANEL_HOVER};
            border: 1px solid rgba(120, 150, 170, 0.35);
        }}
        QLabel {{
            background: transparent;
        }}
    """


def copy_button_style() -> str:
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {TEXT_MUTED};
            border: 1px solid transparent;
            border-radius: {RADIUS_SM};
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            color: {ACCENT_TEXT};
            background-color: {ACCENT_DIM};
            border: 1px solid {BORDER_FOCUS};
        }}
        QPushButton:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
    """


def divider_style() -> str:
    return f"background-color: {BORDER_SUBTLE}; max-height: 1px; border: none;"


def empty_state_style() -> str:
    return f"color: {TEXT_MUTED}; font-size: 13px; padding: 28px 16px;"


def privacy_banner_style() -> str:
    return f"""
        PrivacyBanner {{
            background-color: rgba(42, 34, 18, 0.82);
            border: 1px solid {BORDER_WARN};
            border-radius: {RADIUS_MD};
            margin-bottom: 8px;
        }}
        QLabel {{ background: transparent; }}
    """


def privacy_button_style() -> str:
    return f"""
        QPushButton {{
            background-color: rgba(232, 168, 72, 0.18);
            color: {TEXT_WARN};
            border: 1px solid {BORDER_WARN};
            border-radius: {RADIUS_SM};
            padding: 6px 16px;
            font-weight: 600;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: rgba(232, 168, 72, 0.32);
            color: {TEXT_PRIMARY};
        }}
        QPushButton:focus {{
            border: 1px solid {TEXT_WARN};
        }}
    """


def settings_dialog_style() -> str:
    return f"""
        QDialog {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            font-family: {FONT_UI};
            font-size: {FONT_SIZE}px;
        }}
        QLabel {{
            color: {TEXT_SECONDARY};
        }}
        QLabel#SectionHeader {{
            color: {ACCENT_TEXT};
            font-family: {FONT_MONO};
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.6px;
            padding-top: 8px;
            padding-bottom: 2px;
        }}
        QLabel#HintLabel {{
            color: {TEXT_MUTED};
            font-size: 11px;
            padding-top: 4px;
        }}
        QLineEdit, QComboBox {{
            background-color: {BG_INPUT};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: {RADIUS_SM};
            padding: 6px 8px;
            min-height: 22px;
        }}
        QLineEdit:focus, QComboBox:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
        QCheckBox {{
            color: {TEXT_SECONDARY};
            spacing: 8px;
            padding: 3px 0;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {BORDER_SUBTLE};
            background: {BG_INPUT};
        }}
        QCheckBox::indicator:checked {{
            background: {ACCENT};
            border: 1px solid {ACCENT};
        }}
        QDialogButtonBox QPushButton {{
            background-color: rgba(255, 255, 255, 0.06);
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: {RADIUS_SM};
            padding: 6px 16px;
            min-width: 72px;
            font-size: 12px;
        }}
        QDialogButtonBox QPushButton:hover {{
            background-color: rgba(255, 255, 255, 0.12);
            color: {TEXT_PRIMARY};
        }}
        QDialogButtonBox QPushButton:default {{
            background-color: {ACCENT_DIM};
            color: {ACCENT_TEXT};
            border: 1px solid {BORDER_FOCUS};
            font-weight: 600;
        }}
    """
