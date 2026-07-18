"""DonationReminder — §GRATITUDE

Zeigt nach jeder erfolgreichen Restaurierung eine freundliche
Spenden-Erinnerung mit PayPal-Link.
"""

from __future__ import annotations

import logging
import webbrowser

logger = logging.getLogger(__name__)

PAYPAL_URL = "https://www.paypal.com/donate?hosted_button_id=AURIKDONATE"
PAYPAL_EMAIL = "michael.arnold2307@gmail.com"
PAYPAL_FALLBACK = "https://www.paypal.com/paypalme/aurikdev"

_MESSAGES = [
    "🎵 Dein Song wurde erfolgreich restauriert!",
    "",
    "Aurik ist das Ergebnis tausender Stunden Entwicklungsarbeit —",
    "kostenlos, werbefrei und mit Weltspitze-Qualität.",
    "",
    "Wenn Dir Aurik geholfen hat, freue ich mich über Deine Unterstützung:",
    f"👉 {PAYPAL_URL}",
    "",
    "Jeder Betrag hilft, Aurik weiter zu verbessern. Danke! ❤️",
    "",
    "— Michael (Aurik-Entwickler)",
]


def show_reminder(quality_score: float = 0.0) -> str:
    """Zeigt Spenden-Erinnerung mit personalisiertem Qualitäts-Hinweis."""

    if quality_score > 0.8:
        personal = "🌟 Hervorragende Restaurierung! Aurik hat hier ganze Arbeit geleistet."
    elif quality_score > 0.5:
        personal = "✨ Gute Restaurierung! Aurik konnte den Klang spürbar verbessern."
    else:
        personal = "🎧 Dein Song wurde restauriert. Aurik hat sein Bestes gegeben."

    lines = [personal] + _MESSAGES

    message = "\n".join(lines)
    logger.info(message)
    return message


def open_donation_link() -> bool:
    """Öffnet den Spenden-Link im Browser. Verifiziert via Fallback."""
    try:
        webbrowser.open(PAYPAL_URL)
        logger.debug("Donation link opened: %s", PAYPAL_URL)
        return True
    except Exception:
        try:
            webbrowser.open(PAYPAL_FALLBACK)
            return True
        except Exception:
            return False


def get_donation_info() -> dict:
    """Gibt Spenden-Informationen als Dict zurück."""
    return {
        "url": PAYPAL_URL,
        "fallback": PAYPAL_FALLBACK,
        "email": PAYPAL_EMAIL,
    }
