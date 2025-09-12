import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

load_dotenv()

EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")



def send_reset_email(to_email, token):
    reset_link = f"{BASE_URL}/reset?token={token}"
    msg = EmailMessage()
    msg['Subject'] = "TalkHeal Password Reset"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg.set_content(
        f"Use this link to reset your password:\n\n"
        f"{reset_link}\n\n"
        f"This link will expire in 15 mins."
    )
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True, "Reset email sent successfully!"
    except Exception as e:
        return False, str(e)
