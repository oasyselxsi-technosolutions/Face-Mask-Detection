import smtplib
import ssl
from email.message import EmailMessage

class EmailSender:
    def __init__(self, smtp_server, smtp_port, sender_email, sender_password, receiver_emails):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        # Accept both string and list for receiver_emails
        if isinstance(receiver_emails, str):
            self.receiver_emails = [email.strip() for email in receiver_emails.split(',')]
        else:
            self.receiver_emails = receiver_emails

    def send_email_with_attachment(self, subject, body, attachment_path):
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(self.receiver_emails)
        msg.set_content(body)

        with open(attachment_path, 'rb') as f:
            file_data = f.read()
            file_name = attachment_path.split('/')[-1]
        msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

        context = ssl.create_default_context()
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls(context=context)
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)